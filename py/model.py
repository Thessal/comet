import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaConvEncoder(nn.Module):
    def __init__(self, hidden_dim=64, d_model=256, kernel_size=5):
        super().__init__()
        # Note that currently time index length is 1755 , and only 5 instruments are used for testing. 
        # 1. Temporal Convolution (Shared weights across all instruments)
        # in_channels=1 because we process a single 1D alpha signal per instrument
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=kernel_size)
        
        # 2. Temporal Pooling: Forces variable time_steps into a fixed size of 1
        self.time_pool = nn.AdaptiveMaxPool1d(1)
        
        # 3. Final projection to Transformer dimension
        self.proj = nn.Linear(hidden_dim, d_model)

    def forward(self, alpha_matrix):
        """
        Args:
            alpha_matrix: Shape (batch_size, time_steps, num_instruments) or (time_steps, num_instruments)
        """
        if alpha_matrix.dim() == 2:
            alpha_matrix = alpha_matrix.unsqueeze(0)
            
        batch_size, time_steps, num_instruments = alpha_matrix.shape
        
        # Goal: Apply a 1D Temporal Convolution over the TIME dimension for each instrument independently.
        # PyTorch's Conv1d expects input shape: (N, C, L) where:
        # N = Batch Size, C = Channels, L = Sequence Length (Time)
        #
        # Here, we treat each instrument as its own independent sequence in the batch.
        # So N = batch_size * num_instruments, C = 1 (just the price value), L = time_steps
        
        # 1. Swap time and instrument dimensions to get (batch, instrument, time)
        x = alpha_matrix.transpose(1, 2).contiguous() 
        
        # 2. Merge batch and instrument dimensions to form the 'N' dimension for Conv1d
        x = x.view(batch_size * num_instruments, 1, time_steps)
        
        # 3. Apply Temporal Conv1D
        # Output shape: (batch_size * num_instruments, hidden_dim, time_steps - kernel_size + 1)
        x = F.relu(self.conv1d(x)) 
        
        # 4. Pool across time to get a single vector per instrument
        # Output shape: (batch_size * num_instruments, hidden_dim, 1)
        x = self.time_pool(x)      
        
        # 5. Remove the size-1 time dimension
        # Output shape: (batch_size * num_instruments, hidden_dim)
        x = x.squeeze(-1)          
        
        # 6. Separate batch and instrument dimensions again
        # Output shape: (batch_size, num_instruments, hidden_dim)
        x = x.view(batch_size, num_instruments, -1)
        
        # 7. Pool across instruments (Permutation invariant, handles dynamic universe size)
        # Shape: (batch_size, hidden_dim)
        x = x.mean(dim=1)          
        
        # Project to decoder's expected dimension
        context_vector = self.proj(x) # Shape: (batch_size, d_model)
        return context_vector

class SRDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, nhid=512, nlayers=4, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        
        # 1. The Context Stream (Cross-Attention)
        self.alpha_context_encoder = AlphaConvEncoder(hidden_dim=64, d_model=d_model, kernel_size=5)
        
        # 2. The Target Sequence Stream (Masked Self-Attention)
        # Explicitly renamed from 'input_emb' to 'target_token_emb'
        self.target_token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 500, d_model)) 
        
        # 3. The Decoder Core
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=nhid, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)
        
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.value_linear = nn.Linear(d_model, 1) # Critic value head
        self.dropout = nn.Dropout(dropout)

    def forward(self, shifted_target_tokens, alpha_matrix):
        """
        Args:
            shifted_target_tokens: Tensor of shape (batch_size, seq_len) -> Integer IDs of the SR equation
            alpha_matrix: Tensor of shape (batch_size, num_instruments, time_steps) -> Raw continuous data
        """
        batch_size, seq_len = shifted_target_tokens.size()
        device = shifted_target_tokens.device
        
        # --- STREAM 1: CONTEXT ---
        # Compress the alpha matrix into a fixed condition vector
        # Shape: (batch_size, 1, d_model) -> Acts as the 'Keys' and 'Values' in cross-attention
        memory_context = self.alpha_context_encoder(alpha_matrix).unsqueeze(1)
        
        # --- STREAM 2: AUTOREGRESSIVE TARGET ---
        # Embed the discrete integer tokens into continuous space
        # Shape: (batch_size, seq_len, d_model) -> Acts as the 'Queries' in cross-attention
        tgt_vectors = self.target_token_emb(shifted_target_tokens) * math.sqrt(self.d_model)
        tgt_vectors = tgt_vectors + self.pos_embedding[:, :seq_len, :]
        tgt_vectors = self.dropout(tgt_vectors)
        
        # Prevent target tokens from looking ahead at future tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        
        # --- DECODE ---
        output = self.transformer_decoder(
            tgt=tgt_vectors, 
            memory=memory_context, 
            tgt_mask=causal_mask
        )
        
        logits = self.output_linear(output)
        values = self.value_linear(output).squeeze(-1) # Shape: (batch_size, seq_len)
        return F.log_softmax(logits, dim=-1), values