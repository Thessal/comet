use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::generate_autoregressive_mask,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    tensor::{Tensor, backend::Backend},
};

#[derive(Config, Debug)]
pub struct TransformerModelConfig {
    pub type_vocab_size: usize,
    pub action_vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    #[config(default = 256)]
    pub max_seq_length: usize,
    #[config(default = 8)]
    pub state_size: usize,
}

impl TransformerModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerModel<B> {
        let type_embedding = EmbeddingConfig::new(self.type_vocab_size, self.d_model).init(device);
        let action_embedding = EmbeddingConfig::new(self.action_vocab_size, self.d_model).init(device);

        // A small local transformer encoder to process the 8 state tokens into a cohesive vector
        let state_encoder = TransformerEncoderConfig::new(self.d_model, self.d_ff, self.num_heads, 1) // 1 layer is sufficient for the local state
            .with_dropout(self.dropout)
            .init(device);

        let pos_embedding = EmbeddingConfig::new(self.max_seq_length, self.d_model).init(device);

        let encoder =
            TransformerEncoderConfig::new(self.d_model, self.d_ff, self.num_heads, self.num_layers)
                .with_dropout(self.dropout)
                .init(device);

        // Output predicts the action vocab
        let output_proj = LinearConfig::new(self.d_model, self.action_vocab_size).init(device);

        TransformerModel {
            type_embedding,
            action_embedding,
            state_encoder,
            pos_embedding,
            encoder,
            output_proj,
            state_size: self.state_size,
            d_model: self.d_model,
        }
    }
}

/// Pre-defined standard sizes for the Transformer architecture.
pub enum ModelSize {
    Small,
    Base,
    Large,
}

impl ModelSize {
    pub fn get_config(
        &self,
        type_vocab_size: usize,
        action_vocab_size: usize,
    ) -> TransformerModelConfig {
        match self {
            // ~5M parameters
            ModelSize::Small => {
                TransformerModelConfig::new(type_vocab_size, action_vocab_size, 128, 4, 2, 256)
                    .with_dropout(0.1)
                    .with_max_seq_length(256)
            }

            // ~20M parameters
            ModelSize::Base => {
                TransformerModelConfig::new(type_vocab_size, action_vocab_size, 256, 8, 4, 512)
                    .with_dropout(0.1)
                    .with_max_seq_length(256)
            }

            // ~85M parameters
            ModelSize::Large => {
                TransformerModelConfig::new(type_vocab_size, action_vocab_size, 512, 8, 6, 1024)
                    .with_dropout(0.1)
                    .with_max_seq_length(512)
            }
        }
    }
}

#[derive(Module, Debug)]
pub struct TransformerModel<B: Backend> {
    type_embedding: Embedding<B>,
    action_embedding: Embedding<B>,
    state_encoder: TransformerEncoder<B>,
    pos_embedding: Embedding<B>,
    encoder: TransformerEncoder<B>,
    output_proj: Linear<B>,
    state_size: usize,
    d_model: usize,
}

impl<B: Backend> TransformerModel<B> {
    /// Forward pass processing a sequence of state arrays.
    /// `states` shape: [batch_size, seq_length, state_size]
    /// Output shape: [batch_size, seq_length, action_vocab_size]
    pub fn forward(&self, states: Tensor<B, 3, burn::tensor::Int>) -> Tensor<B, 3> {
        let [batch_size, seq_length, _state_size] = states.dims();
        let device = &states.device();

        // 1. Split PREV_ACTION and TYPES and flatten to 2D
        // prev_action shape: [batch_size * seq_length, 1]
        let prev_action = states.clone()
            .slice([0..batch_size, 0..seq_length, 0..1])
            .reshape([batch_size * seq_length, 1]);
            
        // type_tokens shape: [batch_size * seq_length, state_size - 1]
        let type_tokens = states
            .slice([0..batch_size, 0..seq_length, 1..self.state_size])
            .reshape([batch_size * seq_length, self.state_size - 1]);

        // Embed individually
        let prev_action_emb = self.action_embedding.forward(prev_action); // [batch*seq, 1, d_model]
        let types_emb = self.type_embedding.forward(type_tokens);         // [batch*seq, 7, d_model]
        
        // Re-combine sequences along the explicit state token dimension (dim=1)
        let states_flat = Tensor::cat(vec![prev_action_emb, types_emb], 1); // [batch*seq, 8, d_model]

        // 2. Hierarchical State Encoding (over the 8 local tokens)
        let state_encoded = self.state_encoder.forward(TransformerEncoderInput::new(states_flat));

        // Mean pool across the 8 encoded tokens to yield a single comprehensive state semantic vector
        // Mean pooling over dim 1 yields [batch*seq, 1, d_model]
        let x = state_encoded.mean_dim(1).reshape([batch_size, seq_length, self.d_model]);

        // 3. Add sequence positional encoding
        let pos_indices = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);

        let pos_emb = self.pos_embedding.forward(pos_indices);
        let x = x + pos_emb;

        // 4. Generate causal attention mask for the Sequence Step
        let mask_attn = generate_autoregressive_mask(batch_size, seq_length, device);

        // 5. Causal encoding step processing sequence of complete state vectors
        let input = TransformerEncoderInput::new(x).mask_attn(mask_attn);
        let encoded = self.encoder.forward(input);

        // 6. Predict next action
        self.output_proj.forward(encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::{Tensor, TensorData};

    type Backend = NdArray<f32>;

    #[test]
    fn test_transformer_inference_with_masking() {
        let device = Default::default();
        let type_vocab_size = 10;
        let action_vocab_size = 5; // e.g., 0: Done, 1: Shift, 2: Reduce_A, 3: Reduce_B, 4: Reduce_C

        // 1. Initialize the transformer model gracefully
        let config = ModelSize::Small.get_config(type_vocab_size, action_vocab_size);
        let model = config.init::<Backend>(&device);

        // 2. Create a dummy state prompt sequence
        // Format: [batch_size=1, seq_length=1, state_size=8]
        // Example: [NEXT_UNPROCESSED, STACK_TOP, 0, 0, 0, 0, 0, 0]
        let state_data = TensorData::from([[
            [1, 2, 0, 0, 0, 0, 0, 0], // Current state at t=0
        ]]);
        let states = Tensor::<Backend, 3, burn::tensor::Int>::from_data(
            state_data.convert::<i32>(),
            &device,
        );

        // 3. Run Forward Pass (predict logits over the action vocabulary space)
        let logits = model.forward(states);

        // Output logits shape is [batch_size=1, seq_length=1, action_vocab_size=5]
        assert_eq!(logits.dims(), [1, 1, 5]);

        // 4. Action Masking Logic!
        // Suppose the environment dictates that ONLY actions 1 (Shift) and 3 (Reduce_B) are structurally valid right now.
        // We build a boolean mask where `true` means INVALID (so we can mask them out).
        let is_invalid_action_mask = Tensor::<Backend, 3, burn::tensor::Bool>::from_bool(
            TensorData::from([[[true, false, true, false, true]]]),
            &device,
        );

        // 5. Mask invalid elements down to negative infinity so they have zero probability after Softmax
        // `mask_fill` fills elements where the mask is TRUE.
        let masked_logits = logits.mask_fill(is_invalid_action_mask, -f32::INFINITY);

        // 6. Compute action probabilities
        // Softmax across the action vocabulary dimension (dim 2)
        let probabilities = burn::tensor::activation::softmax(masked_logits, 2);

        // 7. Verify mathematical constraints bounds
        // - Probability of the invalid actions strictly == 0
        // - Probabilities of valid actions sum to exactly 1.0!
        let tensor_data = probabilities.into_data();
        let probs_array: &[f32] = tensor_data.as_slice().unwrap();
        assert_eq!(probs_array[0], 0.0); // Invalid `Done`
        assert_ne!(probs_array[1], 0.0);
        assert_eq!(probs_array[2], 0.0); // Invalid `Reduce_A`
        assert_ne!(probs_array[3], 0.0);
        assert_eq!(probs_array[4], 0.0); // Invalid `Reduce_C`

        let valid_sum = probs_array[1] + probs_array[3];
        assert!(
            (valid_sum - 1.0).abs() < 1e-5,
            "Valid probabilities must sum exactly to 1.0"
        );

        println!("Masked Sampling Probabilities Space: {:?}", probs_array);
    }

    #[test]
    fn test_print_model_architecture() {
        let device = Default::default();
        let config = ModelSize::Small.get_config(20, 20); // typical vocab sizes
        let model = config.init::<Backend>(&device);
        
        println!("Transformer Architecture:\n{}", model);
        // Burn implicitly calculates total trainable parameters and sizes upon module formatting.
    }
}
