from __future__ import annotations
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaConvEncoder(nn.Module):
    """Encodes a 2D signal matrix (time_steps, num_instruments) into d_model."""

    def __init__(self, hidden_dim: int = 64, d_model: int = 256, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(1, hidden_dim, kernel_size=kernel_size)
        self.proj = nn.Linear(hidden_dim, d_model)

    def forward(self, alpha_matrix: torch.Tensor) -> torch.Tensor:
        if alpha_matrix.dim() == 2:
            alpha_matrix = alpha_matrix.unsqueeze(0)
        b, t, n = alpha_matrix.shape
        x = alpha_matrix.transpose(1, 2).contiguous().view(b * n, 1, t)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = F.relu(self.conv1d(x))
        x = x.max(dim=-1).values
        x = x.view(b, n, -1).mean(dim=1)
        return self.proj(x)


class SRDecoderModel(nn.Module):
    """Autoregressive Transformer decoder conditioned on the stack signal context."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        nhid: int = 512,
        nlayers: int = 4,
        max_len: int = 500,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.alpha_encoder = AlphaConvEncoder(64, d_model, kernel_size=5)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=nlayers)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(
        self, shifted_target_tokens: torch.Tensor, alpha_matrix: torch.Tensor
    ) -> torch.Tensor:
        seq_len = shifted_target_tokens.size(1)
        device = shifted_target_tokens.device

        memory = self.alpha_encoder(alpha_matrix).unsqueeze(1)  # (B, 1, d_model)
        tgt = self.token_emb(shifted_target_tokens) * math.sqrt(self.d_model)
        tgt = tgt + self.pos_emb[:, :seq_len, :]

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        out = self.decoder(tgt, memory, tgt_mask=causal_mask)
        return self.output_linear(out)


class ValueNet(nn.Module):
    """Predicts (mu, log_sigma) of expected return from the 5D sequence summary."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, emb_5d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(emb_5d)
        mu = out[..., 0]
        log_sigma = out[..., 1].clamp(-2.0, 2.0)
        return mu, log_sigma


class AgentModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, seq_len: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.decoder = SRDecoderModel(vocab_size, d_model=d_model)
        self.value_net = ValueNet()

    @torch.no_grad()
    def act(
        self,
        shifted_tgt: torch.Tensor,
        alpha_matrix: torch.Tensor,
        step_idx: int,
        mask: torch.Tensor,
        emb_5d: torch.Tensor,
        temperature: float = 10.0,
    ) -> Tuple[int, float, float]:
        logits = self.decoder(shifted_tgt.unsqueeze(0), alpha_matrix.unsqueeze(0))
        step_logits = logits[0, min(step_idx, self.seq_len - 1)]
        masked_logits = step_logits.masked_fill(~mask, float("-inf"))

        log_probs = F.log_softmax(masked_logits / temperature, dim=-1)
        probs = torch.nan_to_num(log_probs.exp(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        prob_sum = probs.sum()
        if prob_sum <= 1e-8:
            probs = mask.float()
            probs = probs / probs.sum().clamp_min(1e-8)
        else:
            probs = probs / prob_sum

        action_idx = int(torch.multinomial(probs, 1).item())
        log_prob = float(log_probs[action_idx].item())

        mu, log_sigma = self.value_net(emb_5d.unsqueeze(0))
        value = float((mu[0] + 0.5 * log_sigma[0].exp()).item())
        return action_idx, log_prob, value
