# SPDX-License-Identifier: Apache-2.0
"""
Decoder-only Transformer (minimal, PyTorch-only).

Structure (3 phases):
1) Input & Embedding
2) Decoder blocks stacked N times:
   - RMSNorm -> Masked Self-Attention (causal) -> Residual
   - RMSNorm -> MLP (SwiGLU) -> Residual
3) Output head:
   - Final RMSNorm -> LM Head (linear) -> logits
   - (optional) decoding strategy in `generate()`: greedy / sampling (top-k/top-p)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


class RMSNorm(torch.nn.Module):
    """RMSNorm: x * (1 / rms(x)) * weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to q/k.

    Shapes:
      q, k: [B, n_heads, T, head_dim]
      cos/sin: [T, head_dim] or broadcastable to q/k.
    """
    # Broadcast cos/sin over batch and heads.
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T,head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(torch.nn.Module):
    """Precomputes RoPE cos/sin tables (in fp32) and serves slices by seq length."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len = 0
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()
        self._cached_seq_len = seq_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._cached_seq_len or self._cos_cached.device != device:
            self._build_cache(seq_len, device)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


class MaskedSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: bool = True,
        rope_base: float = 10000.0,
        attn_dropout_p: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} % {n_heads} != 0")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout_p = attn_dropout_p

        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = torch.nn.Linear(d_model, d_model, bias=False)

        self.use_rope = rope
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)  # [B, T, 3*d]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, nH, T, Hd]
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            assert self.rope is not None
            cos, sin = self.rope(seq_len, device=x.device)
            q, k = apply_rope(q, k, cos.to(dtype=q.dtype), sin.to(dtype=q.dtype))

        # PyTorch 2.x fast path. `is_causal=True` applies causal mask internally.
        if hasattr(F, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=True,
            )  # [B, nH, T, Hd]
        else:
            # Manual attention: [B, nH, T, T]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
            probs = scores.softmax(dim=-1)
            if self.training and self.attn_dropout_p > 0:
                probs = F.dropout(probs, p=self.attn_dropout_p)
            attn = probs @ v

        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out(attn)


class SwiGLU(torch.nn.Module):
    """SwiGLU MLP: silu(W1 x) * (W3 x) -> W2."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, hidden_dim, bias=False)  # gate
        self.w3 = torch.nn.Linear(d_model, hidden_dim, bias=False)  # up
        self.w2 = torch.nn.Linear(hidden_dim, d_model, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_hidden_dim: int,
        rope: bool = True,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        attn_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = MaskedSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            rope=rope,
            rope_base=rope_base,
            attn_dropout_p=attn_dropout_p,
        )
        self.mlp_norm = RMSNorm(d_model, eps=norm_eps)
        self.mlp = SwiGLU(d_model=d_model, hidden_dim=mlp_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sub-layer A: x = x + Attn(RMSNorm(x))
        x = x + self.attn(self.attn_norm(x))
        # Sub-layer B: x = x + MLP(RMSNorm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


@dataclass(frozen=True)
class TransformerDecoderConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    mlp_hidden_dim: int = 2048
    max_seq_len: int = 2048
    norm_eps: float = 1e-6
    rope: bool = True
    rope_base: float = 10000.0
    attn_dropout_p: float = 0.0
    tie_embeddings: bool = True
    absolute_pos_embedding: bool = False


class TransformerDecoderModel(torch.nn.Module):
    """
    Decoder-only Transformer.

    Forward:
      input_ids: [B, T] -> logits: [B, T, vocab_size]
    """

    def __init__(self, cfg: TransformerDecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Phase 1: Input & Embedding
        self.tok_embed = torch.nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = (
            torch.nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.absolute_pos_embedding else None
        )

        # Phase 2: Decoder blocks (repeat N times)
        self.blocks = torch.nn.ModuleList(
            [
                DecoderBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    mlp_hidden_dim=cfg.mlp_hidden_dim,
                    rope=cfg.rope,
                    rope_base=cfg.rope_base,
                    norm_eps=cfg.norm_eps,
                    attn_dropout_p=cfg.attn_dropout_p,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        # Phase 3: Output head
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = torch.nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, T], got shape={tuple(input_ids.shape)}")
        bsz, seq_len = input_ids.shape
        if self.pos_embed is not None and seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")

        # Phase 1
        x = self.tok_embed(input_ids)  # [B, T, d_model]
        if self.pos_embed is not None:
            pos = torch.arange(seq_len, device=input_ids.device)
            x = x + self.pos_embed(pos).unsqueeze(0)

        # Phase 2
        for blk in self.blocks:
            x = blk(x)

        # Phase 3
        x = self.final_norm(x)
        logits = self.lm_head(x)  # [B, T, vocab]
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive decoding.

        - greedy=True: argmax
        - else: sampling with optional temperature/top-k/top-p.
        """
        if max_new_tokens <= 0:
            return input_ids
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0")
        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            logits = self(out)  # [B, T, V]
            next_logits = logits[:, -1, :]  # [B, V]

            if greedy:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                scores = next_logits / temperature

                if top_k is not None:
                    k = min(top_k, scores.size(-1))
                    vals, idx = torch.topk(scores, k, dim=-1)
                    filtered = torch.full_like(scores, float("-inf"))
                    filtered.scatter_(dim=-1, index=idx, src=vals)
                    scores = filtered

                if top_p is not None:
                    sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
                    sorted_probs = torch.softmax(sorted_scores, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)
                    # Remove tokens with cumulative probability above threshold.
                    remove = cumprobs > top_p
                    # Keep at least one token.
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
                    scores = torch.full_like(scores, float("-inf"))
                    scores.scatter_(dim=-1, index=sorted_idx, src=sorted_scores)

                probs = torch.softmax(scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, next_token], dim=1)
        return out


__all__ = [
    "TransformerDecoderConfig",
    "TransformerDecoderModel",
    "DecoderBlock",
    "MaskedSelfAttention",
    "RMSNorm",
]


