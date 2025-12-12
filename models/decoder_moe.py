# SPDX-License-Identifier: Apache-2.0
"""
Decoder-only Transformer with MoE in the MLP sub-layer.

Same overall structure as `transformer_decoder_model.py`:
1) Input & Embedding
2) Decoder blocks stacked N times:
   - RMSNorm -> Masked Self-Attention (causal, optional RoPE) -> Residual
   - RMSNorm -> MoE (router + experts) -> Residual
3) Output head:
   - Final RMSNorm -> LM Head (linear) -> logits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .transformer_decoder_model import MaskedSelfAttention, RMSNorm, SwiGLU


def _try_import_router_ext():
    try:
        import router_ext_cuda  # type: ignore

        return router_ext_cuda
    except Exception:
        return None


class RouterBase(torch.nn.Module):
    """Interface: route logits -> (topk_idx[int64], topk_weight[float])"""

    def forward(self, logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        raise NotImplementedError


class TorchTopKRouter(RouterBase):
    def forward(self, logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
        topk_w = torch.softmax(topk_vals, dim=-1)
        return topk_idx.to(torch.int64), topk_w


class CUDATopKRouter(RouterBase):
    """Uses the `router_ext_cuda` pybind/CUDA extension for top-k routing."""

    def __init__(self):
        super().__init__()
        self._ext = _try_import_router_ext()

    @property
    def available(self) -> bool:
        return self._ext is not None

    def forward(self, logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._ext is None:
            # Fallback to torch if extension isn't built/available.
            return TorchTopKRouter()(logits, top_k)
        if (not logits.is_cuda) or torch.onnx.is_in_onnx_export():
            return TorchTopKRouter()(logits, top_k)
        # Extension supports k<=8; enforce here so errors are clearer.
        if top_k > 8:
            return TorchTopKRouter()(logits, top_k)
        # Extension supports fp16/fp32/bf16 logits.
        if logits.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            return TorchTopKRouter()(logits, top_k)

        idx_i32, w_f32 = self._ext.forward(logits, int(top_k))
        return idx_i32.to(torch.int64), w_f32.to(dtype=logits.dtype)


class MoELayer(torch.nn.Module):
    """
    Token-level MoE with top-k routing.

    - Router: Linear(d_model -> n_experts)
    - Experts: list of SwiGLU MLPs
    - Combine: weighted sum over selected experts per token

    NOTE: For ONNX export, we avoid top-k + boolean dispatch and instead compute a
    dense mixture over all experts (still a valid MoE, just not sparse).
    """

    def __init__(
        self,
        d_model: int,
        expert_hidden_dim: int,
        n_experts: int,
        top_k: int = 2,
        *,
        router_impl: Optional[RouterBase] = None,
    ):
        super().__init__()
        if n_experts <= 0:
            raise ValueError("n_experts must be > 0")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if top_k > n_experts:
            raise ValueError("top_k must be <= n_experts")

        self.n_experts = n_experts
        self.top_k = top_k
        self.router = torch.nn.Linear(d_model, n_experts, bias=False)
        self.router_impl: RouterBase = router_impl if router_impl is not None else TorchTopKRouter()
        self.experts = torch.nn.ModuleList(
            [SwiGLU(d_model=d_model, hidden_dim=expert_hidden_dim) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        bsz, seq_len, d_model = x.shape
        flat = x.view(-1, d_model)  # [S, d_model], S=B*T
        router_logits = self.router(flat)  # [S, E]

        # Export-friendly dense MoE: softmax over all experts.
        if torch.onnx.is_in_onnx_export():
            gates = torch.softmax(router_logits, dim=-1)  # [S, E]
            out = torch.zeros_like(flat)
            for e, expert in enumerate(self.experts):
                expert_out = expert(flat)  # [S, d]
                out = out + expert_out * gates[:, e : e + 1]
            return out.view(bsz, seq_len, d_model)

        # Sparse top-k MoE for regular execution (router implementation is pluggable).
        topk_idx, topk_w = self.router_impl(router_logits, self.top_k)  # [S, K], [S, K]

        out = torch.zeros_like(flat)
        # Dispatch per expert per top-k slot (simple/reference implementation).
        for slot in range(self.top_k):
            idx_s = topk_idx[:, slot]  # [S]
            w_s = topk_w[:, slot].unsqueeze(-1)  # [S, 1]
            for e, expert in enumerate(self.experts):
                mask = idx_s == e
                if mask.any():
                    y = expert(flat[mask])
                    out[mask] = out[mask] + y * w_s[mask]

        return out.view(bsz, seq_len, d_model)


class DecoderMoEBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        moe_expert_hidden_dim: int,
        moe_n_experts: int,
        moe_top_k: int,
        *,
        rope: bool = True,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        attn_dropout_p: float = 0.0,
        router_impl: Optional[RouterBase] = None,
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
        self.moe_norm = RMSNorm(d_model, eps=norm_eps)
        self.moe = MoELayer(
            d_model=d_model,
            expert_hidden_dim=moe_expert_hidden_dim,
            n_experts=moe_n_experts,
            top_k=moe_top_k,
            router_impl=router_impl,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.moe_norm(x))
        return x


@dataclass(frozen=True)
class DecoderMoEConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6

    # MoE params
    moe_expert_hidden_dim: int = 2048
    moe_n_experts: int = 8
    moe_top_k: int = 2

    # Misc
    max_seq_len: int = 2048
    norm_eps: float = 1e-6
    rope: bool = True
    rope_base: float = 10000.0
    attn_dropout_p: float = 0.0
    tie_embeddings: bool = True
    absolute_pos_embedding: bool = False


class DecoderMoEModel(torch.nn.Module):
    """Decoder-only Transformer where the MLP sub-layer is replaced by MoE."""

    def __init__(self, cfg: DecoderMoEConfig, *, router_backend: str = "torch"):
        super().__init__()
        self.cfg = cfg
        if router_backend not in {"torch", "cuda_ext"}:
            raise ValueError("router_backend must be one of: 'torch', 'cuda_ext'")
        router_impl: Optional[RouterBase]
        if router_backend == "cuda_ext":
            router_impl = CUDATopKRouter()
        else:
            router_impl = TorchTopKRouter()

        # Phase 1: Input & Embedding
        self.tok_embed = torch.nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = (
            torch.nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.absolute_pos_embedding else None
        )

        # Phase 2: Decoder MoE blocks
        self.blocks = torch.nn.ModuleList(
            [
                DecoderMoEBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    moe_expert_hidden_dim=cfg.moe_expert_hidden_dim,
                    moe_n_experts=cfg.moe_n_experts,
                    moe_top_k=cfg.moe_top_k,
                    rope=cfg.rope,
                    rope_base=cfg.rope_base,
                    norm_eps=cfg.norm_eps,
                    attn_dropout_p=cfg.attn_dropout_p,
                    router_impl=router_impl,
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

        x = self.tok_embed(input_ids)  # [B, T, d_model]
        if self.pos_embed is not None:
            pos = torch.arange(seq_len, device=input_ids.device)
            x = x + self.pos_embed(pos).unsqueeze(0)

        for blk in self.blocks:
            x = blk(x)

        x = self.final_norm(x)
        return self.lm_head(x)  # [B, T, vocab]


__all__ = [
    "DecoderMoEConfig",
    "DecoderMoEModel",
    "MoELayer",
]


