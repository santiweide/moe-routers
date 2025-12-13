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


@dataclass
class MoERouterMetrics:
    router_backend: str
    tokens: int
    experts: int
    top_k: int
    routes: int
    router_overhead_ms: float
    avg_active_experts_per_token: float
    active_experts: int
    active_expert_fraction: float
    top1_route_fraction: float
    normalized_entropy: float
    route_sparsity: float
    load_mean: float
    load_std: float
    load_cv: float
    load_max_over_mean: float
    load_min_over_mean: float
    # Optional (may be skipped for very large R)
    unique_tokens_per_expert_mean: Optional[float] = None
    unique_tokens_per_expert_std: Optional[float] = None


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

    @property
    def name(self) -> str:
        return self.__class__.__name__


class TorchTopKRouter(RouterBase):
    def forward(self, logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
        topk_w = torch.softmax(topk_vals, dim=-1)
        return topk_idx.to(torch.int64), topk_w


class SinkhornRouter(RouterBase):
    """
    Sinkhorn-Knopp router (reference implementation, Torch-only).

    Idea:
    - Convert router logits -> nonnegative matrix Q (tokens x experts)
    - Run Sinkhorn iterations to approximately enforce:
        row sums ~= 1          (each token distributes prob mass)
        col sums ~= S/E        (balanced expert load)
    - Select top-k experts per token from the resulting assignment matrix.

    This is mainly for *analysis/quality/load-balance* comparisons, not speed.
    Complexity is O(iters * S * E).
    """

    def __init__(self, *, iters: int = 10, epsilon: float = 1e-6, temperature: float = 1.0):
        super().__init__()
        if iters <= 0:
            raise ValueError("SinkhornRouter.iters must be > 0")
        if epsilon <= 0:
            raise ValueError("SinkhornRouter.epsilon must be > 0")
        if temperature <= 0:
            raise ValueError("SinkhornRouter.temperature must be > 0")
        self.iters = int(iters)
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)

    def forward(self, logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        # logits: [S, E]
        if logits.dim() != 2:
            raise ValueError("SinkhornRouter expects logits of shape [tokens, experts]")
        s, e = logits.shape
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if top_k > e:
            raise ValueError("top_k must be <= n_experts")

        # Use fp32 for numerical stability.
        scores = (logits / self.temperature).float()
        # Exponentiate with stabilization.
        scores = scores - scores.max(dim=-1, keepdim=True).values
        q = torch.exp(scores)

        # Target column sum: S / E (balanced load), row sum: 1.
        col_target = float(s) / float(max(e, 1))
        eps = self.epsilon

        for _ in range(self.iters):
            # Normalize rows to sum to 1.
            q = q / (q.sum(dim=1, keepdim=True) + eps)
            # Normalize cols to sum to col_target.
            q = q / (q.sum(dim=0, keepdim=True) + eps) * col_target

        # Final row renorm to get probabilities per token.
        q = q / (q.sum(dim=1, keepdim=True) + eps)

        vals, idx = torch.topk(q, k=top_k, dim=-1)  # [S, K]
        w = vals / (vals.sum(dim=-1, keepdim=True) + eps)
        return idx.to(torch.int64), w.to(dtype=logits.dtype)


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
        track_metrics: bool = False,
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
        self.track_metrics = track_metrics
        self._last_metrics: Optional[MoERouterMetrics] = None
        self.experts = torch.nn.ModuleList(
            [SwiGLU(d_model=d_model, hidden_dim=expert_hidden_dim) for _ in range(n_experts)]
        )

    def get_last_metrics(self) -> Optional[MoERouterMetrics]:
        return self._last_metrics

    def _compute_metrics(
        self,
        *,
        topk_idx: torch.Tensor,
        router_overhead_ms: float,
    ) -> MoERouterMetrics:
        # topk_idx: [S, K] int64
        s, k = topk_idx.shape
        e = self.n_experts
        routes = int(s * k)

        # Average unique experts per token (handles accidental duplicates).
        if k == 1:
            avg_active = 1.0
        else:
            sorted_idx = torch.sort(topk_idx, dim=-1).values
            uniq = 1 + (sorted_idx[:, 1:] != sorted_idx[:, :-1]).to(torch.int32).sum(dim=-1)
            avg_active = float(uniq.float().mean().item())

        expert_ids = topk_idx.reshape(-1)
        counts = torch.bincount(expert_ids, minlength=e).to(torch.float32)  # routes per expert
        active = int((counts > 0).sum().item())
        active_frac = float(active / max(e, 1))
        p = counts / max(routes, 1)
        top1 = float(p.max().item()) if routes > 0 else 0.0
        # Entropy in [0, log(E)] -> normalize to [0,1]
        eps = 1e-12
        entropy = -(p[p > 0] * (p[p > 0] + eps).log()).sum()
        norm_entropy = float((entropy / max(torch.log(torch.tensor(float(e))), torch.tensor(1.0))).item())
        route_sparsity = 1.0 - norm_entropy

        mean = float(counts.mean().item()) if e > 0 else 0.0
        std = float(counts.std(unbiased=False).item()) if e > 0 else 0.0
        cv = float(std / (mean + 1e-9))
        max_over_mean = float((counts.max().item() / (mean + 1e-9))) if e > 0 else 0.0
        min_over_mean = float((counts.min().item() / (mean + 1e-9))) if e > 0 else 0.0

        unique_mean: Optional[float] = None
        unique_std: Optional[float] = None
        # Token-to-expert utilization: unique tokens per expert (skip if huge).
        if routes <= 1_000_000:
            token_ids = (
                torch.arange(s, device=topk_idx.device, dtype=torch.int64)
                .unsqueeze(1)
                .expand(s, k)
                .reshape(-1)
            )
            enc = token_ids * e + expert_ids
            enc_u = torch.unique(enc)
            expert_u = (enc_u % e).to(torch.int64)
            tok_counts = torch.bincount(expert_u, minlength=e).to(torch.float32)
            unique_mean = float(tok_counts.mean().item())
            unique_std = float(tok_counts.std(unbiased=False).item())

        return MoERouterMetrics(
            router_backend=self.router_impl.name,
            tokens=int(s),
            experts=int(e),
            top_k=int(k),
            routes=int(routes),
            router_overhead_ms=float(router_overhead_ms),
            avg_active_experts_per_token=float(avg_active),
            active_experts=int(active),
            active_expert_fraction=float(active_frac),
            top1_route_fraction=float(top1),
            normalized_entropy=float(norm_entropy),
            route_sparsity=float(route_sparsity),
            load_mean=float(mean),
            load_std=float(std),
            load_cv=float(cv),
            load_max_over_mean=float(max_over_mean),
            load_min_over_mean=float(min_over_mean),
            unique_tokens_per_expert_mean=unique_mean,
            unique_tokens_per_expert_std=unique_std,
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
        #
        # Optimization: token bucketing + per-expert batched GEMM.
        # We expand routes (token, expert, weight) -> sort by expert -> run one
        # expert MLP per contiguous bucket -> index_add back to token outputs.
        router_overhead_ms = 0.0
        if self.track_metrics and x.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            topk_idx, topk_w = self.router_impl(router_logits, self.top_k)  # [S, K], [S, K]
            end.record()
            torch.cuda.synchronize()
            router_overhead_ms = float(start.elapsed_time(end))
        else:
            topk_idx, topk_w = self.router_impl(router_logits, self.top_k)  # [S, K], [S, K]

        if self.track_metrics:
            # Ensure int64 for metrics computations.
            self._last_metrics = self._compute_metrics(
                topk_idx=topk_idx.to(torch.int64),
                router_overhead_ms=router_overhead_ms,
            )

        s = flat.size(0)
        k = self.top_k

        # Routes: R = S*K
        token_ids = torch.arange(s, device=flat.device, dtype=torch.int64).unsqueeze(1).expand(s, k).reshape(-1)
        expert_ids = topk_idx.reshape(-1).to(dtype=torch.int64)  # [R]
        weights = topk_w.reshape(-1).unsqueeze(-1)  # [R, 1]

        # Gather inputs for each route.
        x_route = flat.index_select(0, token_ids)  # [R, d]

        # Sort routes by expert id for contiguous buckets.
        sort_idx = torch.argsort(expert_ids)
        expert_ids = expert_ids.index_select(0, sort_idx)
        token_ids = token_ids.index_select(0, sort_idx)
        weights = weights.index_select(0, sort_idx)
        x_route = x_route.index_select(0, sort_idx)

        # Compute bucket sizes for slicing.
        counts = torch.bincount(expert_ids, minlength=self.n_experts)  # [E]
        out = torch.zeros_like(flat)

        start = 0
        for e, expert in enumerate(self.experts):
            cnt = int(counts[e].item())
            if cnt == 0:
                continue
            end = start + cnt
            x_e = x_route[start:end]
            w_e = weights[start:end]
            tok_e = token_ids[start:end]

            y_e = expert(x_e) * w_e  # [cnt, d]
            out.index_add_(0, tok_e, y_e)

            start = end

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
        track_metrics: bool = False,
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
            track_metrics=track_metrics,
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

    def __init__(
        self,
        cfg: DecoderMoEConfig,
        *,
        router_backend: str = "torch",
        track_metrics: bool = False,
        sinkhorn_iters: int = 10,
        sinkhorn_epsilon: float = 1e-6,
        sinkhorn_temperature: float = 1.0,
    ):
        super().__init__()
        self.cfg = cfg
        if router_backend not in {"torch", "cuda_ext", "sinkhorn"}:
            raise ValueError("router_backend must be one of: 'torch', 'cuda_ext', 'sinkhorn'")
        router_impl: Optional[RouterBase]
        if router_backend == "cuda_ext":
            router_impl = CUDATopKRouter()
        elif router_backend == "sinkhorn":
            router_impl = SinkhornRouter(
                iters=sinkhorn_iters, epsilon=sinkhorn_epsilon, temperature=sinkhorn_temperature
            )
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
                    track_metrics=track_metrics,
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

    def get_last_router_metrics(self) -> Optional[MoERouterMetrics]:
        # Report metrics from the first block's MoE layer (representative).
        if not self.blocks:
            return None
        moe = getattr(self.blocks[0], "moe", None)
        if moe is None:
            return None
        return moe.get_last_metrics()

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
    "DecoderMoEConfig",
    "DecoderMoEModel",
    "MoELayer",
    "MoERouterMetrics",
    "SinkhornRouter",
]


