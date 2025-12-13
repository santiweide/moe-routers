# SPDX-License-Identifier: Apache-2.0
"""
Metrics utilities for MoE routers and MoE layer behavior.

Designed for *analysis*, not maximum performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PerformanceEfficiencyMetrics:
    tokens: int
    total_moe_ms: float
    router_ms: float
    expert_ms: float
    token_latency_us: float


@dataclass
class RoutingBehaviorMetrics:
    routing_entropy: float  # nats
    normalized_entropy: float  # in [0,1] (approx), normalized by log(E)
    mean_topk_probability: float  # mean over tokens of sum of top-k probs from full softmax
    top1_route_fraction: float  # max route fraction over experts
    route_distribution: Optional[torch.Tensor] = None  # [E] float32 on CPU (optional)
    per_token_prob_sample: Optional[torch.Tensor] = None  # [E] float32 on CPU (optional)


@dataclass
class ExpertUsageMetrics:
    active_experts: int
    active_expert_fraction: float
    avg_active_experts_per_token: float
    load_mean_routes: float
    load_std_routes: float
    load_cv: float
    load_max_over_mean: float
    load_min_over_mean: float
    dead_experts: int
    aux_load_balance_loss: float
    unique_tokens_per_expert_mean: Optional[float] = None
    unique_tokens_per_expert_std: Optional[float] = None


@dataclass
class OutputCharacteristicsMetrics:
    output_mean: float
    output_var: float
    output_std: float
    output_absmax: float


@dataclass
class MoEMetrics:
    router_backend: str
    performance: PerformanceEfficiencyMetrics
    routing: RoutingBehaviorMetrics
    usage: ExpertUsageMetrics
    output: OutputCharacteristicsMetrics


def routing_entropy_from_distribution(p: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Compute entropy H(p) in nats for a probability vector p."""
    p = p.clamp_min(eps)
    h = -(p * p.log()).sum()
    return float(h.item())


def normalized_entropy(p: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Normalize entropy by log(E)."""
    e = int(p.numel())
    if e <= 1:
        return 0.0
    h = routing_entropy_from_distribution(p, eps=eps)
    return float(h / max(math.log(e), 1e-9))


def mean_topk_probability_from_logits(logits: torch.Tensor, k: int) -> float:
    """
    Mean over tokens of sum of top-k probabilities from *full* softmax(logits).
    """
    if logits.dim() != 2:
        raise ValueError("logits must be [tokens, experts]")
    if k <= 0:
        raise ValueError("k must be > 0")
    probs = torch.softmax(logits.float(), dim=-1)
    k = min(k, probs.size(-1))
    topk = torch.topk(probs, k=k, dim=-1).values.sum(dim=-1)
    return float(topk.mean().item())


def avg_unique_experts_per_token(topk_idx: torch.Tensor) -> float:
    """Average unique experts per token from [S, K] indices (handles duplicates)."""
    if topk_idx.dim() != 2:
        raise ValueError("topk_idx must be [tokens, k]")
    s, k = topk_idx.shape
    if k <= 1:
        return 1.0 if s > 0 else 0.0
    sorted_idx = torch.sort(topk_idx, dim=-1).values
    uniq = 1 + (sorted_idx[:, 1:] != sorted_idx[:, :-1]).to(torch.int32).sum(dim=-1)
    return float(uniq.float().mean().item())


def route_distribution_from_topk(topk_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Route distribution p_e from top-k indices (routes per expert / total routes)."""
    if topk_idx.dim() != 2:
        raise ValueError("topk_idx must be [tokens, k]")
    expert_ids = topk_idx.reshape(-1).to(torch.int64)
    counts = torch.bincount(expert_ids, minlength=n_experts).to(torch.float32)
    routes = float(expert_ids.numel()) if expert_ids.numel() > 0 else 1.0
    return counts / routes


def aux_load_balance_loss(
    *,
    route_frac: torch.Tensor,
    importance: torch.Tensor,
    alpha: float = 1.0,
) -> float:
    """
    Auxiliary load balancing loss (inference-time computable):
      L_aux = alpha * E * sum_e f_e * g_e
    where:
      f_e: expert load fraction from routing decisions
      g_e: expert importance from softmax probabilities (mean prob mass)
    """
    if route_frac.shape != importance.shape:
        raise ValueError("route_frac and importance must have same shape")
    e = int(route_frac.numel())
    val = float((route_frac * importance).sum().item())
    return float(alpha * e * val)


def unique_tokens_per_expert(topk_idx: torch.Tensor, n_experts: int, *, max_routes: int = 1_000_000) -> Optional[tuple[float, float]]:
    """
    Estimate token-to-expert utilization: number of unique tokens per expert.
    Returns (mean, std) or None if too large.
    """
    if topk_idx.dim() != 2:
        raise ValueError("topk_idx must be [tokens, k]")
    s, k = topk_idx.shape
    routes = s * k
    if routes > max_routes:
        return None
    token_ids = torch.arange(s, device=topk_idx.device, dtype=torch.int64).unsqueeze(1).expand(s, k).reshape(-1)
    expert_ids = topk_idx.reshape(-1).to(torch.int64)
    enc = token_ids * n_experts + expert_ids
    enc_u = torch.unique(enc)
    expert_u = (enc_u % n_experts).to(torch.int64)
    tok_counts = torch.bincount(expert_u, minlength=n_experts).to(torch.float32)
    mean = float(tok_counts.mean().item())
    std = float(tok_counts.std(unbiased=False).item())
    return mean, std


def output_distribution_stats(x: torch.Tensor) -> OutputCharacteristicsMetrics:
    xf = x.float()
    return OutputCharacteristicsMetrics(
        output_mean=float(xf.mean().item()),
        output_var=float(xf.var(unbiased=False).item()),
        output_std=float(xf.std(unbiased=False).item()),
        output_absmax=float(xf.abs().max().item()),
    )


