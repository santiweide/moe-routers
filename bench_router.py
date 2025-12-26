# SPDX-License-Identifier: Apache-2.0
"""
Router microbenchmark (single process / single GPU).

Matches the paper-style breakdown:
  T_logit  : router logit matmul (x @ W^T)
  T_select : selection (Top-k / Sinkhorn / CUDA router_ext)
  T_pack   : pack routes (token_ids/expert_ids sort + input gather)
  T_route  : sum of the above

This script is forward-only.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch

from models.decoder_moe import SinkhornRouter, TorchTopKRouter


def _maybe_import_router_ext():
    try:
        import router_ext_cuda  # type: ignore

        return router_ext_cuda
    except Exception:
        return None


@dataclass
class Timing:
    t_logit_ms: float
    t_select_ms: float
    t_pack_ms: float
    t_route_ms: float


def _cuda_time(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _pack_routes(
    x: torch.Tensor, topk_idx: torch.Tensor, topk_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack routes by expert (single-GPU). Returns:
      x_route_sorted: [R, d]
      token_ids_sorted: [R]
      expert_ids_sorted: [R]
    """
    s, k = topk_idx.shape
    d = x.size(-1)
    token_ids = torch.arange(s, device=x.device, dtype=torch.int64).unsqueeze(1).expand(s, k).reshape(-1)
    expert_ids = topk_idx.reshape(-1).to(torch.int64)
    weights = topk_w.reshape(-1).unsqueeze(-1)

    x_route = x.index_select(0, token_ids) * weights  # [R, d]
    sort_idx = torch.argsort(expert_ids)
    return (
        x_route.index_select(0, sort_idx),
        token_ids.index_select(0, sort_idx),
        expert_ids.index_select(0, sort_idx),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--tokens", type=int, default=4096, help="T in the paper (number of tokens)")
    p.add_argument("--d_model", type=int, default=4096)
    p.add_argument("--experts", type=int, default=64, help="E")
    p.add_argument("--top_k", type=int, default=2, help="k")
    p.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="Run multiple strategies and print a comparison table (overrides --strategy).",
    )
    p.add_argument(
        "--all_strategies",
        action="store_true",
        help="Run all strategies: naive_topk, masked_matmul (A), fused_select (B), sinkhorn (C).",
    )
    p.add_argument(
        "--strategy",
        type=str,
        default="naive_topk",
        choices=["naive_topk", "masked_matmul", "fused_select", "sinkhorn"],
    )
    p.add_argument("--sinkhorn_iters", type=int, default=10)
    p.add_argument("--sinkhorn_temperature", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    # Synthetic skew: bias a subset of experts in logits
    p.add_argument("--skew_experts", type=int, default=0, help="How many experts to bias (0=disabled)")
    p.add_argument("--skew_bias", type=float, default=0.0, help="Bias value added to those experts")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    torch.manual_seed(0)
    x = torch.randn(args.tokens, args.d_model, device=device, dtype=dtype)
    # Router weight: [E, d]
    w = torch.randn(args.experts, args.d_model, device=device, dtype=dtype)

    ext = _maybe_import_router_ext()
    torch_router = TorchTopKRouter()
    sinkhorn_router = SinkhornRouter(iters=args.sinkhorn_iters, temperature=args.sinkhorn_temperature)

    strategies_all = ["naive_topk", "masked_matmul", "fused_select", "sinkhorn"]
    if args.all_strategies:
        strategies = strategies_all
    elif args.strategies is not None:
        strategies = args.strategies
    else:
        strategies = [args.strategy]

    unknown = [s for s in strategies if s not in strategies_all]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Valid: {strategies_all}")

    if "fused_select" in strategies and ext is None and device.type == "cuda":
        print("WARNING: strategy=fused_select requested but router_ext_cuda is not available; will fall back to naive_topk.")

    def one_iter(strategy: str) -> Timing:
        if device.type != "cuda":
            # CPU fallback timing (rough)
            t0 = time.perf_counter()
            logits = x.float() @ w.float().t()
            t1 = time.perf_counter()
            if args.skew_experts > 0 and args.skew_bias != 0.0:
                logits[:, : args.skew_experts] += args.skew_bias
            if strategy == "masked_matmul":
                probs = torch.softmax(logits, dim=-1)
                # simulate compute cost only (no selection/pack)
                _ = probs.sum()
                t2 = time.perf_counter()
                return Timing((t1 - t0) * 1e3, (t2 - t1) * 1e3, 0.0, (t2 - t0) * 1e3)

            if strategy == "sinkhorn":
                topk_idx, topk_w = sinkhorn_router(logits, args.top_k)
            else:
                topk_idx, topk_w = torch_router(logits, args.top_k)
            t2 = time.perf_counter()
            _pack_routes(x, topk_idx.to(device=x.device), topk_w.to(device=x.device))
            t3 = time.perf_counter()
            return Timing((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3)

        # CUDA: measure with events
        logits: torch.Tensor
        topk_idx: torch.Tensor
        topk_w: torch.Tensor

        def do_logit():
            nonlocal logits
            logits = x @ w.t()
            if args.skew_experts > 0 and args.skew_bias != 0.0:
                logits[:, : args.skew_experts] += args.skew_bias

        t_logit = _cuda_time(do_logit)

        def do_select():
            nonlocal topk_idx, topk_w
            if strategy == "masked_matmul":
                # Strategy A: avoid permutation by computing a dense mask/probabilities.
                # Here we just compute probabilities; "compute redundancy" happens in expert compute.
                _ = torch.softmax(logits.float(), dim=-1)
                topk_idx = torch.empty((args.tokens, args.top_k), device=device, dtype=torch.int64)
                topk_w = torch.empty((args.tokens, args.top_k), device=device, dtype=dtype)
                return

            if strategy == "sinkhorn":
                topk_idx, topk_w = sinkhorn_router(logits, args.top_k)
                return

            if strategy == "fused_select" and ext is not None and logits.is_cuda and args.top_k <= 8:
                idx_i32, w_f32 = ext.forward(logits, int(args.top_k))
                topk_idx = idx_i32.to(torch.int64)
                topk_w = w_f32.to(dtype=logits.dtype)
                return

            # naive_topk
            topk_vals, topk_idx = torch.topk(logits, k=args.top_k, dim=-1)
            topk_w = torch.softmax(topk_vals, dim=-1)

        t_select = _cuda_time(do_select)

        def do_pack():
            if strategy == "masked_matmul":
                return
            _pack_routes(x, topk_idx, topk_w)

        t_pack = _cuda_time(do_pack)
        return Timing(t_logit, t_select, t_pack, t_logit + t_select + t_pack)

    def run_strategy(strategy: str) -> Timing:
        # Warmup
        for _ in range(args.warmup):
            _ = one_iter(strategy)
        timings = [one_iter(strategy) for _ in range(args.iters)]

        def avg(getter) -> float:
            return sum(getter(t) for t in timings) / max(len(timings), 1)

        return Timing(
            t_logit_ms=avg(lambda t: t.t_logit_ms),
            t_select_ms=avg(lambda t: t.t_select_ms),
            t_pack_ms=avg(lambda t: t.t_pack_ms),
            t_route_ms=avg(lambda t: t.t_route_ms),
        )

    results: dict[str, Timing] = {}
    for s in strategies:
        results[s] = run_strategy(s)

    # Metadata header
    if len(strategies) == 1:
        print("strategy:", strategies[0])
    else:
        print("strategies:", strategies)
    print("device:", device)
    print("dtype:", dtype)
    print("T(tokens):", args.tokens, "d_model:", args.d_model, "E:", args.experts, "k:", args.top_k)
    print("skew_experts:", args.skew_experts, "skew_bias:", args.skew_bias)

    if len(strategies) == 1:
        mean = results[strategies[0]]
        print("T_logit_ms:", mean.t_logit_ms)
        print("T_select_ms:", mean.t_select_ms)
        print("T_pack_ms:", mean.t_pack_ms)
        print("T_route_ms:", mean.t_route_ms)
        return

    # Table for multi-strategy
    print("")
    print("=== Router Latency Breakdown (ms, mean) ===")
    print(f"{'strategy':<14} {'T_logit':>10} {'T_select':>10} {'T_pack':>10} {'T_route':>10}")
    for s in strategies:
        t = results[s]
        print(f"{s:<14} {t.t_logit_ms:10.3f} {t.t_select_ms:10.3f} {t.t_pack_ms:10.3f} {t.t_route_ms:10.3f}")


if __name__ == "__main__":
    main()


