# SPDX-License-Identifier: Apache-2.0
"""
Router microbenchmark (single process / single GPU).

Matches the paper-style breakdown:
  T_logit  : router logit matmul (x @ W^T)
  T_select : selection (Top-k / Sinkhorn / CUDA router_ext)
  T_pack   : pack routes (token_ids/expert_ids sort + input gather)
  T_route  : sum of the above

Extra experiment mode:
  --perm_sweep : sweep batch size and measure permutation/data-movement latency (T_perm)
                for a naive baseline vs a coalesced/sorted pack, and optionally plot.

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


def _pack_routes_naive_mask_scan(
    x: torch.Tensor, topk_idx: torch.Tensor, topk_w: torch.Tensor, n_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Intentionally naive route packing:
    - Build per-route (token_id, expert_id, weight)
    - For each expert, scan expert_ids with a boolean mask and gather those routes
    - Concatenate per-expert buffers

    This is a useful "random/inefficient memory access" baseline for T_perm analysis.
    """
    s, k = topk_idx.shape
    token_ids = (
        torch.arange(s, device=x.device, dtype=torch.int64).unsqueeze(1).expand(s, k).reshape(-1)
    )
    expert_ids = topk_idx.reshape(-1).to(torch.int64)
    weights = topk_w.reshape(-1).unsqueeze(-1)

    # Gather (sequential token_ids -> reasonably contiguous reads), but then do
    # a very inefficient per-expert scan/gather on the routed buffer.
    x_route = x.index_select(0, token_ids) * weights  # [R, d]

    xs = []
    toks = []
    exps = []
    for e in range(int(n_experts)):
        m = expert_ids == e
        if torch.any(m):
            xs.append(x_route[m])
            toks.append(token_ids[m])
            exps.append(expert_ids[m])

    if not xs:
        # Degenerate (shouldn't happen for random routing unless tokens=0)
        empty_x = x_route[:0]
        empty_i = token_ids[:0]
        empty_e = expert_ids[:0]
        return empty_x, empty_i, empty_e

    return torch.cat(xs, dim=0), torch.cat(toks, dim=0), torch.cat(exps, dim=0)


def _peak_dram_bandwidth_gbs() -> Optional[float]:
    """
    Approximate theoretical peak DRAM bandwidth from CUDA device props.
    Returns GB/s, or None if unavailable.
    """
    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    # PyTorch reports these for CUDA devices:
    # - memory_clock_rate: kHz
    # - memory_bus_width: bits
    mem_clock_khz = getattr(p, "memory_clock_rate", None)
    bus_width_bits = getattr(p, "memory_bus_width", None)
    if mem_clock_khz is None or bus_width_bits is None:
        return None
    # GDDR is DDR: multiply by 2.
    bw_bytes_per_s = 2.0 * float(mem_clock_khz) * 1e3 * (float(bus_width_bits) / 8.0)
    return bw_bytes_per_s / 1e9


def _try_plot_perm(csv_path: str, png_path: str) -> bool:
    """
    Best-effort plotting. Returns True if a PNG was written.
    """
    try:
        import csv

        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return False

    batch = [int(r["batch_size"]) for r in rows]
    tokens = [int(r["tokens"]) for r in rows]
    t_naive = [float(r["t_perm_naive_ms"]) for r in rows]
    t_sorted = [float(r["t_perm_sorted_ms"]) for r in rows]
    t_theory = [float(r["t_theory_ms"]) for r in rows]

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(batch, t_naive, marker="o", label="Naive (mask-scan pack)")
    plt.plot(batch, t_sorted, marker="o", label="Coalesced (sorted pack)")
    plt.plot(batch, t_theory, linestyle="--", label="Theoretical DRAM limit (ideal 1-pass)")
    plt.xlabel("Batch Size (B)")
    plt.ylabel("Permutation Latency $T_{perm}$ (ms)")
    plt.title(f"Permutation Latency vs Batch Size (seq_len fixed, tokens=B*T; last={tokens[-1]} tokens)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    return True


def _run_perm_sweep(args, *, device: torch.device, dtype: torch.dtype) -> None:
    if device.type != "cuda":
        raise RuntimeError("--perm_sweep requires --device cuda")

    torch.manual_seed(0)
    bw_gbs = _peak_dram_bandwidth_gbs()

    # We'll generate synthetic routes to isolate packing/permutation cost:
    # topk_idx: [S, K] int64, topk_w: [S, K] same dtype as x.
    batch_sizes = args.batch_sizes
    seq_len = int(args.seq_len)
    e = int(args.experts)
    k = int(args.top_k)
    d = int(args.d_model)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    import csv

    with open(args.csv_out, "w", newline="") as f:
        fieldnames = [
            "batch_size",
            "seq_len",
            "tokens",
            "experts",
            "top_k",
            "d_model",
            "dtype",
            "t_perm_naive_ms",
            "t_perm_sorted_ms",
            "t_theory_ms",
            "peak_bw_gbs",
            "sorted_achieved_bw_gbs",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for bsz in batch_sizes:
            tokens = int(bsz) * seq_len
            x = torch.randn(tokens, d, device=device, dtype=dtype)
            # Random routing (uniform experts); weights normalized per token.
            topk_idx = torch.randint(0, e, (tokens, k), device=device, dtype=torch.int64)
            topk_w = torch.rand(tokens, k, device=device, dtype=dtype)
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)

            # Warmup
            for _ in range(args.warmup):
                _ = _pack_routes_naive_mask_scan(x, topk_idx, topk_w, e)
                _ = _pack_routes(x, topk_idx, topk_w)
            torch.cuda.synchronize()

            def time_naive() -> float:
                def fn():
                    y, tok, exp = _pack_routes_naive_mask_scan(x, topk_idx, topk_w, e)
                    # Prevent DCE / ensure some dependency.
                    _ = (y.sum() + tok.sum().to(y.dtype) + exp.sum().to(y.dtype)) * 0.0

                return _cuda_time(fn)

            def time_sorted() -> float:
                def fn():
                    y, tok, exp = _pack_routes(x, topk_idx, topk_w)
                    _ = (y.sum() + tok.sum().to(y.dtype) + exp.sum().to(y.dtype)) * 0.0

                return _cuda_time(fn)

            t_perm_naive = sum(time_naive() for _ in range(args.iters)) / max(args.iters, 1)
            t_perm_sorted = sum(time_sorted() for _ in range(args.iters)) / max(args.iters, 1)

            # Ideal 1-pass DRAM traffic model:
            # read x (R*d) + write packed (R*d) => 2*R*d elements
            routes = tokens * k
            ideal_bytes = 2.0 * float(routes) * float(d) * float(bytes_per_elem)
            if bw_gbs is None or bw_gbs <= 0:
                t_theory_ms = float("nan")
                achieved_sorted = float("nan")
            else:
                t_theory_ms = (ideal_bytes / (bw_gbs * 1e9)) * 1e3
                achieved_sorted = (ideal_bytes / (t_perm_sorted * 1e-3)) / 1e9

            w.writerow(
                dict(
                    batch_size=int(bsz),
                    seq_len=int(seq_len),
                    tokens=int(tokens),
                    experts=int(e),
                    top_k=int(k),
                    d_model=int(d),
                    dtype=str(dtype).replace("torch.", ""),
                    t_perm_naive_ms=float(t_perm_naive),
                    t_perm_sorted_ms=float(t_perm_sorted),
                    t_theory_ms=float(t_theory_ms),
                    peak_bw_gbs=float(bw_gbs) if bw_gbs is not None else float("nan"),
                    sorted_achieved_bw_gbs=float(achieved_sorted),
                )
            )

    wrote_png = _try_plot_perm(args.csv_out, args.plot_out)
    print("perm_sweep: wrote CSV:", args.csv_out)
    if wrote_png:
        print("perm_sweep: wrote plot:", args.plot_out)
    else:
        print("perm_sweep: matplotlib not available; plot was not generated (CSV still written).")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--tokens", type=int, default=4096, help="T in the paper (number of tokens)")
    p.add_argument("--d_model", type=int, default=4096)
    p.add_argument("--experts", type=int, default=64, help="E")
    p.add_argument("--top_k", type=int, default=2, help="k")
    p.add_argument(
        "--perm_sweep",
        action="store_true",
        help="Sweep batch size and measure permutation latency (T_perm) for naive vs sorted pack.",
    )
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--seq_len", type=int, default=128, help="Used with --perm_sweep: tokens = batch_size * seq_len")
    p.add_argument("--csv_out", type=str, default="perm_efficiency.csv")
    p.add_argument("--plot_out", type=str, default="perm_efficiency.png")
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

    if args.perm_sweep:
        _run_perm_sweep(args, device=device, dtype=dtype)
        return

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


