# SPDX-License-Identifier: Apache-2.0
"""
2-GPU PCIe Expert-Parallel (EP) benchmark for routing strategies.

Run with:
  torchrun --nproc_per_node 2 bench_2gpu_pcie.py ...

Measures (forward-only):
  - T_select and ΔT_select vs Top-k baseline
  - T_comm via all_to_all_single (tokens exchanged to expert owners)
  - Load skew: max(V)/mean(V), CV over expert token counts
  - Dispatch-path latency (selection + A2A dispatch + local expert compute):
      T_dispatch          : sequential
      T_dispatch_overlap  : overlap local expert compute with A2A dispatch (best-effort)
      eta_overlap         : (T_select + T_comm + T_comp) / T_dispatch_overlap
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from models.decoder_moe import SinkhornRouter, TorchTopKRouter
import fused_select_cuda as fsel


def _maybe_import_fused_select():
    try:
        import fused_select_cuda  # type: ignore

        return fused_select_cuda
    except Exception:
        return None


def _setup_dist() -> tuple[int, int, torch.device]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world, torch.device("cuda", rank)


def _all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    output_split_sizes: Optional[list[int]] = None,
    input_split_sizes: Optional[list[int]] = None,
):
    """
    Compatibility wrapper across PyTorch versions.

    Some versions use `output_split_sizes`/`input_split_sizes`, others accept
    `recv_splits`/`send_splits`. Prefer the canonical names and fall back.
    """
    if output_split_sizes is None and input_split_sizes is None:
        return dist.all_to_all_single(output, input)

    # Canonical keyword names (PyTorch)
    try:
        return dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )
    except TypeError:
        pass

    # Alternate keyword names used in some examples/older code
    try:
        return dist.all_to_all_single(
            output,
            input,
            recv_splits=output_split_sizes,
            send_splits=input_split_sizes,
        )
    except TypeError:
        pass

    # Positional fallback
    return dist.all_to_all_single(output, input, output_split_sizes, input_split_sizes)


def _cuda_ms(fn) -> float:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e))


@dataclass
class EPResult:
    t_select_us: float
    t_comm_us: float
    t_comp_us: float
    t_dispatch_us: float
    t_dispatch_overlap_us: float
    eta_overlap: float
    load_max_over_mean: float
    load_cv: float


def _expert_mlp(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    # SwiGLU: silu(x@w1^T) * (x@w3^T) -> @w2^T
    a = torch.nn.functional.silu(x @ w1.t())
    b = x @ w3.t()
    return (a * b) @ w2.t()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--tokens", type=int, default=4096, help="T (tokens per rank)")
    p.add_argument("--d_model", type=int, default=4096)
    p.add_argument("--experts", type=int, default=64, help="E (global experts)")
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="Run multiple strategies and print a comparison table on rank0 (overrides --strategy).",
    )
    p.add_argument(
        "--all_strategies",
        action="store_true",
        help="Run all supported strategies: naive_topk, fused_select (B), sinkhorn (C).",
    )
    p.add_argument(
        "--strategy",
        type=str,
        default="naive_topk",
        choices=["naive_topk", "fused_select", "sinkhorn"],
    )
    p.add_argument("--sinkhorn_iters", type=int, default=10)
    p.add_argument("--sinkhorn_temperature", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    # Synthetic skew (bias the first N experts)
    p.add_argument("--skew_experts", type=int, default=0)
    p.add_argument("--skew_bias", type=float, default=0.0)
    args = p.parse_args()
    rank, world, device = _setup_dist()
    if world != 2:
        raise RuntimeError("This benchmark is designed for world_size=2")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    torch.manual_seed(0)

    E = args.experts
    if E % world != 0:
        raise ValueError("experts must be divisible by world_size")
    E_local = E // world
    k = args.top_k

    # Local tokens on each rank.
    x = torch.randn(args.tokens, args.d_model, device=device, dtype=dtype)

    # Router weight (shared across ranks for fairness): [E, d]
    w_router = torch.randn(E, args.d_model, device=device, dtype=dtype)

    # Local experts' MLP weights: [E_local, hidden, d] simplified as per-expert matrices.
    hidden = args.d_model  # keep simple
    w1 = torch.randn(E_local, hidden, args.d_model, device=device, dtype=dtype)
    w3 = torch.randn(E_local, hidden, args.d_model, device=device, dtype=dtype)
    w2 = torch.randn(E_local, args.d_model, hidden, device=device, dtype=dtype)

    ext = _maybe_import_fused_select()
    torch_router = TorchTopKRouter()
    sinkhorn_router = SinkhornRouter(iters=args.sinkhorn_iters, temperature=args.sinkhorn_temperature)

    strategies_all = ["naive_topk", "fused_select", "sinkhorn"]
    if args.all_strategies:
        strategies = strategies_all
    elif args.strategies is not None:
        strategies = args.strategies
    else:
        strategies = [args.strategy]

    unknown = [s for s in strategies if s not in strategies_all]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Valid: {strategies_all}")

    if "fused_select" in strategies and ext is None and rank == 0:
        print("WARNING: strategy=fused_select requested but fused_select_cuda is not available; will fall back to naive_topk selection.")

    def route_select(logits: torch.Tensor, strategy: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor, float]:
        # Returns topk_idx [S,K] int64, topk_w [S,K] dtype, and t_select_us
        topk_idx: torch.Tensor
        topk_w: torch.Tensor
        st = strategy or args.strategy

        def do():
            nonlocal topk_idx, topk_w
            if st == "sinkhorn":
                topk_idx, topk_w = sinkhorn_router(logits, k)
                return
            if st == "fused_select" and ext is not None and k <= 8:
                idx_i32, w_f32 = fsel.fused_select_forward(logits, int(k))
                topk_idx = idx_i32.to(torch.int64)
                topk_w = w_f32.to(dtype=logits.dtype)
                return
            # naive_topk
            vals, idx = torch.topk(logits, k=k, dim=-1)
            topk_idx = idx.to(torch.int64)
            topk_w = torch.softmax(vals, dim=-1)

        t_ms = _cuda_ms(do)
        return topk_idx, topk_w, t_ms * 1000.0

    def one_iter(overlap: bool, strategy: str) -> EPResult:
        # 1) logits
        logits = x @ w_router.t()
        if args.skew_experts > 0 and args.skew_bias != 0.0:
            logits[:, : args.skew_experts] += args.skew_bias

        # 2) selection timing
        topk_idx, topk_w, t_select_us = route_select(logits, strategy=strategy)

        # 3) compute load skew from routes (global experts)
        expert_ids = topk_idx.reshape(-1)
        counts = torch.bincount(expert_ids, minlength=E).to(torch.float32)  # per-expert routes (local tokens only)
        # Aggregate across ranks
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        mean = float(counts.mean().item())
        std = float(counts.std(unbiased=False).item())
        load_cv = float(std / (mean + 1e-9))
        load_max_over_mean = float((counts.max().item() / (mean + 1e-9)))

        # 4) pack tokens by destination rank
        S = x.size(0)
        token_ids = torch.arange(S, device=device, dtype=torch.int64).unsqueeze(1).expand(S, k).reshape(-1)
        expert_ids = topk_idx.reshape(-1).to(torch.int64)
        weights = topk_w.reshape(-1).unsqueeze(-1)
        x_route = x.index_select(0, token_ids) * weights  # [R, d]

        dest_rank = (expert_ids // E_local).to(torch.int64)  # [R]
        # Sort by dest_rank so we can slice contiguous send buffers
        sort_idx = torch.argsort(dest_rank)
        dest_rank = dest_rank.index_select(0, sort_idx)
        expert_ids = expert_ids.index_select(0, sort_idx)
        x_route = x_route.index_select(0, sort_idx)

        # Split into send buffers for each rank
        send_counts = torch.bincount(dest_rank, minlength=world).to(torch.int64)  # [2]
        send_splits = send_counts.tolist()
        # all_to_all requires recv splits too; exchange counts first
        recv_counts = torch.empty_like(send_counts)
        _all_to_all_single(recv_counts, send_counts)
        recv_splits = recv_counts.tolist()

        # Build contiguous send tensor
        send = x_route
        recv = torch.empty((int(recv_counts.sum().item()), args.d_model), device=device, dtype=dtype)

        # 5) Comm + compute
        # Local part: slice routes that stay on this rank
        local_mask = dest_rank == rank
        local_x = send[local_mask]
        local_e = expert_ids[local_mask] - rank * E_local  # local expert id in [0, E_local)

        # Remote part: send everything; receiver will compute on received tokens.
        # For simplicity we compute only after receive. Overlap mode computes local while comm in flight.
        comm_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()

        t_comm_us = 0.0
        t_comp_us = 0.0

        def do_comm():
            _all_to_all_single(recv, send, output_split_sizes=recv_splits, input_split_sizes=send_splits)

        def do_local_comp():
            # Group by local expert (bucket) and run per-expert batched GEMM
            if local_x.numel() == 0:
                return
            # sort by expert
            sidx = torch.argsort(local_e)
            lx = local_x.index_select(0, sidx)
            le = local_e.index_select(0, sidx)
            counts_e = torch.bincount(le, minlength=E_local)
            start = 0
            out_local = torch.empty_like(lx)
            for e in range(E_local):
                c = int(counts_e[e].item())
                if c == 0:
                    continue
                end = start + c
                out_local[start:end] = _expert_mlp(lx[start:end], w1[e], w3[e], w2[e])
                start = end
            _ = out_local  # placeholder (not used further)

        # Timing total sequential vs overlap
        def sequential_total() -> tuple[float, float, float]:
            # comm
            t_comm = _cuda_ms(do_comm) * 1000.0
            # compute on recv (treat all received routes as local now, but we don't track original expert ids here)
            # NOTE: for benchmark purposes we approximate compute by running local comp on local_x only,
            # and treat recv compute as similar cost. This keeps the harness simple but still stresses comm.
            t_comp = _cuda_ms(do_local_comp) * 1000.0
            t_dispatch = t_select_us + t_comm + t_comp
            return t_comm, t_comp, t_dispatch

        def overlapped_total() -> tuple[float, float, float, float]:
            # Launch comm on comm stream, local compute on compute stream (single execution).
            cur = torch.cuda.current_stream()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            comm_start = torch.cuda.Event(enable_timing=True)
            comm_end = torch.cuda.Event(enable_timing=True)
            comp_start = torch.cuda.Event(enable_timing=True)
            comp_end = torch.cuda.Event(enable_timing=True)

            # Record a global start on current stream, then make both streams wait on it
            start.record(cur)
            comm_stream.wait_event(start)
            compute_stream.wait_event(start)

            with torch.cuda.stream(comm_stream):
                comm_start.record(comm_stream)
                do_comm()
                comm_end.record(comm_stream)

            with torch.cuda.stream(compute_stream):
                comp_start.record(compute_stream)
                do_local_comp()
                comp_end.record(compute_stream)

            # Join streams and record end
            cur.wait_stream(comm_stream)
            cur.wait_stream(compute_stream)
            end.record(cur)
            torch.cuda.synchronize()

            # Durations from the SAME overlapped execution
            t_wall_us = float(start.elapsed_time(end)) * 1000.0
            t_comm_us = float(comm_start.elapsed_time(comm_end)) * 1000.0
            t_comp_us = float(comp_start.elapsed_time(comp_end)) * 1000.0
            t_dispatch_us = t_select_us + t_comm_us + t_comp_us
            t_dispatch_overlap_us = t_select_us + t_wall_us
            return t_comm_us, t_comp_us, t_dispatch_us, t_dispatch_overlap_us

        if overlap:
            t_comm_us, t_comp_us, t_dispatch_us, t_dispatch_overlap_us = overlapped_total()
        else:
            t_comm_us, t_comp_us, t_dispatch_us = sequential_total()
            t_dispatch_overlap_us = t_dispatch_us

        eta = float((t_select_us + t_comm_us + t_comp_us) / max(t_dispatch_overlap_us, 1e-6))
        return EPResult(
            t_select_us=t_select_us,
            t_comm_us=t_comm_us,
            t_comp_us=t_comp_us,
            t_dispatch_us=t_dispatch_us,
            t_dispatch_overlap_us=t_dispatch_overlap_us,
            eta_overlap=eta,
            load_max_over_mean=load_max_over_mean,
            load_cv=load_cv,
        )

    # Warmup
    def run_strategy(strategy: str) -> tuple[list[EPResult], list[EPResult]]:
        for _ in range(args.warmup):
            _ = one_iter(overlap=False, strategy=strategy)
            _ = one_iter(overlap=True, strategy=strategy)
        res_seq = [one_iter(overlap=False, strategy=strategy) for _ in range(args.iters)]
        res_ovl = [one_iter(overlap=True, strategy=strategy) for _ in range(args.iters)]
        return res_seq, res_ovl

    def avg_field(rs, name: str) -> float:
        return sum(getattr(r, name) for r in rs) / max(len(rs), 1)

    # Always compute baseline selection time (naive_topk) so ΔT_select makes sense
    # even if user doesn't include naive_topk in the strategy list.
    def baseline_select_us() -> float:
        # Use a fixed logits tensor to isolate selection overhead.
        logits = x @ w_router.t()
        if args.skew_experts > 0 and args.skew_bias != 0.0:
            logits[:, : args.skew_experts] += args.skew_bias
        # Warmup
        for _ in range(args.warmup):
            _ = route_select(logits, "naive_topk")
        ts = [route_select(logits, "naive_topk")[2] for _ in range(args.iters)]
        return sum(ts) / max(len(ts), 1)

    t_base_select = baseline_select_us()

    results: dict[str, tuple[list[EPResult], list[EPResult]]] = {}
    for s in strategies:
        # Keep ranks in lockstep to avoid NCCL oddities and make logs readable.
        dist.barrier()
        results[s] = run_strategy(s)
    dist.barrier()

    if rank == 0:
        print("=== 2-GPU EP (PCIe) Benchmark ===")
        if len(strategies) == 1:
            print("strategy:", strategies[0])
        else:
            print("strategies:", strategies)
        print("dtype:", args.dtype, "T(tokens/rank):", args.tokens, "d_model:", args.d_model, "E:", args.experts, "k:", args.top_k)
        print("skew_experts:", args.skew_experts, "skew_bias:", args.skew_bias)
        print("T_select_us(baseline naive_topk):", t_base_select)

        if len(strategies) == 1:
            s = strategies[0]
            res_seq, res_ovl = results[s]
            t_sel = avg_field(res_seq, "t_select_us")
            delta_t_select = t_sel - t_base_select
            print("T_select_us(mean):", t_sel, "ΔT_select_us:", delta_t_select)
            print("T_comm_us(mean):", avg_field(res_seq, "t_comm_us"))
            print("load_max_over_mean(mean):", avg_field(res_seq, "load_max_over_mean"))
            print("load_cv(mean):", avg_field(res_seq, "load_cv"))
            print("T_dispatch_us(mean, sequential):", avg_field(res_seq, "t_dispatch_us"))
            print("T_dispatch_us(mean, overlapped):", avg_field(res_ovl, "t_dispatch_overlap_us"))
            print("eta_overlap(mean):", avg_field(res_ovl, "eta_overlap"))
        else:
            print("")
            print("=== Dispatch-Path Latency (us, mean) ===")
            print(
                f"{'strategy':<12} {'T_select':>10} {'ΔT_select':>10} {'T_comm':>10} {'T_dispatch(seq)':>16} {'T_dispatch(ovl)':>16} {'eta':>8}"
            )
            for s in strategies:
                res_seq, res_ovl = results[s]
                t_sel = avg_field(res_seq, "t_select_us")
                print(
                    f"{s:<12} {t_sel:10.1f} {(t_sel - t_base_select):10.1f} {avg_field(res_seq, 't_comm_us'):10.1f} {avg_field(res_seq, 't_dispatch_us'):16.1f} {avg_field(res_ovl, 't_dispatch_overlap_us'):16.1f} {avg_field(res_ovl, 'eta_overlap'):8.3f}"
                )

if __name__ == "__main__":
    # Make NCCL a bit more verbose if desired
    if "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ:
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = os.environ.get("NCCL_ASYNC_ERROR_HANDLING", "1")
    # Avoid deprecated env var warning on newer PyTorch.
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


