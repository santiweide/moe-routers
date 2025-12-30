# SPDX-License-Identifier: Apache-2.0
"""
2-GPU PCIe Expert-Parallel (EP) benchmark for routing strategies.

Run with:
  torchrun --nproc_per_node 2 bench_2gpu_pcie.py ...

What it measures (forward-only, per-iter):
  - T_select_us : router selection time (topk/sinkhorn)
  - T_pack_us   : Python dispatch-pack time (gather + sort + split) for non-fused strategies
  - T_route_us  : T_select_us + T_pack_us  (for fused_select, ext returns select+count+scan+pack as one blob)
  - T_comm_us   : all_to_all_single time (REMOTE ONLY; self-split set to 0)
  - T_comp_us   : expert compute time on LOCAL tokens (optionally overlapped with comm)
  - T_dispatch(seq)        = T_route + T_comm + T_comp   (no overlap)
  - T_dispatch(ovl)        = T_route + wall(comm||local_comp)
  - eta_overlap            = (T_route + T_comm + T_comp) / T_dispatch(ovl)
  - Load skew over experts: max/mean and CV of per-expert route counts (aggregated across ranks)

Notes:
  - To make timing fair, we align semantics across strategies:
      * We DO NOT apply gating weights to x in pack for any strategy in this benchmark.
        (fused_select ext packs raw x twice; reordering weights exactly would require packing route IDs.)
  - Remote-only comm:
      * We keep local routes on-rank and do NOT send them via all_to_all.
      * all_to_all is used only for remote routes, with self split = 0.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.distributed as dist

from models.decoder_moe import SinkhornRouter, TorchTopKRouter


def _maybe_import_router_ext():
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
    t_pack_us: float
    t_route_us: float
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


def _grouped_local_compute(
    x_in: torch.Tensor,
    global_eid: torch.Tensor,
    *,
    rank: int,
    E_local: int,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
) -> None:
    """
    Compute expert MLP for tokens routed to this rank.
    x_in: [N, d]
    global_eid: [N] (global expert id)
    """
    if x_in.numel() == 0:
        return
    local_e = (global_eid - rank * E_local).to(torch.int64)
    # Defensive: keep only valid local ids
    mask = (local_e >= 0) & (local_e < E_local)
    if not bool(mask.all()):
        x_in = x_in[mask]
        local_e = local_e[mask]
        if x_in.numel() == 0:
            return

    # Sort by expert id for bucketed compute
    sidx = torch.argsort(local_e)
    lx = x_in.index_select(0, sidx)
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
    _ = out_local  # placeholder


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
        help="Run all supported strategies: naive_topk, fused_select, sinkhorn.",
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
    if k <= 0:
        raise ValueError("top_k must be >= 1")

    # Local tokens on each rank.
    x = torch.randn(args.tokens, args.d_model, device=device, dtype=dtype)

    # Router weight (shared across ranks for fairness): [E, d]
    w_router = torch.randn(E, args.d_model, device=device, dtype=dtype)

    # Local experts' MLP weights: [E_local, hidden, d] simplified as per-expert matrices.
    hidden = args.d_model  # keep simple
    w1 = torch.randn(E_local, hidden, args.d_model, device=device, dtype=dtype)
    w3 = torch.randn(E_local, hidden, args.d_model, device=device, dtype=dtype)
    w2 = torch.randn(E_local, args.d_model, hidden, device=device, dtype=dtype)

    ext = _maybe_import_router_ext()
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
        print("WARNING: strategy=fused_select requested but fused_select_cuda is not available; will fall back to naive_topk + python pack.")

    # -------------------------
    # Route + Pack (remote-only)
    # -------------------------
    def route_and_pack(
        logits: torch.Tensor,
        x_in: torch.Tensor,
        strategy: str,
    ) -> tuple[
        torch.Tensor,  # topk_idx [S,K] int64
        torch.Tensor,  # topk_w   [S,K] dtype (unused for pack in this benchmark)
        torch.Tensor,  # local_x  [N_local, d]
        torch.Tensor,  # local_eid_global [N_local]
        torch.Tensor,  # send_x_remote [N_remote, d] (to other rank only)
        torch.Tensor,  # send_eid_global_remote [N_remote]
        list[int],     # send_splits for all_to_all (len=2, self split == 0)
        float,         # t_select_us
        float,         # t_pack_us
        float,         # t_route_us
        Optional[torch.Tensor],  # counts_i32 (E,) if fused_select else None
        Optional[torch.Tensor],  # offsets_i32 (E+1,) if fused_select else None
    ]:
        S = x_in.size(0)

        # --- fused_select path: ext does select+count+scan+pack as one chunk ---
        if strategy == "fused_select" and ext is not None and k == 2:
            out = {}

            def do_ext():
                out["idx_i32"], out["w_f32"], out["counts_i32"], out["offsets_i32"], out["packed"] = \
                    ext.fused_select_forward(logits, x_in)

            t_route_us = _cuda_ms(do_ext) * 1000.0
            idx_i32 = out["idx_i32"]
            w_f32 = out["w_f32"]
            counts_i32 = out["counts_i32"]
            offsets_i32 = out["offsets_i32"]
            packed = out["packed"]

            topk_idx = idx_i32.to(torch.int64)
            topk_w = w_f32.to(dtype=logits.dtype)

            # Build per-row global expert-id aligned with packed layout (expert buckets contiguous)
            counts64 = counts_i32.to(torch.int64)
            eid_all = torch.repeat_interleave(
                torch.arange(E, device=device, dtype=torch.int64),
                counts64,
            )
            # offsets are prefix sums over experts; packed is ordered by expert id
            off = offsets_i32.to(torch.int64)

            # Local expert range for this rank
            e0 = rank * E_local
            e1 = (rank + 1) * E_local
            local_start = int(off[e0].item())
            local_end = int(off[e1].item())

            # Remote expert range (other rank)
            other = 1 - rank
            re0 = other * E_local
            re1 = (other + 1) * E_local
            remote_start = int(off[re0].item())
            remote_end = int(off[re1].item())

            local_x = packed[local_start:local_end]
            local_eid = eid_all[local_start:local_end]

            send_x_remote = packed[remote_start:remote_end]
            send_eid_remote = eid_all[remote_start:remote_end]

            n_remote = int(send_x_remote.size(0))
            # send_splits for all_to_all_single: [to_rank0, to_rank1]
            send_splits = [0, 0]
            send_splits[other] = n_remote
            send_splits[rank] = 0

            # We can't split select vs pack inside ext without extra APIs.
            return (
                topk_idx,
                topk_w,
                local_x,
                local_eid,
                send_x_remote,
                send_eid_remote,
                send_splits,
                t_route_us,  # treat as select
                0.0,
                t_route_us,
                counts_i32,
                offsets_i32,
            )

        # --- fallback path: selection then python pack (remote-only) ---
        topk_idx: torch.Tensor
        topk_w: torch.Tensor

        def do_select():
            nonlocal topk_idx, topk_w
            if strategy == "sinkhorn":
                topk_idx, topk_w = sinkhorn_router(logits, k)
            else:
                # naive_topk
                vals, idx = torch.topk(logits, k=k, dim=-1)
                topk_idx = idx.to(torch.int64)
                topk_w = torch.softmax(vals, dim=-1)

        t_select_us = _cuda_ms(do_select) * 1000.0

        packed_out = {}

        def do_pack():
            # Expand routes: [S,K] -> [R]
            token_ids = torch.arange(S, device=device, dtype=torch.int64).unsqueeze(1).expand(S, k).reshape(-1)
            expert_ids = topk_idx.reshape(-1).to(torch.int64)

            # IMPORTANT: align semantics with fused_select -> pack RAW x (no gating weights)
            x_route = x_in.index_select(0, token_ids)  # [R, d]

            dest_rank = (expert_ids // E_local).to(torch.int64)  # [R]
            sort_idx = torch.argsort(dest_rank)

            dest_s = dest_rank.index_select(0, sort_idx)
            eid_s = expert_ids.index_select(0, sort_idx)
            x_s = x_route.index_select(0, sort_idx)

            # Layout after sort: [dest=0 ...][dest=1 ...]
            n0 = int(torch.sum(dest_s == 0).item())
            n1 = int(dest_s.numel() - n0)

            # Local slice (kept on rank, not sent)
            if rank == 0:
                local_x = x_s[:n0]
                local_eid = eid_s[:n0]
                remote_x = x_s[n0:]
                remote_eid = eid_s[n0:]
                send_splits = [0, n1]  # send only to rank1
            else:
                local_x = x_s[n0:]
                local_eid = eid_s[n0:]
                remote_x = x_s[:n0]
                remote_eid = eid_s[:n0]
                send_splits = [n0, 0]  # send only to rank0

            packed_out["local_x"] = local_x
            packed_out["local_eid"] = local_eid
            packed_out["send_x_remote"] = remote_x
            packed_out["send_eid_remote"] = remote_eid
            packed_out["send_splits"] = send_splits

        t_pack_us = _cuda_ms(do_pack) * 1000.0
        t_route_us = t_select_us + t_pack_us

        return (
            topk_idx,
            topk_w,
            packed_out["local_x"],
            packed_out["local_eid"],
            packed_out["send_x_remote"],
            packed_out["send_eid_remote"],
            packed_out["send_splits"],
            t_select_us,
            t_pack_us,
            t_route_us,
            None,
            None,
        )

    # -------------------------
    # One iteration
    # -------------------------
    def one_iter(overlap: bool, strategy: str) -> EPResult:
        # 1) logits
        logits = x @ w_router.t()
        if args.skew_experts > 0 and args.skew_bias != 0.0:
            logits[:, : args.skew_experts] += args.skew_bias

        # 2) route + pack (remote-only)
        (
            topk_idx,
            topk_w,
            local_x,
            local_eid,
            send_remote,
            send_eid_remote,
            send_splits,
            t_select_us,
            t_pack_us,
            t_route_us,
            counts_i32,
            offsets_i32,
        ) = route_and_pack(logits, x, strategy=strategy)

        # 3) load skew from routes (global experts)
        if counts_i32 is not None:
            counts = counts_i32.to(torch.float32)
        else:
            expert_ids_flat = topk_idx.reshape(-1)
            counts = torch.bincount(expert_ids_flat, minlength=E).to(torch.float32)

        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        mean = float(counts.mean().item())
        std = float(counts.std(unbiased=False).item())
        load_cv = float(std / (mean + 1e-9))
        load_max_over_mean = float((counts.max().item() / (mean + 1e-9)))

        # 4) Comm setup (remote-only)
        send_counts = torch.tensor(send_splits, device=device, dtype=torch.int64)  # [2]
        recv_counts = torch.empty_like(send_counts)
        _all_to_all_single(recv_counts, send_counts)
        recv_splits = recv_counts.tolist()

        recv_remote = torch.empty((int(recv_counts.sum().item()), args.d_model), device=device, dtype=dtype)
        # Also exchange expert ids for realism (small overhead)
        send_eid_remote_i64 = send_eid_remote.to(torch.int64).contiguous()
        recv_eid_remote_i64 = torch.empty((int(recv_counts.sum().item()),), device=device, dtype=torch.int64)

        # 5) Comm + local compute (overlap best-effort)
        comm_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()

        def do_comm():
            # x
            _all_to_all_single(
                recv_remote,
                send_remote,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
            )
            # eid
            _all_to_all_single(
                recv_eid_remote_i64,
                send_eid_remote_i64,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
            )

        def do_local_comp():
            _grouped_local_compute(
                local_x,
                local_eid,
                rank=rank,
                E_local=E_local,
                w1=w1,
                w3=w3,
                w2=w2,
            )

        def do_remote_comp():
            # after comm, received tokens all belong to local experts for this rank
            _grouped_local_compute(
                recv_remote,
                recv_eid_remote_i64,
                rank=rank,
                E_local=E_local,
                w1=w1,
                w3=w3,
                w2=w2,
            )

        # Sequential: no overlap between comm and local compute
        def sequential_total() -> tuple[float, float, float]:
            t_comm = _cuda_ms(do_comm) * 1000.0
            # For sequential baseline, compute local and received after comm
            def comp_all():
                do_local_comp()
                do_remote_comp()

            t_comp = _cuda_ms(comp_all) * 1000.0
            t_dispatch = t_route_us + t_comm + t_comp
            return t_comm, t_comp, t_dispatch

        # Overlap: comm || local compute, then remote compute (cannot start before recv)
        def overlapped_total() -> tuple[float, float, float, float]:
            cur = torch.cuda.current_stream()

            # wall start/end
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            # comm events
            comm_start = torch.cuda.Event(enable_timing=True)
            comm_end = torch.cuda.Event(enable_timing=True)

            # local comp events
            lcomp_start = torch.cuda.Event(enable_timing=True)
            lcomp_end = torch.cuda.Event(enable_timing=True)

            # remote comp events (on current stream after comm join)
            rcomp_start = torch.cuda.Event(enable_timing=True)
            rcomp_end = torch.cuda.Event(enable_timing=True)

            start.record(cur)
            comm_stream.wait_event(start)
            compute_stream.wait_event(start)

            with torch.cuda.stream(comm_stream):
                comm_start.record(comm_stream)
                do_comm()
                comm_end.record(comm_stream)

            with torch.cuda.stream(compute_stream):
                lcomp_start.record(compute_stream)
                do_local_comp()
                lcomp_end.record(compute_stream)

            # Wait comm + local comp before remote comp
            cur.wait_stream(comm_stream)
            cur.wait_stream(compute_stream)

            rcomp_start.record(cur)
            do_remote_comp()
            rcomp_end.record(cur)

            end.record(cur)
            torch.cuda.synchronize()

            t_wall_us = float(start.elapsed_time(end)) * 1000.0
            t_comm_us = float(comm_start.elapsed_time(comm_end)) * 1000.0
            t_lcomp_us = float(lcomp_start.elapsed_time(lcomp_end)) * 1000.0
            t_rcomp_us = float(rcomp_start.elapsed_time(rcomp_end)) * 1000.0

            t_comp_us = t_lcomp_us + t_rcomp_us
            t_dispatch_us = t_route_us + t_comm_us + t_comp_us
            t_dispatch_overlap_us = t_route_us + t_wall_us
            return t_comm_us, t_comp_us, t_dispatch_us, t_dispatch_overlap_us

        if overlap:
            t_comm_us, t_comp_us, t_dispatch_us, t_dispatch_overlap_us = overlapped_total()
        else:
            t_comm_us, t_comp_us, t_dispatch_us = sequential_total()
            t_dispatch_overlap_us = t_dispatch_us

        eta = float((t_route_us + t_comm_us + t_comp_us) / max(t_dispatch_overlap_us, 1e-6))

        return EPResult(
            t_select_us=t_select_us,
            t_pack_us=t_pack_us,
            t_route_us=t_route_us,
            t_comm_us=t_comm_us,
            t_comp_us=t_comp_us,
            t_dispatch_us=t_dispatch_us,
            t_dispatch_overlap_us=t_dispatch_overlap_us,
            eta_overlap=eta,
            load_max_over_mean=load_max_over_mean,
            load_cv=load_cv,
        )

    # Warmup and run
    def run_strategy(strategy: str) -> tuple[list[EPResult], list[EPResult]]:
        for _ in range(args.warmup):
            _ = one_iter(overlap=False, strategy=strategy)
            _ = one_iter(overlap=True, strategy=strategy)
        res_seq = [one_iter(overlap=False, strategy=strategy) for _ in range(args.iters)]
        res_ovl = [one_iter(overlap=True, strategy=strategy) for _ in range(args.iters)]
        return res_seq, res_ovl

    def avg_field(rs: list[EPResult], name: str) -> float:
        return sum(getattr(r, name) for r in rs) / max(len(rs), 1)

    # Baseline route time (naive_topk + python pack), so ΔT_route makes sense.
    def baseline_route_us() -> float:
        logits = x @ w_router.t()
        if args.skew_experts > 0 and args.skew_bias != 0.0:
            logits[:, : args.skew_experts] += args.skew_bias

        # Warmup
        for _ in range(args.warmup):
            _ = route_and_pack(logits, x, "naive_topk")

        ts = []
        for _ in range(args.iters):
            *_, t_sel, t_pack, t_route, __, ___ = route_and_pack(logits, x, "naive_topk")
            ts.append(t_route)
        return sum(ts) / max(len(ts), 1)

    t_base_route = baseline_route_us()

    results: dict[str, tuple[list[EPResult], list[EPResult]]] = {}
    for s in strategies:
        dist.barrier()
        results[s] = run_strategy(s)
    dist.barrier()

    if rank == 0:
        print("=== 2-GPU EP (PCIe) Benchmark ===")
        if len(strategies) == 1:
            print("strategy:", strategies[0])
        else:
            print("strategies:", strategies)
        print(
            "dtype:", args.dtype,
            "T(tokens/rank):", args.tokens,
            "d_model:", args.d_model,
            "E:", args.experts,
            "k:", args.top_k,
        )
        print("skew_experts:", args.skew_experts, "skew_bias:", args.skew_bias)
        print("T_route_us(baseline naive_topk+pack):", t_base_route)

        print("")
        print("=== Dispatch-Path Latency (us, mean) ===")
        print(
            f"{'strategy':<12} "
            f"{'T_select':>10} {'T_pack':>10} {'T_route':>10} {'ΔT_route':>10} "
            f"{'T_comm':>10} {'T_comp':>10} "
            f"{'T_dispatch(seq)':>16} {'T_dispatch(ovl)':>16} {'eta':>8} "
            f"{'max/mean':>10} {'cv':>8}"
        )
        for s in strategies:
            res_seq, res_ovl = results[s]
            t_sel = avg_field(res_seq, "t_select_us")
            t_pack = avg_field(res_seq, "t_pack_us")
            t_route = avg_field(res_seq, "t_route_us")
            print(
                f"{s:<12} "
                f"{t_sel:10.1f} {t_pack:10.1f} {t_route:10.1f} {(t_route - t_base_route):10.1f} "
                f"{avg_field(res_seq, 't_comm_us'):10.1f} {avg_field(res_seq, 't_comp_us'):10.1f} "
                f"{avg_field(res_seq, 't_dispatch_us'):16.1f} {avg_field(res_ovl, 't_dispatch_overlap_us'):16.1f} {avg_field(res_ovl, 'eta_overlap'):8.3f} "
                f"{avg_field(res_seq, 'load_max_over_mean'):10.3f} {avg_field(res_seq, 'load_cv'):8.3f}"
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
