# SPDX-License-Identifier: Apache-2.0
"""
Standalone minimal runnable demo: layernum=1 model + a custom Triton op (with Torch fallback).

Run (CUDA + Triton if available):
  python run.py --device cuda --dtype fp16

Run (CPU fallback):
  python run.py --device cpu --dtype fp32
"""

from __future__ import annotations

import argparse
import time

import torch

from model import OneLayerModel, _parse_dtype
from triton_compat import HAS_TRITON


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    dtype = _parse_dtype(args.dtype)
    x = torch.randn(args.batch, args.hidden, device=device, dtype=dtype)

    model = OneLayerModel(args.hidden).to(device=device, dtype=dtype).eval()

    # Warmup
    with torch.inference_mode():
        for _ in range(args.warmup):
            y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark-ish
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.iters):
            y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Print a small checksum so you can see it's doing real work.
    print("device:", device)
    print("dtype:", dtype)
    print("HAS_TRITON:", HAS_TRITON)
    print("x:", tuple(x.shape))
    print("y:", tuple(y.shape))
    print("y.mean:", float(y.float().mean().cpu()))
    print("latency_ms/iter:", (t1 - t0) * 1000.0 / max(args.iters, 1))


if __name__ == "__main__":
    main()


