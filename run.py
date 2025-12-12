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
from pathlib import Path

import torch

from model import OneLayerModel, _parse_dtype
from triton_compat import HAS_TRITON


def _export_onnx_with_external_data(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    out_dir: Path,
) -> None:
    """Export an ONNX model split into graph (__model__) + external weights (__params__)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "__model__"
    params_name = "__params__"

    # 1) Export ONNX to a temp file first.
    tmp_model_path = out_dir / "_tmp_model.onnx"

    torch.onnx.export(
        model,
        example_input,
        tmp_model_path.as_posix(),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        # Be conservative for maximum compatibility with python-side constructs.
        dynamo=False,
    )

    # 2) Re-save with external data so we can split graph vs weights.
    try:
        import onnx  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'onnx'. Install it via: pip install onnx"
        ) from e

    m = onnx.load(tmp_model_path.as_posix())
    onnx.save_model(
        m,
        model_path.as_posix(),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=params_name,
    )

    # Cleanup temp file (leave __model__ / __params__)
    try:
        tmp_model_path.unlink()
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--export_onnx_dir",
        type=str,
        default=None,
        help=(
            "If set, export a Netron-friendly ONNX pair to this directory: "
            "__model__ (graph) + __params__ (external weights). Export is done on CPU/fp32."
        ),
    )
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

    if args.export_onnx_dir:
        # Export on CPU for maximum compatibility.
        # Note: some PyTorch CPU kernels may not support fp16/bf16; for ONNX export
        # Netron doesn't care about runtime dtype, so we export in fp32.
        export_dir = Path(args.export_onnx_dir)
        _ = _parse_dtype(args.dtype)  # keep arg validation consistent with run path
        export_dtype = torch.float32
        export_model = OneLayerModel(args.hidden).to(device="cpu", dtype=export_dtype).eval()
        export_x = torch.randn(args.batch, args.hidden, device="cpu", dtype=export_dtype)

        _export_onnx_with_external_data(export_model, export_x, export_dir)
        print("Exported:")
        print("  __model__  :", (export_dir / "__model__").as_posix())
        print("  __params__ :", (export_dir / "__params__").as_posix())
        print("Open '__model__' in Netron.")


if __name__ == "__main__":
    main()


