# SPDX-License-Identifier: Apache-2.0
"""
Export the 1-layer model to a Netron-friendly ONNX pair:
- __model__   : ONNX graph
- __params__  : external weights blob (single file)

Run:
  python export_netron.py --out_dir /tmp/netron_model --batch 2 --hidden 128 --dtype fp16

Open in Netron:
  open the file "__model__" in the output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import OneLayerModel, _parse_dtype


def _export_onnx_with_external_data(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    out_dir: Path,
) -> None:
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
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Export on CPU for maximum compatibility.
    # Note: some PyTorch CPU kernels may not support fp16/bf16; for ONNX export
    # Netron doesn't care about runtime dtype, so we export in fp32 by default.
    _ = _parse_dtype(args.dtype)  # keep arg validation
    dtype = torch.float32
    model = OneLayerModel(args.hidden).to(device="cpu", dtype=dtype).eval()
    x = torch.randn(args.batch, args.hidden, device="cpu", dtype=dtype)

    _export_onnx_with_external_data(model, x, Path(args.out_dir))

    print("Exported:")
    print("  __model__  :", (Path(args.out_dir) / "__model__").as_posix())
    print("  __params__ :", (Path(args.out_dir) / "__params__").as_posix())
    print("Open '__model__' in Netron.")


if __name__ == "__main__":
    main()


