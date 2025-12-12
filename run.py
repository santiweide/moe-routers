# SPDX-License-Identifier: Apache-2.0
"""
Standalone minimal runnable demo:
- one_layer: Linear + residual + custom Triton op (with Torch fallback)
- transformer_decoder: decoder-only Transformer (RMSNorm + causal self-attn + SwiGLU)

Run (CUDA + Triton if available):
  python run.py --device cuda --dtype fp16

Run (CPU fallback):
  python run.py --device cpu --dtype fp32

Run the transformer decoder:
  python run.py --model transformer_decoder --device cuda --dtype fp16 --batch 2 --seq_len 128
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from model import OneLayerModel, _parse_dtype
from triton_compat import HAS_TRITON
from decoder_moe import DecoderMoEConfig, DecoderMoEModel
from transformer_decoder_model import TransformerDecoderConfig, TransformerDecoderModel


def _export_onnx_with_external_data(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    out_dir: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
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
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
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
    parser.add_argument(
        "--model",
        type=str,
        default="one_layer",
        choices=["one_layer", "transformer_decoder", "decoder_moe"],
        help="Which model to run.",
    )
    parser.add_argument("--batch", type=int, default=8)
    # One-layer demo params
    parser.add_argument("--hidden", type=int, default=4096, help="(one_layer) hidden size")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    # Transformer-decoder params
    parser.add_argument("--seq_len", type=int, default=128, help="(transformer_decoder) sequence length")
    parser.add_argument("--vocab_size", type=int, default=32000, help="(transformer_decoder) vocabulary size")
    parser.add_argument("--d_model", type=int, default=512, help="(transformer_decoder) model width")
    parser.add_argument("--n_heads", type=int, default=8, help="(transformer_decoder) number of heads")
    parser.add_argument("--n_layers", type=int, default=6, help="(transformer_decoder) number of layers")
    parser.add_argument("--mlp_hidden_dim", type=int, default=2048, help="(transformer_decoder) MLP hidden dim")
    parser.add_argument("--moe_n_experts", type=int, default=8, help="(decoder_moe) number of experts")
    parser.add_argument("--moe_top_k", type=int, default=2, help="(decoder_moe) top-k routing")
    parser.add_argument(
        "--moe_router",
        type=str,
        default="torch",
        choices=["torch", "cuda_ext"],
        help="(decoder_moe) router backend: torch or CUDA extension (router_ext_cuda).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="(transformer_decoder) maximum sequence length (used for absolute pos embedding cache size)",
    )
    parser.add_argument(
        "--absolute_pos_embedding",
        action="store_true",
        help="(transformer_decoder) use absolute positional embedding instead of RoPE",
    )
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
    if args.model == "one_layer":
        example = torch.randn(args.batch, args.hidden, device=device, dtype=dtype)
        model: torch.nn.Module = OneLayerModel(args.hidden).to(device=device, dtype=dtype).eval()
    elif args.model == "transformer_decoder":
        if args.seq_len <= 0:
            raise ValueError("--seq_len must be > 0")
        if args.vocab_size <= 0:
            raise ValueError("--vocab_size must be > 0")
        cfg = TransformerDecoderConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            mlp_hidden_dim=args.mlp_hidden_dim,
            max_seq_len=args.max_seq_len,
            absolute_pos_embedding=bool(args.absolute_pos_embedding),
            rope=not bool(args.absolute_pos_embedding),
        )
        example = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch, args.seq_len),
            device=device,
            dtype=torch.long,
        )
        model = TransformerDecoderModel(cfg).to(device=device, dtype=dtype).eval()
    else:
        if args.seq_len <= 0:
            raise ValueError("--seq_len must be > 0")
        if args.vocab_size <= 0:
            raise ValueError("--vocab_size must be > 0")
        if args.moe_n_experts <= 0:
            raise ValueError("--moe_n_experts must be > 0")
        if args.moe_top_k <= 0:
            raise ValueError("--moe_top_k must be > 0")
        cfg = DecoderMoEConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            moe_expert_hidden_dim=args.mlp_hidden_dim,
            moe_n_experts=args.moe_n_experts,
            moe_top_k=args.moe_top_k,
            max_seq_len=args.max_seq_len,
            absolute_pos_embedding=bool(args.absolute_pos_embedding),
            rope=not bool(args.absolute_pos_embedding),
        )
        example = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch, args.seq_len),
            device=device,
            dtype=torch.long,
        )
        model = DecoderMoEModel(cfg, router_backend=args.moe_router).to(device=device, dtype=dtype).eval()

    # Warmup
    with torch.inference_mode():
        for _ in range(args.warmup):
            y = model(example)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark-ish
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.iters):
            y = model(example)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Print a small checksum so you can see it's doing real work.
    print("model:", args.model)
    print("device:", device)
    print("dtype:", dtype)
    print("HAS_TRITON:", HAS_TRITON)
    if args.model == "one_layer":
        print("x:", tuple(example.shape))
        print("y:", tuple(y.shape))
        print("y.mean:", float(y.float().mean().cpu()))
    else:
        print("input_ids:", tuple(example.shape))
        print("logits:", tuple(y.shape))
        print("logits.mean:", float(y.float().mean().cpu()))
    print("latency_ms/iter:", (t1 - t0) * 1000.0 / max(args.iters, 1))

    if args.export_onnx_dir:
        # Export on CPU for maximum compatibility.
        # Note: some PyTorch CPU kernels may not support fp16/bf16; for ONNX export
        # Netron doesn't care about runtime dtype, so we export in fp32.
        export_dir = Path(args.export_onnx_dir)
        _ = _parse_dtype(args.dtype)  # keep arg validation consistent with run path
        export_dtype = torch.float32
        if args.model == "one_layer":
            export_model = OneLayerModel(args.hidden).to(device="cpu", dtype=export_dtype).eval()
            export_x = torch.randn(args.batch, args.hidden, device="cpu", dtype=export_dtype)
            _export_onnx_with_external_data(
                export_model,
                export_x,
                export_dir,
                input_names=["x"],
                output_names=["y"],
                dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
            )
        elif args.model == "transformer_decoder":
            cfg = TransformerDecoderConfig(
                vocab_size=args.vocab_size,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                mlp_hidden_dim=args.mlp_hidden_dim,
                max_seq_len=args.max_seq_len,
                absolute_pos_embedding=bool(args.absolute_pos_embedding),
                rope=not bool(args.absolute_pos_embedding),
            )
            export_model = TransformerDecoderModel(cfg).to(device="cpu", dtype=export_dtype).eval()
            export_ids = torch.randint(
                low=0,
                high=args.vocab_size,
                size=(args.batch, args.seq_len),
                device="cpu",
                dtype=torch.long,
            )
            _export_onnx_with_external_data(
                export_model,
                export_ids,
                export_dir,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "logits": {0: "batch", 1: "seq"},
                },
            )
        else:
            cfg = DecoderMoEConfig(
                vocab_size=args.vocab_size,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                moe_expert_hidden_dim=args.mlp_hidden_dim,
                moe_n_experts=args.moe_n_experts,
                moe_top_k=args.moe_top_k,
                max_seq_len=args.max_seq_len,
                absolute_pos_embedding=bool(args.absolute_pos_embedding),
                rope=not bool(args.absolute_pos_embedding),
            )
            # For export we force torch router (export path in MoE is dense anyway).
            export_model = DecoderMoEModel(cfg, router_backend="torch").to(device="cpu", dtype=export_dtype).eval()
            export_ids = torch.randint(
                low=0,
                high=args.vocab_size,
                size=(args.batch, args.seq_len),
                device="cpu",
                dtype=torch.long,
            )
            _export_onnx_with_external_data(
                export_model,
                export_ids,
                export_dir,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "logits": {0: "batch", 1: "seq"},
                },
            )
        print("Exported:")
        print("  __model__  :", (export_dir / "__model__").as_posix())
        print("  __params__ :", (export_dir / "__params__").as_posix())
        print("Open '__model__' in Netron.")


if __name__ == "__main__":
    main()


