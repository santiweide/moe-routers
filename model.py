# SPDX-License-Identifier: Apache-2.0
"""
Model definition for the standalone demo.

This file intentionally contains the *model structure* (and the small custom op
wrapper it uses), so scripts like `run.py` / `export_netron.py` can stay as thin
entrypoints.
"""

from __future__ import annotations

from typing import Any

import torch

from triton_compat import HAS_TRITON, tl, triton


@triton.jit
def _add_silu_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x + y

    # SiLU(z) = z * sigmoid(z)
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    out = z * sigmoid
    tl.store(out_ptr + offsets, out.to(OUT_DTYPE), mask=mask)


def _torch_add_silu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = x + y
    return z * torch.sigmoid(z)


def add_silu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute (x + y) * sigmoid(x + y), using Triton on CUDA when available."""
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: x={tuple(x.shape)} y={tuple(y.shape)}")
    if x.dtype != y.dtype:
        raise ValueError(f"dtype mismatch: x={x.dtype} y={y.dtype}")
    if x.device != y.device:
        raise ValueError(f"device mismatch: x={x.device} y={y.device}")

    # ONNX export cannot capture Triton kernels; force Torch path during export.
    if torch.onnx.is_in_onnx_export():
        return _torch_add_silu(x, y)

    # Fallback path: CPU or Triton unavailable.
    if (not x.is_cuda) or (not HAS_TRITON):
        return _torch_add_silu(x, y)

    dtype_map: dict[torch.dtype, Any] = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    if x.dtype not in dtype_map:
        raise ValueError(f"unsupported dtype for demo: {x.dtype}")

    out = torch.empty_like(x)
    n_elements = x.numel()
    x_ = x.contiguous().view(-1)
    y_ = y.contiguous().view(-1)
    out_ = out.view(-1)

    block = 1024
    grid = (triton.cdiv(n_elements, block),)
    _add_silu_kernel[grid](
        x_,
        y_,
        out_,
        n_elements,
        BLOCK_SIZE=block,
        OUT_DTYPE=dtype_map[x.dtype],
        num_warps=4,
    )
    return out


class _AddSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        # Inference-only demo: no backward.
        return add_silu(x, y)


class OneLayerModel(torch.nn.Module):
    """layernum=1: Linear + custom AddSiLU op (with residual)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        # residual + activation via custom op
        #
        # NOTE: during ONNX export, avoid torch.autograd.Function (may export as
        # PythonOp). We still reuse the same math via add_silu(), which itself
        # forces the Torch path when exporting.
        if torch.onnx.is_in_onnx_export():
            return add_silu(y, x)
        return _AddSiLU.apply(y, x)


def _parse_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype: {dtype}")


__all__ = [
    "HAS_TRITON",
    "OneLayerModel",
    "_parse_dtype",
    "add_silu",
]


