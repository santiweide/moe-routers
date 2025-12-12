# SPDX-License-Identifier: Apache-2.0
"""
Build the CUDA router extension in-place.

Usage (from repo root):
  python setup_router_ext.py build_ext --inplace

This builds a Python module named `router_ext_cuda` which provides:
  - router_ext_cuda.forward(logits, k) -> (topk_idx[int32], topk_weight[float32])
"""

from __future__ import annotations

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="router_ext_cuda",
    ext_modules=[
        CUDAExtension(
            name="router_ext_cuda",
            sources=["kernels/router_ext.cpp", "kernels/router_ext.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


