# SPDX-License-Identifier: Apache-2.0
"""
Build the CUDA router extension in-place.

Usage (from repo root):
  python setup_router_ext.py build_ext --inplace

This builds a Python module named `router_ext_cuda` which provides:
  - router_ext_cuda.forward(logits, k) -> (topk_idx[int32], topk_weight[float32])
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _parse_gcc_major(version_text: str) -> int | None:
    # Handles: "g++ (GCC) 11.2.0", "gcc (GCC) 9.3.0", etc.
    m = re.search(r"\b(GCC|gcc|g\+\+)\b.*?(\d+)\.(\d+)\.(\d+)", version_text)
    if m:
        return int(m.group(2))
    m = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", version_text)
    if m:
        return int(m.group(1))
    return None


def _compiler_version_ok(cxx: str) -> None:
    try:
        out = subprocess.check_output([cxx, "--version"], text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        raise RuntimeError(f"Failed to run CXX compiler '{cxx} --version'") from e
    major = _parse_gcc_major(out)
    if major is not None and major < 9:
        raise RuntimeError(
            f"Your C++ compiler appears to be too old (detected major={major}). "
            "PyTorch extensions require GCC/G++ 9+.\n"
            "Fix: load a newer compiler module and export CC/CXX/CUDAHOSTCXX, e.g.:\n"
            "  export CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=g++-11\n"
        )


# Allow overriding compilers from environment (useful on clusters).
if "ROUTER_EXT_CC" in os.environ:
    os.environ["CC"] = os.environ["ROUTER_EXT_CC"]
if "ROUTER_EXT_CXX" in os.environ:
    os.environ["CXX"] = os.environ["ROUTER_EXT_CXX"]
if "ROUTER_EXT_CUDAHOSTCXX" in os.environ:
    os.environ["CUDAHOSTCXX"] = os.environ["ROUTER_EXT_CUDAHOSTCXX"]

_compiler_version_ok(os.environ.get("CXX", "g++"))

# Optionally force NVCC host compiler bindir (maps to nvcc --compiler-bindir).
ccbin = os.environ.get("ROUTER_EXT_CCBIN")
if not ccbin:
    # If user set CUDAHOSTCXX to an absolute path, infer bindir from it.
    ch = os.environ.get("CUDAHOSTCXX")
    if ch and ("/" in ch):
        ccbin = str(Path(ch).resolve().parent)

nvcc_flags = ["-O3", "--use_fast_math"]
if ccbin:
    nvcc_flags.extend(["--compiler-bindir", ccbin])

setup(
    name="router_ext_cuda",
    ext_modules=[
        CUDAExtension(
            name="router_ext_cuda",
            # IMPORTANT: basenames must be distinct or ninja will emit colliding .o files.
            sources=["kernels/router_ext_bindings.cpp", "kernels/router_ext.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


