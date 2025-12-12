# SPDX-License-Identifier: Apache-2.0
"""
Local Triton compatibility shim for the standalone demo.

Goals:
- No dependency on vllm.*
- If Triton is not installed, importing this file should still succeed so the
  demo can run with a pure-Torch fallback.
"""

from __future__ import annotations

import types
from importlib.util import find_spec


HAS_TRITON = find_spec("triton") is not None


class _TritonLanguagePlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton.language")
        self.constexpr = None
        self.float16 = None
        self.bfloat16 = None
        self.float32 = None

        self.arange = None
        self.cdiv = None
        self.program_id = None
        self.load = None
        self.store = None
        self.exp = None


class _TritonPlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton")
        self.__version__ = "0.0.0"
        self.jit = self._dummy_decorator("jit")

    def _dummy_decorator(self, _name: str):
        def decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f

        return decorator

    @staticmethod
    def cdiv(a: int, b: int) -> int:
        return (a + b - 1) // b


if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
else:
    triton = _TritonPlaceholder()
    tl = _TritonLanguagePlaceholder()


__all__ = ["HAS_TRITON", "triton", "tl"]


