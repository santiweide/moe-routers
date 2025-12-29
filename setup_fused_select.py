from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 确保指向正确的 cutlass/include 路径
cutlass_dir = os.path.join(os.path.dirname(__file__), "third_party", "cutlass", "include")

# 统一的编译参数
common_nvcc_flags = [
    '-O3', '--use_fast_math', '-lineinfo',
    '-std=c++17',  # <---【关键】CUTLASS 需要 C++17
    '-gencode=arch=compute_80,code=sm_80',
    '-gencode=arch=compute_90,code=sm_90',
    # <---【关键】防止 PyTorch 和 CUDA 之间的 half 运算符冲突
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
]

setup(
    name='moe_extensions', # 包名
    ext_modules=[
        # 模块 1: Router (Fused Select)
        CUDAExtension(
            name='fused_select_cuda',
            sources=['kernels/fused_select.cu'],
            # fused_select 现在移除了 cutlass 依赖，这里不加 include 也可以，加了也没事
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': common_nvcc_flags
            }
        ),
        # 模块 2: GEMM (Grouped FFN)
        CUDAExtension(
            name='grouped_ffn_cuda',
            sources=['kernels/grouped_ffn.cu'],
            include_dirs=[cutlass_dir], # <---【必须】Grouped GEMM 依赖 Cutlass 头文件
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'], # Host 编译器也要 C++17
                'nvcc': common_nvcc_flags
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)