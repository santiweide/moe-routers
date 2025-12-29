#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#ifndef CUTLASS_CHECK
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }
#endif

// 使用 Sm80 配置以兼容 Sm80/Sm90
using ArchTag = cutlass::arch::Sm80;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t; 
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
  ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,
  ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,
  ElementOutput, LayoutC, ElementAccumulator,
  cutlass::arch::OpClassTensorOp, ArchTag,
  cutlass::gemm::GemmShape<128, 128, 32>,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementAccumulator>,
  cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
  4
>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

std::vector<torch::Tensor> grouped_ffn_forward(
    std::vector<torch::Tensor> inputs,
    std::vector<torch::Tensor> weights) 
{
    int problem_count = inputs.size();
    if (problem_count == 0) return {};

    std::vector<cutlass::gemm::GemmCoord> problem_sizes_host;
    std::vector<int64_t> lda_host, ldb_host, ldc_host;
    std::vector<ElementA *> ptr_A_host;
    std::vector<ElementB *> ptr_B_host;
    std::vector<ElementC *> ptr_C_host;

    std::vector<torch::Tensor> outputs;
    outputs.reserve(problem_count);

    auto opts = inputs[0].options();

    for (int i = 0; i < problem_count; ++i) {
        int M = inputs[i].size(0);
        int K = inputs[i].size(1);
        // FIX: Use size(1) for N because weights are [K, N]
        int N = weights[i].size(1); 

        auto out = torch::empty({M, N}, opts);
        outputs.push_back(out);

        problem_sizes_host.emplace_back(M, N, K);
        lda_host.push_back(inputs[i].stride(0));
        ldb_host.push_back(weights[i].stride(0));
        ldc_host.push_back(out.stride(0));

        ptr_A_host.push_back(reinterpret_cast<ElementA*>(inputs[i].data_ptr<at::Half>()));
        ptr_B_host.push_back(reinterpret_cast<ElementB*>(weights[i].data_ptr<at::Half>()));
        ptr_C_host.push_back(reinterpret_cast<ElementC*>(out.data_ptr<at::Half>()));
    }

    auto int64_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    
    torch::Tensor problem_sizes_dev = torch::empty(
        problem_count * sizeof(cutlass::gemm::GemmCoord), 
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    
    torch::Tensor lda_dev = torch::tensor(lda_host, int64_opts);
    torch::Tensor ldb_dev = torch::tensor(ldb_host, int64_opts);
    torch::Tensor ldc_dev = torch::tensor(ldc_host, int64_opts);
    
    // Copy pointers
    // Tip: Using a pre-allocated tensor for pointers is better, but this works for demo
    torch::Tensor ptr_A_dev = torch::empty(problem_count * sizeof(void*), torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    torch::Tensor ptr_B_dev = torch::empty(problem_count * sizeof(void*), torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    torch::Tensor ptr_C_dev = torch::empty(problem_count * sizeof(void*), torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    
    cudaMemcpy(problem_sizes_dev.data_ptr(), problem_sizes_host.data(), problem_count * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_A_dev.data_ptr(), ptr_A_host.data(), problem_count * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_B_dev.data_ptr(), ptr_B_host.data(), problem_count * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_C_dev.data_ptr(), ptr_C_host.data(), problem_count * sizeof(void*), cudaMemcpyHostToDevice);

    GemmGrouped gemm_op;
    
    typename GemmGrouped::Arguments args(
        reinterpret_cast<cutlass::gemm::GemmCoord*>(problem_sizes_dev.data_ptr()),
        problem_count,
        256, 
        {1.0f, 0.0f},              
        reinterpret_cast<ElementA**>(ptr_A_dev.data_ptr()),
        reinterpret_cast<ElementB**>(ptr_B_dev.data_ptr()),
        reinterpret_cast<ElementC**>(ptr_C_dev.data_ptr()),
        reinterpret_cast<ElementC**>(ptr_C_dev.data_ptr()), 
        (int64_t*)lda_dev.data_ptr(),
        (int64_t*)ldb_dev.data_ptr(),
        (int64_t*)ldc_dev.data_ptr(),
        (int64_t*)ldc_dev.data_ptr()
    );

    size_t ws_size = gemm_op.get_workspace_size(args);
    auto workspace = torch::empty({(long)ws_size}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream();
    CUTLASS_CHECK(gemm_op.initialize(args, workspace.data_ptr(), stream));
    CUTLASS_CHECK(gemm_op.run(stream));

    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grouped_ffn_forward", &grouped_ffn_forward, "Grouped FFN Forward (CUTLASS)");
}