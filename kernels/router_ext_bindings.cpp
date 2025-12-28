#include <torch/extension.h>

#include <stdexcept>
#include <vector>

std::vector<torch::Tensor> topk_router_forward_cuda(torch::Tensor logits, int64_t k) {
  const int tokens  = static_cast<int>(logits.size(0));
  const int experts = static_cast<int>(logits.size(1));
  const int kk = static_cast<int>(k);
  TORCH_CHECK(1 <= kk && kk <= MAX_K, "k must be in [1, ", MAX_K, "]");

  auto idx = torch::empty({tokens, kk}, logits.options().dtype(torch::kInt32));
  auto w   = torch::empty({tokens, kk}, logits.options().dtype(torch::kFloat32));

  // 根据 experts 选择线程数：小维度用 128（给 CUB），大维度 256 更饱和
  const bool use_cub = (experts <= 128);
  constexpr int THREADS_CUB   = 128;
  constexpr int THREADS_SHFL  = 256;

  dim3 blocks(tokens);
  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  logits.scalar_type(), "topk_router_forward_cuda", [&] {
    switch (kk) {
      case 1:
        if (use_cub)
          topk_router_kernel_cub_sort<scalar_t,1,THREADS_CUB><<<blocks, THREADS_CUB, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        else
          topk_router_kernel_warp_reduce<scalar_t,1,THREADS_SHFL><<<blocks, THREADS_SHFL, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        break;
      case 2:
        if (use_cub)
          topk_router_kernel_cub_sort<scalar_t,2,THREADS_CUB><<<blocks, THREADS_CUB, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        else
          topk_router_kernel_warp_reduce<scalar_t,2,THREADS_SHFL><<<blocks, THREADS_SHFL, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        break;
      case 3:
        if (use_cub)
          topk_router_kernel_cub_sort<scalar_t,3,THREADS_CUB><<<blocks, THREADS_CUB, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        else
          topk_router_kernel_warp_reduce<scalar_t,3,THREADS_SHFL><<<blocks, THREADS_SHFL, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        break;
      case 4:
        if (use_cub)
          topk_router_kernel_cub_sort<scalar_t,4,THREADS_CUB><<<blocks, THREADS_CUB, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        else
          topk_router_kernel_warp_reduce<scalar_t,4,THREADS_SHFL><<<blocks, THREADS_SHFL, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        break;
      case 5:
        if (use_cub)
          topk_router_kernel_cub_sort<scalar_t,5,THREADS_CUB><<<blocks, THREADS_CUB, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        else
          topk_router_kernel_warp_reduce<scalar_t,5,THREADS_SHFL><<<blocks, THREADS_SHFL, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        break;
      case 6:
        if (use_cub)
          topk_router_kernel_cub_sort<scalar_t,6,THREADS_CUB><<<blocks, THREADS_CUB, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        else
          topk_router_kernel_warp_reduce<scalar_t,6,THREADS_SHFL><<<blocks, THREADS_SHFL, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        break;
      case 7:
        if (use_cub)
          topk_router_kernel_cub_sort<scalar_t,7,THREADS_CUB><<<blocks, THREADS_CUB, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        else
          topk_router_kernel_warp_reduce<scalar_t,7,THREADS_SHFL><<<blocks, THREADS_SHFL, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        break;
      case 8:
        if (use_cub)
          topk_router_kernel_cub_sort<scalar_t,8,THREADS_CUB><<<blocks, THREADS_CUB, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        else
          topk_router_kernel_warp_reduce<scalar_t,8,THREADS_SHFL><<<blocks, THREADS_SHFL, 0, stream>>>(
              logits.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), w.data_ptr<float>(), tokens, experts);
        break;
    }
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {idx, w};
}
