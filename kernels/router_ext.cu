#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int MAX_K = 8;

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<at::Half>(at::Half v) {
  return __half2float(reinterpret_cast<const __half&>(v));
}

#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
template <>
__device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 v) {
#if (__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__)
  // A100 supports bf16 natively.
  __nv_bfloat16 bv;
  bv.x = v.x;
  return __bfloat162float(bv);
#else
  // Fallback (shouldn't be hit on A100).
  return static_cast<float>(v);
#endif
}
#endif

__device__ __forceinline__ void insert_topk(float val, int idx, float* topv, int* topi, int k) {
  // Keep arrays sorted descending.
  int pos = k;
  for (int j = 0; j < k; ++j) {
    if (val > topv[j]) {
      pos = j;
      break;
    }
  }
  if (pos == k) return;
  for (int j = k - 1; j > pos; --j) {
    topv[j] = topv[j - 1];
    topi[j] = topi[j - 1];
  }
  topv[pos] = val;
  topi[pos] = idx;
}

template <typename scalar_t>
__global__ void topk_router_kernel(
    const scalar_t* __restrict__ logits,
    int32_t* __restrict__ out_idx,
    float* __restrict__ out_w,
    int tokens,
    int experts,
    int k) {
  int t = blockIdx.x;
  if (t >= tokens) return;

  // Per-thread local top-k.
  float local_v[MAX_K];
  int local_i[MAX_K];
  #pragma unroll
  for (int i = 0; i < MAX_K; ++i) {
    local_v[i] = -INFINITY;
    local_i[i] = -1;
  }

  const scalar_t* row = logits + static_cast<int64_t>(t) * experts;
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    float v = to_float<scalar_t>(row[e]);
    insert_topk(v, e, local_v, local_i, k);
  }

  extern __shared__ unsigned char smem[];
  float* sh_v = reinterpret_cast<float*>(smem);                     // [blockDim * MAX_K]
  int* sh_i = reinterpret_cast<int*>(sh_v + blockDim.x * MAX_K);    // [blockDim * MAX_K]

  int base = threadIdx.x * MAX_K;
  #pragma unroll
  for (int i = 0; i < MAX_K; ++i) {
    sh_v[base + i] = local_v[i];
    sh_i[base + i] = local_i[i];
  }
  __syncthreads();

  // Merge all candidates in thread0 (simple reference implementation).
  if (threadIdx.x == 0) {
    float topv[MAX_K];
    int topi[MAX_K];
    #pragma unroll
    for (int i = 0; i < MAX_K; ++i) {
      topv[i] = -INFINITY;
      topi[i] = -1;
    }

    for (int th = 0; th < blockDim.x; ++th) {
      int b = th * MAX_K;
      #pragma unroll
      for (int i = 0; i < MAX_K; ++i) {
        float v = sh_v[b + i];
        int idx = sh_i[b + i];
        if (idx >= 0) {
          insert_topk(v, idx, topv, topi, k);
        }
      }
    }

    // Softmax over top-k logits -> weights.
    float m = topv[0];
    for (int i = 1; i < k; ++i) m = fmaxf(m, topv[i]);
    float denom = 0.0f;
    float exps[MAX_K];
    for (int i = 0; i < k; ++i) {
      float ev = __expf(topv[i] - m);
      exps[i] = ev;
      denom += ev;
    }
    denom = fmaxf(denom, 1e-9f);

    int32_t* idx_row = out_idx + static_cast<int64_t>(t) * k;
    float* w_row = out_w + static_cast<int64_t>(t) * k;
    for (int i = 0; i < k; ++i) {
      idx_row[i] = static_cast<int32_t>(topi[i]);
      w_row[i] = exps[i] / denom;
    }
  }
}

}  // namespace

std::vector<torch::Tensor> topk_router_forward_cuda(torch::Tensor logits, int64_t k) {
  // logits: [tokens, experts] on CUDA
  const auto tokens = static_cast<int>(logits.size(0));
  const auto experts = static_cast<int>(logits.size(1));
  const int kk = static_cast<int>(k);

  auto idx = torch::empty({tokens, kk}, logits.options().dtype(torch::kInt32));
  auto w = torch::empty({tokens, kk}, logits.options().dtype(torch::kFloat32));

  const int threads = 128;
  const dim3 blocks(tokens);
  const size_t shmem = threads * MAX_K * (sizeof(float) + sizeof(int));

  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits.scalar_type(),
      "topk_router_forward_cuda",
      [&] {
        topk_router_kernel<scalar_t><<<blocks, threads, shmem, stream>>>(
            reinterpret_cast<const scalar_t*>(logits.data_ptr()),
            idx.data_ptr<int32_t>(),
            w.data_ptr<float>(),
            tokens,
            experts,
            kk);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {idx, w};
}


