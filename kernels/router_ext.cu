#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

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

template <>
__device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 v) {
  // Use ATen's bf16 conversion; avoids depending on __nv_bfloat16 internal layout.
  return static_cast<float>(v);
}

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

template <int K>
__device__ __forceinline__ void init_topk(float* v, int* i) {
  #pragma unroll
  for (int j = 0; j < K; ++j) {
    v[j] = -INFINITY;
    i[j] = -1;
  }
}

template <int K>
__device__ __forceinline__ void insert_topk_k(float val, int idx, float* topv, int* topi) {
  // Keep arrays sorted descending.
  int pos = K;
  #pragma unroll
  for (int j = 0; j < K; ++j) {
    if (val > topv[j]) {
      pos = j;
      break;
    }
  }
  if (pos == K) return;
  for (int j = K - 1; j > pos; --j) {
    topv[j] = topv[j - 1];
    topi[j] = topi[j - 1];
  }
  topv[pos] = val;
  topi[pos] = idx;
}

template <int K>
__device__ __forceinline__ void merge_two_lists(float* va, int* ia, const float* vb, const int* ib) {
  // Merge list_b into list_a and keep top-K in list_a.
  float tv[2 * K];
  int ti[2 * K];
  #pragma unroll
  for (int j = 0; j < K; ++j) {
    tv[j] = va[j];
    ti[j] = ia[j];
    tv[K + j] = vb[j];
    ti[K + j] = ib[j];
  }

  // Select top-K (descending) via small selection sort.
  #pragma unroll
  for (int m = 0; m < K; ++m) {
    int best = m;
    #pragma unroll
    for (int j = m + 1; j < 2 * K; ++j) {
      if (tv[j] > tv[best]) best = j;
    }
    // swap
    float fv = tv[m];
    int fi = ti[m];
    tv[m] = tv[best];
    ti[m] = ti[best];
    tv[best] = fv;
    ti[best] = fi;
  }

  #pragma unroll
  for (int j = 0; j < K; ++j) {
    va[j] = tv[j];
    ia[j] = ti[j];
  }
}

template <typename scalar_t, int K>
__global__ void topk_router_kernel_warp_reduce(
    const scalar_t* __restrict__ logits,
    int32_t* __restrict__ out_idx,
    float* __restrict__ out_w,
    int tokens,
    int experts) {
  int t = blockIdx.x;
  if (t >= tokens) return;

  float local_v[K];
  int local_i[K];
  init_topk<K>(local_v, local_i);

  const scalar_t* row = logits + static_cast<int64_t>(t) * experts;
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    float v = to_float<scalar_t>(row[e]);
    insert_topk_k<K>(v, e, local_v, local_i);
  }

  // Warp-level reduction with shuffles: merge top-K lists across lanes.
  const unsigned mask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_v[K];
    int other_i[K];
    if (lane + offset < 32) {
      #pragma unroll
      for (int j = 0; j < K; ++j) {
        other_v[j] = __shfl_down_sync(mask, local_v[j], offset);
        other_i[j] = __shfl_down_sync(mask, local_i[j], offset);
      }
    } else {
      #pragma unroll
      for (int j = 0; j < K; ++j) {
        other_v[j] = -INFINITY;
        other_i[j] = -1;
      }
    }
    merge_two_lists<K>(local_v, local_i, other_v, other_i);
  }

  // Shared memory for per-warp winners (lane0 holds the warp top-K).
  __shared__ float sh_v[(128 / 32) * MAX_K];
  __shared__ int sh_i[(128 / 32) * MAX_K];
  if (lane == 0) {
    #pragma unroll
    for (int j = 0; j < K; ++j) {
      sh_v[warp * MAX_K + j] = local_v[j];
      sh_i[warp * MAX_K + j] = local_i[j];
    }
  }
  __syncthreads();

  // Final reduction in warp 0: reduce over num_warps lists.
  if (warp == 0) {
    float v[K];
    int i[K];
    init_topk<K>(v, i);

    const int num_warps = blockDim.x / 32;
    if (lane < num_warps) {
      #pragma unroll
      for (int j = 0; j < K; ++j) {
        v[j] = sh_v[lane * MAX_K + j];
        i[j] = sh_i[lane * MAX_K + j];
      }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other_v[K];
      int other_i[K];
      if (lane + offset < 32) {
        #pragma unroll
        for (int j = 0; j < K; ++j) {
          other_v[j] = __shfl_down_sync(mask, v[j], offset);
          other_i[j] = __shfl_down_sync(mask, i[j], offset);
        }
      } else {
        #pragma unroll
        for (int j = 0; j < K; ++j) {
          other_v[j] = -INFINITY;
          other_i[j] = -1;
        }
      }
      merge_two_lists<K>(v, i, other_v, other_i);
    }

    if (lane == 0) {
      // Softmax over top-k logits -> weights.
      float m = v[0];
      #pragma unroll
      for (int j = 1; j < K; ++j) m = fmaxf(m, v[j]);
      float denom = 0.0f;
      float exps[K];
      #pragma unroll
      for (int j = 0; j < K; ++j) {
        float ev = __expf(v[j] - m);
        exps[j] = ev;
        denom += ev;
      }
      denom = fmaxf(denom, 1e-9f);

      int32_t* idx_row = out_idx + static_cast<int64_t>(t) * K;
      float* w_row = out_w + static_cast<int64_t>(t) * K;
      #pragma unroll
      for (int j = 0; j < K; ++j) {
        idx_row[j] = static_cast<int32_t>(i[j]);
        w_row[j] = exps[j] / denom;
      }
    }
  }
}

template <typename scalar_t, int K, int BLOCK_THREADS = 128>
__global__ void topk_router_kernel_cub_sort(
    const scalar_t* __restrict__ logits,
    int32_t* __restrict__ out_idx,
    float* __restrict__ out_w,
    int tokens,
    int experts) {
  int t = blockIdx.x;
  if (t >= tokens) return;

  // One item per thread (pad with -inf).
  int e = threadIdx.x;
  float key = -INFINITY;
  int val = e;
  if (e < experts) {
    const scalar_t* row = logits + static_cast<int64_t>(t) * experts;
    key = to_float<scalar_t>(row[e]);
  }

  using BlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, 1, int>;
  __shared__ typename BlockSort::TempStorage temp_storage;
  BlockSort(temp_storage).SortPairsDescending(key, val);

  __shared__ float topv[MAX_K];
  __shared__ int topi[MAX_K];
  if (threadIdx.x < K) {
    topv[threadIdx.x] = key;
    topi[threadIdx.x] = val;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float m = topv[0];
    #pragma unroll
    for (int j = 1; j < K; ++j) m = fmaxf(m, topv[j]);
    float denom = 0.0f;
    float exps[K];
    #pragma unroll
    for (int j = 0; j < K; ++j) {
      float ev = __expf(topv[j] - m);
      exps[j] = ev;
      denom += ev;
    }
    denom = fmaxf(denom, 1e-9f);

    int32_t* idx_row = out_idx + static_cast<int64_t>(t) * K;
    float* w_row = out_w + static_cast<int64_t>(t) * K;
    #pragma unroll
    for (int j = 0; j < K; ++j) {
      idx_row[j] = static_cast<int32_t>(topi[j]);
      w_row[j] = exps[j] / denom;
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

  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits.scalar_type(),
      "topk_router_forward_cuda",
      [&] {
        // Strategy selection:
        // - experts <= 128: CUB BlockRadixSort (fast full sort in-block)
        // - otherwise     : warp-shuffle reduction (merge top-K lists)
        const bool use_cub = (experts <= 128);

        switch (kk) {
          case 1:
            if (use_cub) {
              topk_router_kernel_cub_sort<scalar_t, 1><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            } else {
              topk_router_kernel_warp_reduce<scalar_t, 1><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            }
            break;
          case 2:
            if (use_cub) {
              topk_router_kernel_cub_sort<scalar_t, 2><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            } else {
              topk_router_kernel_warp_reduce<scalar_t, 2><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            }
            break;
          case 3:
            if (use_cub) {
              topk_router_kernel_cub_sort<scalar_t, 3><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            } else {
              topk_router_kernel_warp_reduce<scalar_t, 3><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            }
            break;
          case 4:
            if (use_cub) {
              topk_router_kernel_cub_sort<scalar_t, 4><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            } else {
              topk_router_kernel_warp_reduce<scalar_t, 4><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            }
            break;
          case 5:
            if (use_cub) {
              topk_router_kernel_cub_sort<scalar_t, 5><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            } else {
              topk_router_kernel_warp_reduce<scalar_t, 5><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            }
            break;
          case 6:
            if (use_cub) {
              topk_router_kernel_cub_sort<scalar_t, 6><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            } else {
              topk_router_kernel_warp_reduce<scalar_t, 6><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            }
            break;
          case 7:
            if (use_cub) {
              topk_router_kernel_cub_sort<scalar_t, 7><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            } else {
              topk_router_kernel_warp_reduce<scalar_t, 7><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            }
            break;
          case 8:
            if (use_cub) {
              topk_router_kernel_cub_sort<scalar_t, 8><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            } else {
              topk_router_kernel_warp_reduce<scalar_t, 8><<<blocks, threads, 0, stream>>>(
                  reinterpret_cast<const scalar_t*>(logits.data_ptr()),
                  idx.data_ptr<int32_t>(),
                  w.data_ptr<float>(),
                  tokens,
                  experts);
            }
            break;
          default:
            // k is validated in the C++ binding (k<=8), but keep a guard here too.
            TORCH_CHECK(false, "topk_router_forward_cuda: unsupported k (expected 1..8)");
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {idx, w};
}


