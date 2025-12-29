#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cub/cub.cuh>

#include <vector>
#include <algorithm>  // std::min
#include <cmath>      // INFINITY

// NOTE: Removed CUTLASS dependency for this file to avoid conflicts.

using torch::Tensor;

namespace {

constexpr int E = 64;   // experts
constexpr int K = 2;    // top-k
constexpr int WARP = 32;

// -------------------- utils --------------------
template <typename T>
__device__ __forceinline__ float to_f32(T v) { return static_cast<float>(v); }

template <>
__device__ __forceinline__ float to_f32<at::Half>(at::Half v) {
  return __half2float(*reinterpret_cast<const __half*>(&v));
}

template <>
__device__ __forceinline__ float to_f32<at::BFloat16>(at::BFloat16 v) {
  return static_cast<float>(v);
}

// -------------------- Top-2 over 64 experts --------------------
template <typename scalar_t, int WARPS_PER_CTA=4>
__global__ void top2_select_64_kernel(
    const scalar_t* __restrict__ logits, // [T, 64]
    int32_t* __restrict__ out_idx,       // [T, 2]
    float* __restrict__ out_w,           // [T, 2]
    int T)
{
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5; 
  const int t = (blockIdx.x * WARPS_PER_CTA) + warp;
  if (t >= T) return;

  const scalar_t* row = logits + (int64_t)t * E;

  float v0 = to_f32(row[lane]);         int i0 = lane;
  float v1 = to_f32(row[lane + 32]);    int i1 = lane + 32;

  // 1st argmax
  const unsigned mask = __activemask();
  float v = (v0 >= v1) ? v0 : v1;
  int   i = (v0 >= v1) ? i0 : i1;

  #pragma unroll
  for (int ofs = 16; ofs > 0; ofs >>= 1) {
    float ov = __shfl_xor_sync(mask, v, ofs);
    int   oi = __shfl_xor_sync(mask, i, ofs);
    if (ov > v) { v = ov; i = oi; }
  }
  float vmax = __shfl_sync(mask, v, 0);
  int   imax = __shfl_sync(mask, i, 0);

  // 2nd argmax
  if (i0 == imax) v0 = -INFINITY;
  if (i1 == imax) v1 = -INFINITY;

  float v2 = (v0 >= v1) ? v0 : v1;
  int   i2 = (v0 >= v1) ? i0 : i1;

  #pragma unroll
  for (int ofs = 16; ofs > 0; ofs >>= 1) {
    float ov = __shfl_xor_sync(mask, v2, ofs);
    int   oi = __shfl_xor_sync(mask, i2, ofs);
    if (ov > v2) { v2 = ov; i2 = oi; }
  }
  float smax = __shfl_sync(mask, v2, 0);
  int   i2max= __shfl_sync(mask, i2, 0);

  // softmax
  float m = fmaxf(vmax, smax);
  float e0 = __expf(vmax - m);
  float e1 = __expf(smax - m);
  float inv = 1.f / fmaxf(e0 + e1, 1e-9f);

  if (lane == 0) {
    out_idx[t*K + 0] = imax;
    out_idx[t*K + 1] = i2max;
    out_w  [t*K + 0] = e0 * inv;
    out_w  [t*K + 1] = e1 * inv;
  }
}

// -------------------- Pass 1: Count --------------------
__global__ void count_expert_kernel(
    const int32_t* __restrict__ expert_idx, 
    int* __restrict__ counts,               
    int T)
{
  extern __shared__ int sh[];
  int* hist = sh; 
  for (int e = threadIdx.x; e < E; e += blockDim.x) hist[e] = 0;
  __syncthreads();

  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < T; t += gridDim.x * blockDim.x) {
    int e0 = expert_idx[t*2 + 0];
    int e1 = expert_idx[t*2 + 1];
    atomicAdd(&hist[e0], 1);
    atomicAdd(&hist[e1], 1);
  }
  __syncthreads();

  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    atomicAdd(&counts[e], hist[e]);
  }
}

// -------------------- Pass 2: Pack --------------------
__global__ void pack_copy_kernel(
    const at::Half* __restrict__ x,           
    const int32_t* __restrict__ expert_idx,   
    const int* __restrict__ offsets,          
    int* __restrict__ counters,               
    at::Half* __restrict__ out,               
    int T, int d_model)
{
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane    = threadIdx.x & 31;
  if (warp_id >= T) return;

  const int32_t e0 = expert_idx[warp_id*2 + 0];
  const int32_t e1 = expert_idx[warp_id*2 + 1];

  const uint4* src = reinterpret_cast<const uint4*>(x + (int64_t)warp_id * d_model);
  const int vecs = (d_model * sizeof(at::Half)) / 16; 

  #pragma unroll 2
  for (int pick = 0; pick < 2; ++pick) {
    const int e = (pick == 0) ? e0 : e1;
    const int slot = atomicAdd(&counters[e], 1);
    const int dst_token = offsets[e] + slot;
    uint4* dst = reinterpret_cast<uint4*>(out + (int64_t)dst_token * d_model);

    for (int v = lane; v < vecs; v += WARP) {
      dst[v] = src[v];
    }
  }
}

} // namespace

// -------------------- Frontend --------------------
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
fused_select_forward_cuda(Tensor logits, Tensor x) {
  TORCH_CHECK(logits.is_cuda() && x.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be half");
  TORCH_CHECK(logits.dim()==2 && logits.size(1)==E, "logits shape [T,64]");
  TORCH_CHECK(x.dim()==2 && x.size(0)==logits.size(0), "x shape [T,d_model]");

  const int T = logits.size(0);
  const int d_model = x.size(1);

  auto idx = torch::empty({T, K}, logits.options().dtype(at::kInt));
  auto w   = torch::empty({T, K}, logits.options().dtype(at::kFloat));
  auto counts  = torch::zeros({E}, logits.options().dtype(at::kInt));
  auto offsets = torch::empty({E+1}, logits.options().dtype(at::kInt));

  auto stream = at::cuda::getDefaultCUDAStream();

  // 1. Top-2 Selection
  const int WARPS = 4;
  dim3 blocks_sel( (T + WARPS - 1) / WARPS );
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, logits.scalar_type(),
    "top2_select_64", [&]{
      top2_select_64_kernel<scalar_t, WARPS><<<blocks_sel, WARPS*WARP, 0, stream>>>(
        logits.data_ptr<scalar_t>(),
        idx.data_ptr<int32_t>(),
        w.data_ptr<float>(),
        T);
    });

  // 2. Count
  const int cta = 256;
  dim3 blocks_cnt( std::min<int>( (T + cta - 1) / cta, 1024) );
  size_t shmem = E * sizeof(int);
  count_expert_kernel<<<blocks_cnt, cta, shmem, stream>>>(
      idx.data_ptr<int32_t>(),
      counts.data_ptr<int>(),
      T);

  // 3. Scan (Fix: Use InclusiveSum to get offsets[1..E] correctly)
  {
    // offsets[0] = 0 (Start of Expert 0)
    AT_CUDA_CHECK(cudaMemsetAsync(offsets.data_ptr<int>(), 0, sizeof(int), stream));

    // InclusiveSum: counts -> offsets[1..E]
    // counts: [10, 20, 30] -> offsets+1: [10, 30, 60]
    // Result offsets: [0, 10, 30, 60]. 
    // offsets[0] is start of Exp0, offsets[1] start of Exp1... offsets[E] is Total.
    void* d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        d_temp, temp_bytes,
        counts.data_ptr<int>(),      // In
        offsets.data_ptr<int>() + 1, // Out
        E, stream);

    auto tmp = torch::empty({(long long)temp_bytes}, logits.options().dtype(at::kByte));
    d_temp = tmp.data_ptr();

    cub::DeviceScan::InclusiveSum(
        d_temp, temp_bytes,
        counts.data_ptr<int>(),
        offsets.data_ptr<int>() + 1,
        E, stream);
  }

  // 4. Get Total Size (offsets[E])
  int sum_counts_host = 0;
  AT_CUDA_CHECK(cudaMemcpyAsync(
      &sum_counts_host,
      offsets.data_ptr<int>() + E,
      sizeof(int),
      cudaMemcpyDeviceToHost,
      stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  // 5. Pack
  auto packed = torch::empty({sum_counts_host, d_model}, x.options()); 
  auto counters = torch::zeros({E}, counts.options()); 

  const int warps_pack = 4; 
  dim3 blocks_pack( (T + warps_pack - 1) / warps_pack );
  pack_copy_kernel<<<blocks_pack, warps_pack*WARP, 0, stream>>>(
      x.data_ptr<at::Half>(),
      idx.data_ptr<int32_t>(),
      offsets.data_ptr<int>(),
      counters.data_ptr<int>(),
      packed.data_ptr<at::Half>(),
      T, d_model);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {idx, w, counts, offsets, packed};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_select_forward", &fused_select_forward_cuda,
        "Fused Top2-Select + Count/Scan + Pack (E=64, k=2, half)");
}