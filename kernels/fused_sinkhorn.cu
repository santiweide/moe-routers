#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <limits>

using torch::Tensor;

namespace {

constexpr int E    = 64;  // experts
constexpr int K    = 2;   // top-k
constexpr int WARP = 32;

// -------------------- Utils --------------------
template <typename T>
__device__ __forceinline__ float to_f32(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_f32<at::Half>(at::Half v) {
  return __half2float(*reinterpret_cast<const __half*>(&v));
}

template <>
__device__ __forceinline__ float to_f32<at::BFloat16>(at::BFloat16 v) {
  return static_cast<float>(v);
}

__device__ __forceinline__ bool is_aligned_16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

// Float atomicMax implementation (using CAS)
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
    return old;
}

// -------------------- Sinkhorn Kernels --------------------

// 1. Row Normalization: L[t, e] = L[t, e] - LogSumExp_e(L[t, :])
// One warp per token row.
template <typename scalar_t>
__global__ void sinkhorn_row_norm_kernel(
    scalar_t* __restrict__ logits, // [T, 64] In/Out
    int T)
{
    const int t = blockIdx.x * blockDim.y + threadIdx.y;
    if (t >= T) return;

    const int lane = threadIdx.x; // 0..31
    scalar_t* row = logits + (int64_t)t * E;

    // Load row (64 elements)
    float v0 = to_f32(row[lane]);
    float v1 = to_f32(row[lane + 32]);

    // Find Max for stability
    float m = fmaxf(v0, v1);
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, offset));
    }
    float row_max = __shfl_sync(0xffffffff, m, 0);

    // Sum Exp
    float s = __expf(v0 - row_max) + __expf(v1 - row_max);
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        s += __shfl_xor_sync(0xffffffff, s, offset);
    }
    float row_sum = __shfl_sync(0xffffffff, s, 0);

    float lse = row_max + __logf(row_sum + 1e-9f);

    // Update in place
    row[lane]      = static_cast<scalar_t>(v0 - lse);
    row[lane + 32] = static_cast<scalar_t>(v1 - lse);
}

// 2. Col Stats: Compute Max and SumExp per expert column.
// Uses shared memory to reduce global atomic contention.
template <typename scalar_t>
__global__ void sinkhorn_col_stats_kernel(
    const scalar_t* __restrict__ logits, // [T, 64]
    float* __restrict__ col_max,         // [64]
    float* __restrict__ col_sum,         // [64]
    int T)
{
    // Block size is 256. We map threads to (t, e) where e = tid % 64.
    // This ensures coalesced reads: tid 0 reads (t, 0), tid 1 reads (t, 1)...
    const int tid = threadIdx.x;
    const int lane = tid & 63; // expert index 0..63
    
    extern __shared__ float smem[]; 
    // Layout: smem_max[64], smem_sum[64]
    float* s_max = smem;
    float* s_sum = smem + 64;

    if (tid < 64) {
        s_max[tid] = -INFINITY;
        s_sum[tid] = 0.0f;
    }
    __syncthreads();

    // Grid stride loop over T
    // Each block processes chunks of rows. 
    // Threads with same `lane` (expert id) accumulate locally.
    float local_max = -INFINITY;
    
    // Stride is blockDim.x / 64 rows at a time? No, simple linear index.
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // element index = idx.
    // row = idx / 64, col = idx % 64.
    
    for (int i = blockIdx.x * blockDim.x + tid; i < T * E; i += gridDim.x * blockDim.x) {
        float val = to_f32(logits[i]);
        local_max = fmaxf(local_max, val);
    }

    // Reduce max into shared memory using atomics (intra-block)
    // Note: Standard atomicMax for float is tricky, but E=64 is small. 
    // We can also just reduce within warp if we organize differently, but atomics are easiest here.
    atomicMaxFloat(&s_max[lane], local_max);
    
    __syncthreads();

    // Write block-local max to global (Pass 1: find global max)
    if (tid < 64) {
        atomicMaxFloat(&col_max[tid], s_max[tid]);
    }
}

// 2b. Col Stats Sum: After finding global max, compute sum exp.
template <typename scalar_t>
__global__ void sinkhorn_col_sum_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ col_max,
    float* __restrict__ col_sum,
    int T)
{
    const int tid = threadIdx.x;
    const int lane = tid & 63;
    
    extern __shared__ float s_sum[]; // [64]
    if (tid < 64) s_sum[tid] = 0.0f;
    __syncthreads();

    // Cache global max in shared or register?
    // All threads in a warp likely need different maxes, just read from global is fine (L2 cached).
    // Or better: read col_max into shared once.
    __shared__ float s_gmax[64];
    if (tid < 64) s_gmax[tid] = col_max[tid];
    __syncthreads();

    float my_gmax = s_gmax[lane];
    float local_sum = 0.0f;

    for (int i = blockIdx.x * blockDim.x + tid; i < T * E; i += gridDim.x * blockDim.x) {
        float val = to_f32(logits[i]);
        local_sum += __expf(val - my_gmax);
    }

    atomicAdd(&s_sum[lane], local_sum);
    __syncthreads();

    if (tid < 64) {
        atomicAdd(&col_sum[tid], s_sum[tid]);
    }
}

// 3. Col Update: L[t,e] -= (Max + Log(Sum))
template <typename scalar_t>
__global__ void sinkhorn_col_update_kernel(
    scalar_t* __restrict__ logits,
    const float* __restrict__ col_max,
    const float* __restrict__ col_sum,
    int T)
{
    const int tid = threadIdx.x;
    const int lane = tid & 63;
    
    // Load offsets once per block
    __shared__ float s_offset[64];
    if (tid < 64) {
        s_offset[tid] = col_max[tid] + __logf(col_sum[tid] + 1e-9f);
    }
    __syncthreads();

    float my_offset = s_offset[lane];

    for (int i = blockIdx.x * blockDim.x + tid; i < T * E; i += gridDim.x * blockDim.x) {
        float val = to_f32(logits[i]);
        logits[i] = static_cast<scalar_t>(val - my_offset);
    }
}

// -------------------- Existing Top-2 / Count / Pack --------------------
// (这些部分保持不变，仅复制必要的部分以保证完整性)

// Top-2 Select 64 Kernel (Modified slightly to accept mutable/temp logits if needed)
template <typename scalar_t, int WARPS_PER_CTA = 4>
__global__ void top2_select_64_kernel(
    const scalar_t* __restrict__ logits, // [T, 64]
    int32_t* __restrict__ out_idx,       // [T, 2]
    float* __restrict__ out_w,         // [T, 2]
    int T)
{
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int t    = (int)(blockIdx.x * WARPS_PER_CTA + warp);
  if (t >= T) return;

  const scalar_t* row = logits + (int64_t)t * E;
  float v0 = to_f32(row[lane]);      int i0 = lane;
  float v1 = to_f32(row[lane + 32]); int i1 = lane + 32;
  const unsigned mask = 0xffffffffu;
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
  float smax  = __shfl_sync(mask, v2, 0);
  int   i2max = __shfl_sync(mask, i2, 0);
  imax  = max(0, min(E - 1, imax));
  i2max = max(0, min(E - 1, i2max));
  
  // Softmax on top 2 (Standard gating behavior)
  float m   = fmaxf(vmax, smax);
  float e0  = __expf(vmax - m);
  float e1  = __expf(smax - m);
  float inv = 1.f / fmaxf(e0 + e1, 1e-9f);

  if (lane == 0) {
    out_idx[t*K + 0] = (int32_t)imax;
    out_idx[t*K + 1] = (int32_t)i2max;
    out_w  [t*K + 0] = e0 * inv;
    out_w  [t*K + 1] = e1 * inv;
  }
}

__global__ void count_expert_kernel(
    const int32_t* __restrict__ expert_idx, int32_t* __restrict__ counts, int T)
{
  extern __shared__ int32_t sh[];
  int32_t* hist = sh;
  for (int e = threadIdx.x; e < E; e += blockDim.x) hist[e] = 0;
  __syncthreads();
  for (int t = (int)(blockIdx.x * blockDim.x + threadIdx.x); t < T; t += (int)(gridDim.x * blockDim.x)) {
    int32_t e0 = expert_idx[t*2 + 0];
    int32_t e1 = expert_idx[t*2 + 1];
    if ((uint32_t)e0 < (uint32_t)E) atomicAdd(&hist[e0], 1);
    if ((uint32_t)e1 < (uint32_t)E) atomicAdd(&hist[e1], 1);
  }
  __syncthreads();
  for (int e = threadIdx.x; e < E; e += blockDim.x) atomicAdd(&counts[e], hist[e]);
}

__global__ void pack_copy_kernel(
    const at::Half* __restrict__ x, const int32_t* __restrict__ expert_idx,
    const int32_t* __restrict__ offsets, int32_t* __restrict__ counters,
    at::Half* __restrict__ out, int T, int d_model, int total_packed)
{
  const int warp_id = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int lane    = (int)(threadIdx.x & 31);
  if (warp_id >= T) return;
  const int32_t e0 = expert_idx[warp_id*2 + 0];
  const int32_t e1 = expert_idx[warp_id*2 + 1];
  const at::Half* src_h = x + (int64_t)warp_id * d_model;
  const bool can_vec = (d_model % 8 == 0) && is_aligned_16(src_h);
  #pragma unroll 2
  for (int pick = 0; pick < 2; ++pick) {
    const int32_t e = (pick == 0) ? e0 : e1;
    if ((uint32_t)e >= (uint32_t)E) continue;
    const int32_t slot = atomicAdd(&counters[e], 1);
    const int32_t dst_token = offsets[e] + slot;
    if ((uint32_t)dst_token >= (uint32_t)total_packed) continue;
    at::Half* dst_h = out + (int64_t)dst_token * d_model;
    const bool vec_ok = can_vec && is_aligned_16(dst_h);
    if (vec_ok) {
      const uint4* src = reinterpret_cast<const uint4*>(src_h);
      uint4* dst       = reinterpret_cast<uint4*>(dst_h);
      const int vecs   = d_model / 8;
      for (int v = lane; v < vecs; v += WARP) dst[v] = src[v];
    } else {
      for (int j = lane; j < d_model; j += WARP) dst_h[j] = src_h[j];
    }
  }
}

} // namespace

// -------------------- Frontend --------------------
// Added sinkhorn_iters argument
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
fused_sinkhorn_forward_cuda(Tensor logits, Tensor x, int sinkhorn_iters = 3) {
  TORCH_CHECK(logits.is_cuda() && x.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
  // Logits can be bf16 or half, we dispatch.
  
  const int T = (int)logits.size(0);
  const int d_model = (int)x.size(1);
  
  // Clone logits because Sinkhorn is in-place and destructive
  Tensor s_logits = logits.clone();

  auto idx     = torch::empty({T, K}, logits.options().dtype(at::kInt));
  auto w       = torch::empty({T, K}, logits.options().dtype(at::kFloat));
  auto counts  = torch::zeros({E},   logits.options().dtype(at::kInt));
  auto offsets = torch::empty({E + 1}, logits.options().dtype(at::kInt));

  // Buffers for Column Reduction
  auto col_max = torch::empty({E}, logits.options().dtype(at::kFloat));
  auto col_sum = torch::empty({E}, logits.options().dtype(at::kFloat));

  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  // 1) Sinkhorn Iterations
  const int warps_row = 4;
  dim3 blocks_row((T + warps_row - 1) / warps_row);
  dim3 blocks_row_threads(32, warps_row);

  const int cta_col = 256;
  dim3 blocks_col((T * E + cta_col - 1) / cta_col); // Grid size for col reduction
  if (blocks_col.x > 1024) blocks_col.x = 1024; // Cap grid size

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, s_logits.scalar_type(), "sinkhorn_loop", [&] {
      for(int i=0; i<sinkhorn_iters; ++i) {
          // A. Row Norm (LSE & Subtract)
          sinkhorn_row_norm_kernel<scalar_t><<<blocks_row, blocks_row_threads, 0, stream>>>(
              s_logits.data_ptr<scalar_t>(), T
          );
          
          // B. Col Stats: Reset
          // We can use a small kernel or cudaMemset/Fill.
          // Since we use atomics, we must init col_max to -inf and col_sum to 0
          AT_CUDA_CHECK(cudaMemsetAsync(col_sum.data_ptr<float>(), 0, E * sizeof(float), stream));
          // For -inf, we need a kernel or copy.
          // Quick hack: fill with huge negative or use a tiny init kernel.
          // Let's use a lambda or Thrust/Cub if available, but raw kernel is safer dependency-wise.
          // Creating a tiny init kernel inline or helper:
          // Just use `fill_` from pytorch
          col_max.fill_(-std::numeric_limits<float>::infinity());

          // C. Col Stats: Find Max
          sinkhorn_col_stats_kernel<scalar_t><<<blocks_col, cta_col, 2 * 64 * sizeof(float), stream>>>(
              s_logits.data_ptr<scalar_t>(),
              col_max.data_ptr<float>(),
              col_sum.data_ptr<float>(), // unused in this pass
              T
          );

          // D. Col Stats: Sum Exp
          sinkhorn_col_sum_kernel<scalar_t><<<blocks_col, cta_col, 2 * 64 * sizeof(float), stream>>>(
              s_logits.data_ptr<scalar_t>(),
              col_max.data_ptr<float>(),
              col_sum.data_ptr<float>(),
              T
          );

          // E. Col Update
          sinkhorn_col_update_kernel<scalar_t><<<blocks_col, cta_col, 64 * sizeof(float), stream>>>(
              s_logits.data_ptr<scalar_t>(),
              col_max.data_ptr<float>(),
              col_sum.data_ptr<float>(),
              T
          );
      }
      
      // 2) Top-2 Selection (on Normalized Logits)
      // Note: s_logits is now roughly doubly stochastic (rows sum=1, cols sum=T/E).
      // We pick the top 2 values.
      const int WARPS_SEL = 4;
      dim3 blocks_sel((T + WARPS_SEL - 1) / WARPS_SEL);
      top2_select_64_kernel<scalar_t, WARPS_SEL><<<blocks_sel, WARPS_SEL * 32, 0, stream>>>(
          s_logits.data_ptr<scalar_t>(),
          idx.data_ptr<int32_t>(),
          w.data_ptr<float>(),
          T
      );
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 3) Count (Same as before)
  const int cta = 256;
  dim3 blocks_cnt(std::min<int>((T + cta - 1) / cta, 1024));
  size_t shmem_cnt = E * sizeof(int32_t);
  count_expert_kernel<<<blocks_cnt, cta, shmem_cnt, stream>>>(
      idx.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), T);

  // 4) Scan (Same as before)
  {
    AT_CUDA_CHECK(cudaMemsetAsync(offsets.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
    void* d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, counts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>() + 1, E, stream);
    auto tmp = torch::empty({(long long)temp_bytes}, logits.options().dtype(at::kByte));
    d_temp = tmp.data_ptr();
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, counts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>() + 1, E, stream);
  }

  // 5) Read total packed
  int32_t total_packed = 0;
  AT_CUDA_CHECK(cudaMemcpyAsync(&total_packed, offsets.data_ptr<int32_t>() + E, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  // 6) Pack (Same as before)
  auto packed   = torch::empty({(int64_t)total_packed, d_model}, x.options());
  auto counters = torch::zeros({E}, counts.options());
  const int warps_pack = 4;
  dim3 blocks_pack((T + warps_pack - 1) / warps_pack);
  pack_copy_kernel<<<blocks_pack, warps_pack * 32, 0, stream>>>(
      x.data_ptr<at::Half>(), idx.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
      counters.data_ptr<int32_t>(), packed.data_ptr<at::Half>(), T, d_model, total_packed);

  return {idx, w, counts, offsets, packed};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_sinkhorn_forward", &fused_sinkhorn_forward_cuda,
        "Sinkhorn-Fused Top2 (E=64, x=half, logits=half/bf16)",
        torch::arg("logits"), torch::arg("x"), torch::arg("sinkhorn_iters") = 3);
}