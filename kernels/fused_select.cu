// fused_select.cu
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

using torch::Tensor;

namespace {

constexpr int E    = 64;  // experts
constexpr int K    = 2;   // top-k
constexpr int WARP = 32;

// -------------------- utils --------------------
template <typename T>
__device__ __forceinline__ float to_f32(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_f32<at::Half>(at::Half v) {
  // at::Half is compatible with __half in device code
  return __half2float(*reinterpret_cast<const __half*>(&v));
}

template <>
__device__ __forceinline__ float to_f32<at::BFloat16>(at::BFloat16 v) {
  return static_cast<float>(v);
}

__device__ __forceinline__ bool is_aligned_16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

// -------------------- Top-2 over 64 experts --------------------
// One warp processes one token row [64].
template <typename scalar_t, int WARPS_PER_CTA = 4>
__global__ void top2_select_64_kernel(
    const scalar_t* __restrict__ logits, // [T, 64]
    int32_t* __restrict__ out_idx,       // [T, 2]
    float*   __restrict__ out_w,         // [T, 2]
    int T)
{
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int t    = (int)(blockIdx.x * WARPS_PER_CTA + warp);
  if (t >= T) return;

  const scalar_t* row = logits + (int64_t)t * E;

  // lanes 0..31 read row[0..31] and row[32..63]
  float v0 = to_f32(row[lane]);      int i0 = lane;
  float v1 = to_f32(row[lane + 32]); int i1 = lane + 32;

  // Full-warp mask is safer for warp shuffles here.
  const unsigned mask = 0xffffffffu;

  // 1st argmax
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

  // 2nd argmax (mask out chosen imax)
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

  // Defensive clamp to [0, E-1]
  imax  = max(0, min(E - 1, imax));
  i2max = max(0, min(E - 1, i2max));

  // softmax over two selected logits
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

// -------------------- Pass 1: Count --------------------
__global__ void count_expert_kernel(
    const int32_t* __restrict__ expert_idx, // [T,2]
    int32_t* __restrict__ counts,           // [E]
    int T)
{
  extern __shared__ int32_t sh[];
  int32_t* hist = sh;

  for (int e = threadIdx.x; e < E; e += blockDim.x) hist[e] = 0;
  __syncthreads();

  for (int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
       t < T;
       t += (int)(gridDim.x * blockDim.x))
  {
    int32_t e0 = expert_idx[t*2 + 0];
    int32_t e1 = expert_idx[t*2 + 1];

    if ((uint32_t)e0 < (uint32_t)E) atomicAdd(&hist[e0], 1);
    if ((uint32_t)e1 < (uint32_t)E) atomicAdd(&hist[e1], 1);
  }

  __syncthreads();

  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    atomicAdd(&counts[e], hist[e]);
  }
}

// -------------------- Pass 2: Pack --------------------
// Each warp handles one token t. It writes x[t] twice into packed using per-expert slots.
// total_packed = packed.size(0) == 2*T (validated on host).
__global__ void pack_copy_kernel(
    const at::Half* __restrict__ x,            // [T, d_model]
    const int32_t*  __restrict__ expert_idx,   // [T, 2]
    const int32_t*  __restrict__ offsets,      // [E+1]
    int32_t*        __restrict__ counters,     // [E]
    at::Half*       __restrict__ out,          // [total_packed, d_model]
    int T, int d_model, int total_packed)
{
  const int warp_id = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int lane    = (int)(threadIdx.x & 31);
  if (warp_id >= T) return;

  const int32_t e0 = expert_idx[warp_id*2 + 0];
  const int32_t e1 = expert_idx[warp_id*2 + 1];
  if ((uint32_t)e0 >= (uint32_t)E || (uint32_t)e1 >= (uint32_t)E) return;

  const at::Half* src_h = x + (int64_t)warp_id * d_model;

  // Fast path requires 16B alignment and d_model%8==0 (8 half == 16B)
  const bool can_vec = (d_model % 8 == 0) && is_aligned_16(src_h);

  #pragma unroll 2
  for (int pick = 0; pick < 2; ++pick) {
    const int32_t e = (pick == 0) ? e0 : e1;

    const int32_t slot = atomicAdd(&counters[e], 1);
    const int32_t dst_token = offsets[e] + slot;

    // Hard bound check: never write outside packed
    if ((uint32_t)dst_token >= (uint32_t)total_packed) continue;

    at::Half* dst_h = out + (int64_t)dst_token * d_model;

    // If dst not 16B aligned, vector stores can still fault on some setups
    const bool vec_ok = can_vec && is_aligned_16(dst_h);

    if (vec_ok) {
      const uint4* src = reinterpret_cast<const uint4*>(src_h);
      uint4* dst       = reinterpret_cast<uint4*>(dst_h);
      const int vecs   = d_model / 8; // 8 half = 16B = 1 uint4
      for (int v = lane; v < vecs; v += WARP) {
        dst[v] = src[v];
      }
    } else {
      // Safe scalar copy (no alignment assumptions)
      for (int j = lane; j < d_model; j += WARP) {
        dst_h[j] = src_h[j];
      }
    }
  }
}

} // namespace

// -------------------- Frontend --------------------
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
fused_select_forward_cuda(Tensor logits, Tensor x) {
  TORCH_CHECK(logits.is_cuda() && x.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16 (Half)");
  TORCH_CHECK(logits.scalar_type() == at::kHalf || logits.scalar_type() == at::kBFloat16,
              "logits must be fp16/bf16");
  TORCH_CHECK(logits.dim() == 2 && logits.size(1) == E, "logits shape must be [T, 64]");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == logits.size(0), "x shape must be [T, d_model]");

  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(logits.stride(1) == 1, "logits must be row-major");
  TORCH_CHECK(x.stride(1) == 1, "x must be row-major");

  const int T = (int)logits.size(0);
  const int d_model = (int)x.size(1);

  // We'll do vector copy only when aligned; still require d_model multiple of 8 for that fast path.
  TORCH_CHECK(d_model > 0, "d_model must be > 0");

  auto idx     = torch::empty({T, K}, logits.options().dtype(at::kInt));
  auto w       = torch::empty({T, K}, logits.options().dtype(at::kFloat));
  auto counts  = torch::zeros({E},   logits.options().dtype(at::kInt));
  auto offsets = torch::empty({E + 1}, logits.options().dtype(at::kInt));

  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  // 1) Top-2 selection
  const int WARPS = 4;
  dim3 blocks_sel((T + WARPS - 1) / WARPS);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, logits.scalar_type(),
    "top2_select_64", [&] {
      top2_select_64_kernel<scalar_t, WARPS>
        <<<blocks_sel, WARPS * WARP, 0, stream>>>(
          logits.data_ptr<scalar_t>(),
          idx.data_ptr<int32_t>(),
          w.data_ptr<float>(),
          T);
    });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 2) Count
  const int cta = 256;
  dim3 blocks_cnt(std::min<int>((T + cta - 1) / cta, 1024));
  size_t shmem = E * sizeof(int32_t);

  count_expert_kernel<<<blocks_cnt, cta, shmem, stream>>>(
      idx.data_ptr<int32_t>(),
      counts.data_ptr<int32_t>(),
      T);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 3) Scan counts -> offsets[1..E], offsets[0]=0
  {
    AT_CUDA_CHECK(cudaMemsetAsync(offsets.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    cub::DeviceScan::InclusiveSum(
        d_temp, temp_bytes,
        counts.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>() + 1,
        E,
        stream);

    auto tmp = torch::empty({(long long)temp_bytes}, logits.options().dtype(at::kByte));
    d_temp = tmp.data_ptr();

    cub::DeviceScan::InclusiveSum(
        d_temp, temp_bytes,
        counts.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>() + 1,
        E,
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 4) Read offsets[E] to host to get total packed size
  int32_t total_packed = 0;
  AT_CUDA_CHECK(cudaMemcpyAsync(
      &total_packed,
      offsets.data_ptr<int32_t>() + E,
      sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  TORCH_CHECK(total_packed == 2 * T, "sum(counts) must be 2*T, got ", (int)total_packed);

  // 5) Pack
  auto packed   = torch::empty({(int64_t)total_packed, d_model}, x.options());
  auto counters = torch::zeros({E}, counts.options());

  const int warps_pack = 4;
  dim3 blocks_pack((T + warps_pack - 1) / warps_pack);

  pack_copy_kernel<<<blocks_pack, warps_pack * WARP, 0, stream>>>(
      x.data_ptr<at::Half>(),
      idx.data_ptr<int32_t>(),
      offsets.data_ptr<int32_t>(),
      counters.data_ptr<int32_t>(),
      packed.data_ptr<at::Half>(),
      T, d_model, total_packed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {idx, w, counts, offsets, packed};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_select_forward", &fused_select_forward_cuda,
        "Fused Top2-Select + Count/Scan + Pack (E=64, k=2, x=half, logits=half/bf16)");
}
