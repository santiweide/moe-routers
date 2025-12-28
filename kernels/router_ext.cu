namespace {

constexpr int MAX_K = 8;

template <typename T> struct Pack2T { using type = T; static constexpr int width = 1; };
template <> struct Pack2T<float> { using type = float2; static constexpr int width = 2; };
#if defined(__CUDA_ARCH__)
template <> struct Pack2T<at::Half> { using type = __half2; static constexpr int width = 2; };
template <> struct Pack2T<at::BFloat16> { using type = __nv_bfloat162; static constexpr int width = 2; };
#endif

template <typename T>
__device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }

template <>
__device__ __forceinline__ float to_float<at::Half>(at::Half v) {
  return __half2float(*reinterpret_cast<const __half*>(&v));
}

template <>
__device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 v) {
  return static_cast<float>(v);
}

template <int K>
__device__ __forceinline__ void init_topk(float* v, int* i) {
  #pragma unroll
  for (int j = 0; j < K; ++j) { v[j] = -CUDART_INF_F; i[j] = -1; }
}

template <int K>
__device__ __forceinline__ void insert_topk_k(float val, int idx, float* topv, int* topi) {
  int pos = K;
  #pragma unroll
  for (int j = 0; j < K; ++j) { if (val > topv[j]) { pos = j; break; } }
  if (pos == K) return;
  for (int j = K - 1; j > pos; --j) { topv[j] = topv[j - 1]; topi[j] = topi[j - 1]; }
  topv[pos] = val; topi[pos] = idx;
}

template <int K>
__device__ __forceinline__ void merge_two_lists(float* va, int* ia,
                                                const float* vb, const int* ib) {
  float tv[2 * K]; int ti[2 * K];
  #pragma unroll
  for (int j = 0; j < K; ++j) { tv[j] = va[j]; ti[j] = ia[j]; tv[K + j] = vb[j]; ti[K + j] = ib[j]; }
  #pragma unroll
  for (int m = 0; m < K; ++m) {
    int best = m;
    #pragma unroll
    for (int j = m + 1; j < 2 * K; ++j) { if (tv[j] > tv[best]) best = j; }
    float fv = tv[m]; int fi = ti[m];
    tv[m] = tv[best]; ti[m] = ti[best]; tv[best] = fv; ti[best] = fi;
  }
  #pragma unroll
  for (int j = 0; j < K; ++j) { va[j] = tv[j]; ia[j] = ti[j]; }
}

template <typename scalar_t>
__device__ __forceinline__ void load_pair_accumulate(const scalar_t* base, int e0, int emax,
                                                     float* local_v, int* local_i) {
  using P2 = typename Pack2T<scalar_t>::type;
  constexpr int W = Pack2T<scalar_t>::width;
  if constexpr (W == 2) {
    const int e = e0 * 2;
    if (e + 1 < emax) {
      // 对齐：若未对齐，GPU 也能正确访问，但可能降速；实际训练中 logits 常 128/256 对齐。
      const P2 v2 = *reinterpret_cast<const P2*>(base + e);
      float v0, v1;
      if constexpr (std::is_same<scalar_t,float>::value) {
        const float2& f2 = reinterpret_cast<const float2&>(v2);
        v0 = f2.x; v1 = f2.y;
      } else if constexpr (std::is_same<scalar_t,at::Half>::value) {
        const __half2& h2 = reinterpret_cast<const __half2&>(v2);
        v0 = __half2float(__low2half(h2)); v1 = __half2float(__high2half(h2));
      } else { // bfloat16
        const __nv_bfloat162& b2 = reinterpret_cast<const __nv_bfloat162&>(v2);
        v0 = __bfloat162float(__low2bfloat16(b2)); v1 = __bfloat162float(__high2bfloat16(b2));
      }
      insert_topk_k<MAX_K>(v0, e,   local_v, local_i);
      insert_topk_k<MAX_K>(v1, e+1, local_v, local_i);
    } else if (e < emax) {
      float v = to_float<scalar_t>(base[e]);
      insert_topk_k<MAX_K>(v, e, local_v, local_i);
    }
  } else {
    const int e = e0;
    if (e < emax) {
      float v = to_float<scalar_t>(base[e]);
      insert_topk_k<MAX_K>(v, e, local_v, local_i);
    }
  }
}

template <typename scalar_t, int K, int BLOCK_THREADS>
__global__ void topk_router_kernel_warp_reduce(
    const scalar_t* __restrict__ logits,
    int32_t* __restrict__ out_idx,
    float* __restrict__ out_w,
    int tokens,
    int experts) {

  static_assert(K <= MAX_K, "K exceeds MAX_K");
  const int t = blockIdx.x;
  if (t >= tokens) return;

  float local_v[K]; int local_i[K];
  init_topk<K>(local_v, local_i);

  const scalar_t* row = logits + static_cast<int64_t>(t) * experts;

  // 线程块内分块扫描；half/bf16/float 根据类型选择 2 或 1 元矢量化
  constexpr int W = Pack2T<scalar_t>::width;
  const int stride = BLOCK_THREADS;
  if constexpr (W == 2) {
    for (int e2 = threadIdx.x; e2 * 2 < experts; e2 += stride)
      load_pair_accumulate(row, e2, experts, local_v, local_i);
  } else {
    for (int e = threadIdx.x; e < experts; e += stride)
      load_pair_accumulate(row, e, experts, local_v, local_i);
  }

  const unsigned mask = __activemask();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  // warp 内 XOR-蝶形合并
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float ov[K]; int oi[K];
    #pragma unroll
    for (int j = 0; j < K; ++j) {
      ov[j] = __shfl_xor_sync(mask, local_v[j], offset);
      oi[j] = __shfl_xor_sync(mask, local_i[j], offset);
    }
    merge_two_lists<K>(local_v, local_i, ov, oi);
  }

  // 将每个 warp 的优胜 K 写到共享内存
  constexpr int NUM_WARPS = BLOCK_THREADS / 32;
  __shared__ float sh_v[NUM_WARPS * MAX_K];
  __shared__ int   sh_i[NUM_WARPS * MAX_K];
  if (lane == 0) {
    #pragma unroll
    for (int j = 0; j < K; ++j) {
      sh_v[warp * MAX_K + j] = local_v[j];
      sh_i[warp * MAX_K + j] = local_i[j];
    }
  }
  __syncthreads();

  // 由 warp0 再做一次 XOR 归约（跨 warp）
  if (warp == 0) {
    float v[K]; int i[K];
    init_topk<K>(v, i);

    if (lane < NUM_WARPS) {
      #pragma unroll
      for (int j = 0; j < K; ++j) {
        v[j] = sh_v[lane * MAX_K + j];
        i[j] = sh_i[lane * MAX_K + j];
      }
    }

    const unsigned m0 = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float ov[K]; int oi[K];
      #pragma unroll
      for (int j = 0; j < K; ++j) {
        ov[j] = __shfl_xor_sync(m0, v[j], offset);
        oi[j] = __shfl_xor_sync(m0, i[j], offset);
      }
      merge_two_lists<K>(v, i, ov, oi);
    }

    if (lane == 0) {
      // 仅对 top-K 做稳定 Softmax
      float m = v[0];
      #pragma unroll
      for (int j = 1; j < K; ++j) m = fmaxf(m, v[j]);
      float denom = 0.f, exps[K];
      #pragma unroll
      for (int j = 0; j < K; ++j) { exps[j] = __expf(v[j] - m); denom += exps[j]; }
      denom = fmaxf(denom, 1e-9f);

      int32_t* idx_row = out_idx + static_cast<int64_t>(t) * K;
      float*   w_row   = out_w   + static_cast<int64_t>(t) * K;
      #pragma unroll
      for (int j = 0; j < K; ++j) { idx_row[j] = i[j]; w_row[j] = exps[j] / denom; }
    }
  }
}

template <typename scalar_t, int K, int BLOCK_THREADS>
__global__ void topk_router_kernel_cub_sort(
    const scalar_t* __restrict__ logits,
    int32_t* __restrict__ out_idx,
    float* __restrict__ out_w,
    int tokens,
    int experts) {

  static_assert(K <= MAX_K, "K exceeds MAX_K");
  const int t = blockIdx.x;
  if (t >= tokens) return;

  const int e = threadIdx.x;
  float key = -CUDART_INF_F;
  int   val = e;

  if (e < experts) {
    const scalar_t* row = logits + static_cast<int64_t>(t) * experts;
    key = to_float<scalar_t>(row[e]);
  }

  using BlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, 1, int>;
  __shared__ typename BlockSort::TempStorage temp_storage;
  BlockSort(temp_storage).SortPairsDescending(key, val);

  __shared__ float topv[MAX_K];
  __shared__ int   topi[MAX_K];
  if (threadIdx.x < K) { topv[threadIdx.x] = key; topi[threadIdx.x] = val; }
  __syncthreads();

  if (threadIdx.x == 0) {
    float m = topv[0]; #pragma unroll
    for (int j = 1; j < K; ++j) m = fmaxf(m, topv[j]);
    float denom = 0.f, exps[K];
    #pragma unroll
    for (int j = 0; j < K; ++j) { exps[j] = __expf(topv[j] - m); denom += exps[j]; }
    denom = fmaxf(denom, 1e-9f);

    int32_t* idx_row = out_idx + static_cast<int64_t>(t) * K;
    float*   w_row   = out_w   + static_cast<int64_t>(t) * K;
    #pragma unroll
    for (int j = 0; j < K; ++j) { idx_row[j] = topi[j]; w_row[j] = exps[j] / denom; }
  }
}

} // namespace
