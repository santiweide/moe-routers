# Moe routers

## Files
kernels: cuda kernels for fused select strategy. Also grouped ffn and fused sinkhorn under construction.

models: different model structures defined with torch or cuda kernel-pybind api

third_party: cutlass. Currently only grouped gemm is used, and the cutlass fp16 type.

tools: pcie_batchwidth.py profiles the p2p bandwidth between 2 GPUs with PCIe connection.


## Overhead-Aware Routing Strategies (Experiment Harness)

This repo includes two scripts that map to the experiment design:

- `bench_router.py`: **single-process** router microbenchmark that reports a breakdown:
  - `T_logit` (router matmul), `T_select` (selection), `T_pack` (pack/permutation), `T_route`
  - Supports strategies:
    - `naive_topk`
    - `masked_matmul` (Strategy A, no permutation; only viable for small E)
    - `fused_select` (partial Strategy B: uses `fused_select_cuda.forward()` for top-k+softmax; pack still in Torch)
    - `sinkhorn` (Strategy C: algorithmic load balancing)

- `bench_2gpu_pcie.py`: **2-GPU EP (PCIe) benchmark** run via `torchrun`, measuring:
  - `ΔT_select` vs `naive_topk`
  - `T_comm` via `all_to_all_single`
  - load skew (`max(V)/mean(V)`, `CV`)
  - end-to-end layer latency with and without overlap + `eta_overlap`

### Single-GPU Router Breakdown (Table-style numbers)

Baseline (Naive Top-k):

```bash
python bench_router.py --device cuda --dtype fp16 --tokens 4096 --d_model 4096 --experts 64 --top_k 2 --strategy naive_topk
```

Strategy A (Masked Matmul, no permutation):

```bash
python bench_router.py --device cuda --dtype fp16 --tokens 4096 --d_model 4096 --experts 8 --top_k 2 --strategy masked_matmul
```

Strategy B (Fused selection, requires extension):

```bash
python setup_fused_select.py build_ext --inplace
python bench_router.py --device cuda --dtype fp16 --tokens 4096 --d_model 4096 --experts 64 --top_k 2 --strategy fused_select
```

Strategy C (Sinkhorn routing):

```bash
python bench_router.py --device cuda --dtype fp16 --tokens 4096 --d_model 4096 --experts 64 --top_k 2 --strategy sinkhorn --sinkhorn_iters 10 --sinkhorn_temperature 1.0
```

Synthetic skew (bias first N experts to simulate load imbalance):

```bash
python bench_router.py --device cuda --dtype fp16 --tokens 4096 --d_model 4096 --experts 64 --top_k 2 --strategy naive_topk --skew_experts 8 --skew_bias 2.0
```

### 2-GPU PCIe EP Benchmark (Communication + Load Balance + Overlap)

Run on a node with **2×A100 (PCIe)**:

Note: **Strategy A (`masked_matmul`) is single-GPU only** in this repo (no permutation / dense masking idea) and is **not implemented** in the 2-GPU EP benchmark. For 2-GPU, you can compare:
- Baseline: `naive_topk`
- Strategy B (partial): `fused_select` (fused top-k+softmax selection; pack/dispatch still in Torch)
- Strategy C: `sinkhorn` (balanced routing)

Naive Top-k baseline:

```bash
torchrun --nproc_per_node 2 bench_2gpu_pcie.py --dtype fp16 --tokens 4096 --d_model 4096 --experts 64 --top_k 2 --strategy naive_topk
```

Fused selection (partial Strategy B):

```bash
python setup_fused_select.py build_ext --inplace
torchrun --nproc_per_node 2 bench_2gpu_pcie.py --dtype fp16 --tokens 4096 --d_model 4096 --experts 64 --top_k 2 --strategy fused_select
```

Sinkhorn (Strategy C):

```bash
torchrun --nproc_per_node 2 bench_2gpu_pcie.py --dtype fp16 --tokens 4096 --d_model 4096 --experts 64 --top_k 2 --strategy sinkhorn --sinkhorn_iters 10 --sinkhorn_temperature 1.0
```

Run all supported 2-GPU strategies (prints a comparison table on rank0):

```bash
torchrun --nproc_per_node 2 bench_2gpu_pcie.py --dtype fp16 --tokens 4096 --d_model 4096 --experts 64 --top_k 2 --all_strategies
```

With synthetic skew:

```bash
torchrun --nproc_per_node 2 bench_2gpu_pcie.py --dtype fp16 --tokens 4096 --d_model 4096 --experts 64 --top_k 2 --strategy naive_topk --skew_experts 8 --skew_bias 2.0

torchrun --nproc_per_node 2 bench_2gpu_pcie.py --dtype fp16 --tokens 4096 --d_model 4096 --experts 64 --top_k 2 --all_strategies --skew_experts 8 --skew_bias 2.0
```


### Kernel Profile

Profile with Nsight Compute:
```
ncu --print-details \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    -k regex:fused \
    -c 1 \
    python bench_router.py --device cuda --dtype fp16 --tokens 4096 --d_model 4096 --experts 64 --top_k 2 --strategy fused_select
```


Kernel profile with torch profile and tensorboard:
```
 pip install tensorboard && tensorboard --logdir=./log/moe_trace

 ```





## Export (Netron-friendly ONNX)

Export as an ONNX pair:
- `__model__`: graph
- `__params__`: external weights blob

```bash
python run.py --device cpu --dtype fp32 --batch 2 --hidden 128 --export_onnx_dir /tmp/netron_model
```

Export the Transformer decoder model:

```bash
python run.py --model transformer_decoder --device cpu --dtype fp32 --batch 2 --seq_len 128 --vocab_size 32000 --d_model 512 --n_heads 8 --n_layers 6 --mlp_hidden_dim 2048 --export_onnx_dir /tmp/netron_transformer
```

Export the Decoder MoE model:

```bash
python run.py --model decoder_moe --device cpu --dtype fp32 --batch 2 --seq_len 128 --vocab_size 32000 --d_model 512 --n_heads 8 --n_layers 6 --mlp_hidden_dim 2048 --moe_n_experts 8 --moe_top_k 2 --export_onnx_dir /tmp/netron_moe
```

Open `__model__` in Netron.

