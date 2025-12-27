# Minimal layernum=1 + Triton custom op (standalone)

This folder is **standalone**: it does **not** import anything from `vllm.*`.

## Files

- `models/`: model definitions
- `models/model.py`: 1-layer model definition (`Linear + residual + custom op`)
- `models/transformer_decoder_model.py`: decoder-only Transformer
- `models/decoder_moe.py`: decoder-only Transformer with MoE MLP
- `run.py`: runnable entrypoint / benchmark for the model
- `triton_compat.py`: optional Triton import shim (falls back to Torch when Triton is absent)
- `export_netron.py`: export the model to a Netron-friendly ONNX pair
- `kernels/`: C++/CUDA kernels (e.g. MoE router extension sources)
- `requirements.txt`: minimal dependencies

## Run

From repo root:

```bash
python run.py --device cpu --dtype fp32 --batch 2 --hidden 128
```

If you have CUDA + Triton:

```bash
python run.py --device cuda --dtype fp16 --batch 8 --hidden 4096
```

Run the Transformer decoder demo:

```bash
python run.py --model transformer_decoder --device cuda --dtype fp16 --batch 2 --seq_len 128 --vocab_size 32000 --d_model 512 --n_heads 8 --n_layers 6 --mlp_hidden_dim 2048
```

Run the Decoder MoE demo (MLP replaced by MoE):

```bash
python run.py --model decoder_moe --device cuda --dtype fp16 --batch 2 --seq_len 128 --vocab_size 32000 --d_model 512 --n_heads 8 --n_layers 6 --mlp_hidden_dim 2048 --moe_n_experts 8 --moe_top_k 2
```

Compare router metrics (torch vs cuda_ext):

```bash
python run.py --model decoder_moe --moe_router torch --moe_metrics --device cuda --dtype fp16 --batch 2 --seq_len 128 --vocab_size 32000 --d_model 512 --n_heads 8 --n_layers 6 --mlp_hidden_dim 2048 --moe_n_experts 8 --moe_top_k 2
python run.py --model decoder_moe --moe_router cuda_ext --moe_metrics --device cuda --dtype fp16 --batch 2 --seq_len 128 --vocab_size 32000 --d_model 512 --n_heads 8 --n_layers 6 --mlp_hidden_dim 2048 --moe_n_experts 8 --moe_top_k 2
```

Note: for per-token latency style benchmarking, use `--batch 1` (and often `--seq_len 1`) to isolate token-level overhead.

Compare with Sinkhorn (balanced) routing:

```bash
python run.py --model decoder_moe --moe_router sinkhorn --sinkhorn_iters 10 --sinkhorn_temperature 1.0 --moe_metrics --device cuda --dtype fp16 --batch 2 --seq_len 128 --vocab_size 32000 --d_model 512 --n_heads 8 --n_layers 6 --mlp_hidden_dim 2048 --moe_n_experts 8 --moe_top_k 2
```

Build and use the CUDA router extension (A100-friendly):

```bash
python setup_router_ext.py build_ext --inplace
python run.py --model decoder_moe --moe_router cuda_ext --device cuda --dtype fp16 --batch 2 --seq_len 128 --vocab_size 32000 --d_model 512 --n_heads 8 --n_layers 6 --mlp_hidden_dim 2048 --moe_n_experts 8 --moe_top_k 2
```

### Troubleshooting: GCC too old (PyTorch extensions require GCC/G++ 9+)

If you see:
`You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later.`

You need to compile the extension with a newer host compiler. On clusters this usually means
loading a newer GCC module and setting:

```bash
module load gcc/11.2.0
python setup_router_ext.py build_ext --inplace
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

## Notes

- If Triton is not installed (or you're on CPU), it automatically uses the Torch fallback implementation.
- This demo is inference-only (no backward).

## Overhead-Aware Routing Strategies (Experiment Harness)

This repo includes two scripts that map to the experiment design:

- `bench_router.py`: **single-process** router microbenchmark that reports a breakdown:
  - `T_logit` (router matmul), `T_select` (selection), `T_pack` (pack/permutation), `T_route`
  - Supports strategies:
    - `naive_topk`
    - `masked_matmul` (Strategy A, no permutation; only viable for small E)
    - `fused_select` (partial Strategy B: uses `router_ext_cuda.forward()` for top-k+softmax; pack still in Torch)
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
python setup_router_ext.py build_ext --inplace
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
python setup_router_ext.py build_ext --inplace
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
```




Memory latency
```
python bench_router.py \
  --perm_sweep --device cuda --dtype fp16 \
  --d_model 4096 --experts 64 --top_k 2 --seq_len 128 \
  --batch_sizes 1 2 4 8 16 32 64 \
  --naive_impl atomic_triton \
  --peak_bw_gbs 1555 \
  --warmup 5 --iters 20 \
  --csv_out perm_efficiency.csv \
  --plot_out perm_efficiency.png
```
