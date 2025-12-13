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


