# Minimal layernum=1 + Triton custom op (standalone)

This folder is **standalone**: it does **not** import anything from `vllm.*`.

## Files

- `model.py`: model definition (`Linear + residual + custom op`)
- `run.py`: runnable entrypoint / benchmark for the model
- `triton_compat.py`: optional Triton import shim (falls back to Torch when Triton is absent)
- `export_netron.py`: export the model to a Netron-friendly ONNX pair
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

## Export (Netron-friendly ONNX)

Export as an ONNX pair:
- `__model__`: graph
- `__params__`: external weights blob

```bash
python run.py --device cpu --dtype fp32 --batch 2 --hidden 128 --export_onnx_dir /tmp/netron_model
```

Open `__model__` in Netron.

## Notes

- If Triton is not installed (or you're on CPU), it automatically uses the Torch fallback implementation.
- This demo is inference-only (no backward).


