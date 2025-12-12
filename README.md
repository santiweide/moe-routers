# Minimal layernum=1 + Triton custom op (standalone)

This folder is **standalone**: it does **not** import anything from `vllm.*`.

## Files

- `run.py`: 1-layer PyTorch model (`Linear + residual + custom op`)
- `triton_compat.py`: optional Triton import shim (falls back to Torch when Triton is absent)
- `requirements.txt`: minimal dependencies

## Run

From repo root:

```bash
python standalone/minimal_layernum1_triton/run.py --device cpu --dtype fp32 --batch 2 --hidden 128
```

If you have CUDA + Triton:

```bash
python standalone/minimal_layernum1_triton/run.py --device cuda --dtype fp16 --batch 8 --hidden 4096
```

## Notes

- If Triton is not installed (or you're on CPU), it automatically uses the Torch fallback implementation.
- This demo is inference-only (no backward).


