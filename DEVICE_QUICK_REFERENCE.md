# Device Selection Quick Reference

##  Usage

In FaceFusion nodes, select device:
- **auto** - Detect best (recommended)
- **cuda** - Force GPU
- **cpu** - Force CPU

## Settings

### GPU Users
```
device: auto (or cuda)
unload_models: True
```
→ Fast inference, memory freed after use

### CPU Users
```
device: cpu
unload_models: False
```
→ Reliable, works anywhere

## Troubleshooting

### CUDA not working?
```bash
# Check CUDA availability
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

# If missing CUDAExecutionProvider:
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

### Out of GPU memory?
- Enable `unload_models: True`
- Reduce `pixel_boost` (256x256 instead of 512x512)
- Or use `device: cpu`

## Performance

| Device | Speed | Memory |
|--------|-------|---------|
| cuda | 10-50x | 1-2GB |
| cpu | baseline | none |
| auto | best available | varies |

## Diagnostic

```bash
python3 diagnose_gpu.py
```

See DEVICE_SELECTION_GUIDE.md for detailed info.