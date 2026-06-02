# Device Selection Feature - Usage Guide

## New Feature: Device Selection

You can now choose which device to use for inference: **auto**, **cuda (GPU)**, or **cpu**.

### Device Options

| Option | Description | Performance | Use Case |
|--------|-------------|-------------|----------|
| **auto** | Automatically detect best available | Fastest if CUDA available | Recommended for most users |
| **cuda** | Force GPU usage | Very fast (10-50x) | When you have NVIDIA GPU with CUDA |
| **cpu** | Force CPU usage | Slow but reliable | When GPU is not available or debugging |

### How to Use

In ComfyUI FaceFusion nodes:

1. **SwapFaceImage** node:
   - Find the `device` parameter
   - Select from dropdown: `auto`, `cuda`, or `cpu`

2. **AdvancedSwapFaceImage** node:
   - Find the `device` parameter (after `face_mask_padding`)
   - Select from dropdown: `auto`, `cuda`, or `cpu`

### Recommended Settings

#### For GPU Users (NVIDIA)
```
device: auto (or cuda)
unload_models: True (to save memory after each run)
```

**Benefits:**
- Fast inference (10-50x faster than CPU)
- GPU memory properly managed
- Automatic cleanup when unload_models=True

#### For CPU Users
```
device: cpu
unload_models: False (not needed, no GPU memory to free)
```

**Benefits:**
- Reliable inference without GPU
- Lower memory usage
- Works on any machine

### Troubleshooting

#### Issue: "device: cuda" but still slow

**Check if CUDA is available:**
```bash
python3 -c "import onnxruntime as ort; print('CUDA' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'CPU only')"
```

**If output is "CPU only":**
```bash
# Install GPU version
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

#### Issue: Out of memory with GPU

**Solution 1:** Use `unload_models: True`
```
device: cuda
unload_models: True  # Frees GPU memory after each run
```

**Solution 2:** Force CPU if GPU memory is insufficient
```
device: cpu
unload_models: False
```

### Performance Comparison

| Device | Relative Speed | GPU Memory | Best For |
|--------|---------------|------------|----------|
| **cuda** | 10-50x | 1-2GB | Production, batch processing |
| **cpu** | 1x (baseline) | None | Debugging, systems without GPU |
| **auto** | Same as cuda if available | Same as cuda | Recommended default |

### Technical Details

#### Auto Mode Behavior

1. Checks if `CUDAExecutionProvider` is available
2. If CUDA available → uses GPU
3. If CUDA not available → uses CPU
4. Logs the selected device

#### Memory Management

- **GPU mode**: Models loaded to GPU memory, can be unloaded
- **CPU mode**: Models in RAM, no unload needed
- **unload_models=True**: Only effective with GPU mode

### Advanced Usage

#### Multi-GPU Systems

To use a specific GPU (future feature):
```python
# Not yet implemented in UI, but supported internally
device: "cuda:0"  # Use first GPU
device: "cuda:1"  # Use second GPU
```

#### Device Switching

You can switch devices between runs:
1. Run 1: `device: cuda` (fast processing)
2. Run 2: `device: cpu` (if GPU memory needed elsewhere)
3. Models will automatically reinitialize with new device

### Diagnostic Tools

Run device diagnostics:
```bash
python3 diagnose_gpu.py
```

This will show:
- Available ONNX Runtime providers
- CUDA availability
- GPU information
- Current device setting

### Example Workflows

#### Workflow 1: Fast Production (GPU)
```
[Source Image] → [AdvancedSwapFaceImage] → [Preview]
                     device: auto
                     unload_models: True
```

#### Workflow 2: CPU-Only System
```
[Source Image] → [AdvancedSwapFaceImage] → [Preview]
                     device: cpu
                     unload_models: False
```

#### Workflow 3: Memory-Constrained GPU
```
[Source Image] → [AdvancedSwapFaceImage] → [Preview]
                     device: cuda
                     pixel_boost: 256x256  (reduced for less memory)
                     unload_models: True
```

### Migration from Previous Version

**Before (hardcoded CUDA):**
- Models always tried to use GPU
- No fallback to CPU if CUDA unavailable
- Users couldn't control device

**After (device selection):**
- Users choose device
- Graceful fallback to CPU
- Better error messages

### FAQ

**Q: Which device should I choose?**
A: Use `auto` - it automatically selects the best option.

**Q: Why is `device: cuda` slower than expected?**
A: You may have `onnxruntime` (CPU) instead of `onnxruntime-gpu`. Install the GPU version.

**Q: Should I enable `unload_models` with CPU?**
A: No, it has no effect on CPU mode. Only useful for GPU memory management.

**Q: Can I use multiple GPUs?**
A: Currently uses default GPU (cuda:0). Multi-GPU selection coming soon.

**Q: Does switching devices reload models?**
A: Yes, changing device requires reinitializing models on the new device.

### Related Documentation

- **CRITICAL_FIX_REQUIRED.md** - Installing onnxruntime-gpu
- **FIX_SUMMARY.md** - GPU memory management details
- **TESTING_GUIDE.md** - How to test device selection