# Update Summary - Device Selection Feature

## What Was Added

### 1. Device Selection Parameter
All FaceFusion nodes now have a **device** parameter with 3 options:
- **auto**: Automatically detect and use best available device
- **cuda**: Force GPU inference (requires CUDA)
- **cpu**: Force CPU inference

### 2. Device Management System
New centralized device management module (`device_manager.py`):
- Auto-detects available devices
- Configures ONNX Runtime providers
- Manages GPU memory limits
- Provides device information and diagnostics

### 3. Enhanced Memory Management
Device-aware model initialization:
- Models adapt to selected device
- Proper GPU memory configuration
- Graceful fallback to CPU if CUDA unavailable

## Files Modified

1. **facefusion_api/device_manager.py** (NEW)
   - Central device management
   - Provider configuration
   - Memory limit settings

2. **facefusion_api/models/swapper.py**
   - Added device parameter to constructor
   - Device-aware initialization
   - Dynamic provider selection

3. **facefusion_api/swap_local.py**
   - Added device parameter to main function
   - Passes device through call chain

4. **facefusion_api/nodes/image_nodes.py**
   - Added device parameter to SwapFaceImage
   - Added device parameter to AdvancedSwapFaceImage
   - Updated all swap_face calls

## How It Works

### Auto Mode (Recommended)
```
1. Check if CUDAExecutionProvider available
2. If yes → use GPU (fast)
3. If no → use CPU (reliable)
4. Log selected device
```

### CUDA Mode
```
1. Check if CUDA available
2. If yes → initialize models on GPU
3. If no → fallback to CPU + warning
4. Configure GPU memory limits
```

### CPU Mode
```
1. Use CPUExecutionProvider only
2. No GPU memory management
3. Slower but always works
```

## Benefits

### For GPU Users
✅ Explicit device control
✅ Better error messages when CUDA unavailable
✅ Can force CPU if GPU memory constrained
✅ Can test performance differences

### For CPU Users
✅ No need to install onnxruntime-gpu
✅ Explicit CPU mode (no confusion)
✅ No unnecessary GPU checks
✅ Reliable performance

### For All Users
✅ Clear device selection in UI
✅ Graceful degradation
✅ Better diagnostics
✅ More predictable behavior

## Migration Guide

### Previous Behavior (Hardcoded)
```python
# Old: Always tried GPU
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)
```

### New Behavior (User Controlled)
```python
# New: Device selection
device = 'auto'  # User choice
providers = get_providers(device)  # Based on device
provider_options = get_provider_options(device)  # Memory limits
session = ort.InferenceSession(model_path, providers=providers, ...)
```

## Testing

### Test Device Selection
```bash
# 1. Run diagnostic
python3 diagnose_gpu.py

# 2. Test in ComfyUI
# - Set device: auto
# - Run face swap
# - Check console for "[LocalFaceSwapper] Using device: ..."

# 3. Try different devices
# - device: cuda (should be fast)
# - device: cpu (should be slower but work)
```

### Test Memory Management
```bash
# 1. Use GPU mode
device: cuda
unload_models: True

# 2. Run face swap multiple times
# 3. Check GPU memory with nvidia-smi
# 4. Should see memory released after each run
```

## Performance Impact

| Scenario | Device | Speed | Memory Usage |
|----------|--------|-------|--------------|
| GPU available | auto | Fast (GPU) | Managed |
| GPU unavailable | auto | Slow (CPU) | Minimal |
| Force GPU | cuda | Fast | Managed |
| Force CPU | cpu | Slow | Minimal |
| GPU fallback | cuda→cpu | Slow | Minimal |

## Compatibility

- ✅ Works with `onnxruntime` (CPU)
- ✅ Works with `onnxruntime-gpu` (GPU)
- ✅ Backwards compatible (no breaking changes)
- ✅ No additional dependencies required

## Future Enhancements

Planned features (not yet implemented):
- Multi-GPU selection (`cuda:0`, `cuda:1`)
- GPU memory limit configuration in UI
- Device performance benchmarking
- Automatic performance optimization

## Documentation

Created comprehensive guides:
- **DEVICE_SELECTION_GUIDE.md** - Full usage guide
- **DEVICE_QUICK_REFERENCE.md** - Quick reference
- **diagnose_gpu.py** - Diagnostic tool
- **fix_gpu_memory.sh** - GPU setup script
- **CRITICAL_FIX_REQUIRED.md** - onnxruntime-gpu installation

## Summary

This update solves the root cause of GPU memory issues by:
1. Giving users explicit device control
2. Making device selection transparent
3. Providing proper fallbacks
4. Improving error messages
5. Adding diagnostic tools

**Key Insight**: The 1.6GB GPU memory issue was because users had `onnxruntime` (CPU) installed, not `onnxruntime-gpu`. With device selection, users can now:
- See what device is being used
- Force CPU if GPU unavailable
- Get clear guidance on fixing CUDA issues
- Test different devices easily

**Recommendation**:
- Use `device: auto` for most users
- Use `device: cuda` + `unload_models: True` for GPU users
- Use `device: cpu` for systems without GPU
- Run `diagnose_gpu.py` if unsure about device availability