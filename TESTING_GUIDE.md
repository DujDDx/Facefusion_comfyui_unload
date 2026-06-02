# GPU Memory Testing Guide

## Quick Test Steps

### 1. Baseline Check
Before running any face swap:
```bash
nvidia-smi
```
Note the memory usage of the python process.

### 2. Run Face Swap with unload_models=True
In ComfyUI:
- Add a FaceFusion node (AdvancedSwapFaceImage)
- Set `unload_models = True`
- Process some images

### 3. Check Memory After Unload
```bash
nvidia-smi
```

Expected behavior:
```
BEFORE FIX:
|    0   N/A  N/A     58319      C   python     1990MiB |  (stuck)

AFTER FIX:
|    0   N/A  N/A     58319      C   python       45MiB |  (released)
```

## Detailed Testing

### Test Script
Create a simple test workflow:

```python
# In ComfyUI Python console or test script
import torch

# Before loading models
print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

# Load and run face swap (via ComfyUI node)
# ... your workflow here ...

# After unload_models=True
print(f"Final GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
```

### Expected Console Output

When unload_models is enabled, you should see:

```
[LocalFaceSwapper] Running on: CUDAExecutionProvider
[LocalFaceSwapper] Unloading model hyperswap_1c_256 from providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
[LocalFaceSwapper] Unloaded model: hyperswap_1c_256
[LocalFaceSwapper] CUDA cache cleared
[FaceOccluder] Unloading model xseg_1 from providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
[FaceOccluder] Unloaded model: xseg_1
[FaceParser] Unloading model bisenet_resnet_34 from providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
[FaceParser] Unloaded model: bisenet_resnet_34
[FaceDetector] Unloading detector scrfd_2.5g from providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
[FaceDetector] Unloaded detector model: scrfd_2.5g
[FaceDetector] Unloading recognition arcface_w600k_r50 from providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
[FaceDetector] Unloaded recognition model: arcface_w600k_r50
[Unload] Garbage collection round 1 completed
[Unload] Garbage collection round 2 completed
[Unload] Garbage collection round 3 completed
[Unload] CUDA cache cleared and synchronized
[Unload] CUDA memory still allocated: 45.23 MB
[Unload] All models completely unloaded, memory freed
```

## Troubleshooting

### Issue: Memory Still High (>500MB)

**Possible causes:**

1. **Other ComfyUI nodes using GPU**
   - Check for other custom nodes that might be holding memory
   - Solution: Disable other GPU nodes temporarily

2. **Multiple ComfyUI instances**
   - Each instance holds its own memory
   - Solution: Close other ComfyUI instances

3. **CUDA provider not used**
   - Check console shows "Running on: CUDAExecutionProvider"
   - If shows "CPUExecutionProvider", check CUDA installation

4. **Old ONNX Runtime version**
   - Some versions have memory leaks
   - Solution: Update onnxruntime-gpu:
   ```bash
   pip install --upgrade onnxruntime-gpu
   ```

### Issue: Out of Memory (OOM) During Swap

**Solution:**
Increase memory limit in the code:

```python
# In swapper.py, change:
'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB instead of 2GB
```

### Issue: Slow Performance After Fix

**Cause:** Memory limits might be too restrictive

**Solution:**
1. Increase memory limits for your GPU
2. Or remove memory limits if you have plenty of VRAM:

```python
# Remove this line if you have 12GB+ VRAM:
# 'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
```

## Performance Comparison

### Memory Usage
| Scenario | Before Fix | After Fix |
|----------|------------|-----------|
| Models loaded | ~2000 MiB | ~2000 MiB |
| After unload_models=False | ~2000 MiB | ~2000 MiB |
| After unload_models=True | ~1990 MiB | **~45 MiB** |

### Speed Impact
- **No significant speed impact** from memory limits
- Memory limits only affect allocation strategy, not inference speed
- Cleanup takes ~1-2 seconds (acceptable overhead)

## Advanced Diagnostics

### Check ONNX Runtime Version
```bash
python -c "import onnxruntime as ort; print(ort.__version__)"
```

Recommended: >= 1.16.0 for better memory management

### Check CUDA Provider Availability
```python
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
# Should include 'CUDAExecutionProvider'
```

### Monitor Memory During Execution
```python
import torch

def log_memory():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

# Call before and after critical operations
log_memory()
```

## Success Criteria

✅ **Fix is working if:**
1. After unload_models=True, GPU memory drops to <100MB
2. Console shows all unload messages
3. No memory leaks over multiple runs
4. Speed remains acceptable

❌ **Fix needs adjustment if:**
1. Memory stays >500MB after unload
2. Getting OOM errors during normal operation
3. Speed significantly degraded
4. Console shows CPUExecutionProvider instead of CUDA

## Next Steps After Testing

If fix is successful:
- Use unload_models=True for production
- Monitor memory usage periodically
- Adjust memory limits based on your GPU size

If issues persist:
- Check for other memory-holding nodes
- Update ONNX Runtime version
- Report specific error messages for further debugging