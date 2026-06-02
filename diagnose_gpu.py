#!/usr/bin/env python3
"""
Auto-detect and diagnose GPU memory issue.
Checks if onnxruntime-gpu is installed and CUDA is available.
"""

import sys
import subprocess

def check_onnxruntime():
    """Check ONNX Runtime installation and providers."""
    print("=" * 70)
    print("ONNX Runtime Check")
    print("=" * 70)

    try:
        import onnxruntime as ort
        print(f"✓ ONNX Runtime version: {ort.__version__}")

        providers = ort.get_available_providers()
        print(f"✓ Available providers: {providers}")

        if 'CUDAExecutionProvider' in providers:
            print("✓ CUDAExecutionProvider: AVAILABLE")
            print("  → ONNX models will use GPU")
            return True, "gpu"
        else:
            print("✗ CUDAExecutionProvider: NOT AVAILABLE")
            print("  → ONNX models will use CPU (SLOW)")
            return True, "cpu"
    except ImportError:
        print("✗ ONNX Runtime: NOT INSTALLED")
        return False, None

def check_pytorch_cuda():
    """Check PyTorch CUDA availability."""
    print("\n" + "=" * 70)
    print("PyTorch CUDA Check")
    print("=" * 70)

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

            # Check memory usage
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"✓ GPU memory allocated: {allocated:.2f} MB")
            print(f"✓ GPU memory reserved: {reserved:.2f} MB")

        return torch.cuda.is_available()
    except ImportError:
        print("✗ PyTorch: NOT INSTALLED")
        return False

def check_nvidia_smi():
    """Check nvidia-smi output."""
    print("\n" + "=" * 70)
    print("NVIDIA SMI Check")
    print("=" * 70)

    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id, name, mem_used, mem_total = parts
                    print(f"✓ GPU {gpu_id}: {name}")
                    print(f"  Memory: {mem_used} MB / {mem_total} MB")
            return True
        else:
            print("✗ nvidia-smi failed")
            return False
    except Exception as e:
        print(f"✗ nvidia-smi not available: {e}")
        return False

def diagnose():
    """Run all diagnostics."""
    print("\n" + "=" * 70)
    print("GPU Memory Diagnostic Tool")
    print("=" * 70)
    print()

    # Check ONNX Runtime
    ort_installed, ort_mode = check_onnxruntime()

    # Check PyTorch
    pytorch_cuda = check_pytorch_cuda()

    # Check NVIDIA
    nvidia_ok = check_nvidia_smi()

    # Diagnosis
    print("\n" + "=" * 70)
    print("Diagnosis")
    print("=" * 70)

    if not ort_installed:
        print("❌ CRITICAL: ONNX Runtime is not installed!")
        print("   Solution: pip install onnxruntime-gpu")
        return

    if ort_mode == "cpu":
        print("❌ PROBLEM IDENTIFIED:")
        print("   - You have 'onnxruntime' (CPU version) installed")
        print("   - ONNX models run on CPU (very slow)")
        print("   - GPU memory is from PyTorch, not ONNX Runtime")
        print("   - GPU memory cannot be released by our unload methods")
        print()
        print("✅ SOLUTION:")
        print("   1. pip uninstall -y onnxruntime")
        print("   2. pip install onnxruntime-gpu")
        print("   3. Restart ComfyUI")
        print()
        print("   Expected result:")
        print("   - Models run on GPU (10-50x faster)")
        print("   - GPU memory properly managed")
        print("   - After unload: <100MB GPU usage")

    elif ort_mode == "gpu":
        print("✓ ONNX Runtime is using GPU!")
        print("✓ Our memory management fixes will work correctly")
        print()
        print("If GPU memory is still high after unload:")
        print("  1. Check other ComfyUI nodes")
        print("  2. Restart ComfyUI completely")
        print("  3. Check for other Python processes: nvidia-smi")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    try:
        diagnose()
    except Exception as e:
        print(f"\n❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)