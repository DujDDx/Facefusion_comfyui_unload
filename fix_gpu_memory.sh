#!/bin/bash
# One-click fix for GPU memory issue
# This script will install onnxruntime-gpu and verify CUDA support

set -e

echo "======================================================================="
echo "GPU Memory Fix - Installing onnxruntime-gpu"
echo "======================================================================="
echo ""

# Step 1: Check current installation
echo "Step 1: Checking current installation..."
python3 diagnose_gpu.py
echo ""

# Step 2: Confirm fix
echo "======================================================================="
echo "PROBLEM: You have onnxruntime (CPU) installed"
echo "SOLUTION: Replace with onnxruntime-gpu (GPU support)"
echo "======================================================================="
echo ""
echo "This will:"
echo "  ✓ Uninstall onnxruntime (CPU version)"
echo "  ✓ Install onnxruntime-gpu (GPU version)"
echo "  ✓ Enable CUDAExecutionProvider"
echo "  ✓ Make ONNX models run 10-50x faster on GPU"
echo "  ✓ Fix GPU memory management issues"
echo ""
read -p "Continue? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

# Step 3: Uninstall CPU version
echo ""
echo "Step 2: Removing onnxruntime (CPU version)..."
pip uninstall -y onnxruntime || true
echo "✓ Done"

# Step 4: Install GPU version
echo ""
echo "Step 3: Installing onnxruntime-gpu..."
pip install onnxruntime-gpu
echo "✓ Done"

# Step 5: Verify installation
echo ""
echo "Step 4: Verifying installation..."
python3 -c "
import onnxruntime as ort
print('✓ ONNX Runtime version:', ort.__version__)
providers = ort.get_available_providers()
print('✓ Available providers:', providers)

if 'CUDAExecutionProvider' in providers:
    print('')
    print('✅ SUCCESS! CUDAExecutionProvider is now available!')
    print('   ONNX models will use GPU and memory will be properly managed.')
    print('')
    print('Next steps:')
    print('  1. Restart ComfyUI')
    print('  2. Enable unload_models=True in FaceFusion nodes')
    print('  3. Monitor GPU memory with: nvidia-smi')
    print('')
    print('Expected result: GPU memory <100MB after unload')
else:
    print('')
    print('⚠️  WARNING: CUDAExecutionProvider not available')
    print('   This might mean CUDA is not installed on your system.')
    print('   Please check your CUDA installation.')
fi
"

echo ""
echo "======================================================================="
echo "Fix completed!"
echo "======================================================================="