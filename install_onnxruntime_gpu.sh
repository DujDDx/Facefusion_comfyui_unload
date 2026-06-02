#!/bin/bash
# Fix GPU Memory Issue - Install onnxruntime-gpu

echo "==================================================================="
echo "GPU Memory Fix - Replacing onnxruntime with onnxruntime-gpu"
echo "==================================================================="

echo ""
echo "Current situation:"
echo "- onnxruntime (CPU) is installed: models run on CPU"
echo "- PyTorch uses GPU for other operations"
echo "- GPU memory not properly managed"
echo ""

echo "Solution:"
echo "1. Remove onnxruntime (CPU version)"
echo "2. Install onnxruntime-gpu (GPU version)"
echo "3. This will enable CUDAExecutionProvider"
echo "4. GPU memory will be properly managed and released"
echo ""

read -p "Continue with fix? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Step 1: Removing onnxruntime (CPU)..."
pip uninstall -y onnxruntime

echo ""
echo "Step 2: Installing onnxruntime-gpu..."
pip install onnxruntime-gpu

echo ""
echo "Step 3: Verifying installation..."
python3 -c "import onnxruntime as ort; print('Version:', ort.__version__); print('Providers:', ort.get_available_providers())"

echo ""
echo "==================================================================="
echo "Fix completed!"
echo "==================================================================="
echo ""
echo "Next steps:"
echo "1. Restart ComfyUI"
echo "2. Test face swap with unload_models=True"
echo "3. Check GPU memory with: nvidia-smi"
echo ""
echo "Expected result: GPU memory should drop to <100MB after unload"
echo ""