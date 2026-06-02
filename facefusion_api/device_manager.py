"""
Device management utilities for ONNX Runtime.
Provides centralized device selection and provider configuration.
"""
import os
from typing import List, Tuple, Optional


class DeviceManager:
    """Manages device selection and ONNX Runtime provider configuration."""

    _instance = None
    _current_device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available ONNX Runtime providers."""
        try:
            import onnxruntime as ort
            return ort.get_available_providers()
        except Exception:
            return ['CPUExecutionProvider']

    @classmethod
    def is_cuda_available(cls) -> bool:
        """Check if CUDA is available."""
        providers = cls.get_available_providers()
        return 'CUDAExecutionProvider' in providers

    @classmethod
    def set_device(cls, device: str) -> bool:
        """
        Set the device for inference.

        Args:
            device: 'auto', 'cuda', 'cpu', or 'cuda:0', 'cuda:1', etc.

        Returns:
            True if device is available and set successfully
        """
        device = device.lower().strip()

        if device == 'auto':
            # Auto-detect best available device
            if cls.is_cuda_available():
                cls._current_device = 'cuda'
                print(f"[DeviceManager] Auto-detected: CUDA available, using GPU")
                return True
            else:
                cls._current_device = 'cpu'
                print(f"[DeviceManager] Auto-detected: CUDA not available, using CPU")
                return True

        elif device.startswith('cuda'):
            # Check if CUDA is available
            if not cls.is_cuda_available():
                print(f"[DeviceManager] Warning: CUDA requested but not available, falling back to CPU")
                cls._current_device = 'cpu'
                return False

            # Extract device ID if specified (e.g., 'cuda:0')
            if ':' in device:
                device_id = int(device.split(':')[1])
                # Could add GPU ID validation here
                cls._current_device = f'cuda:{device_id}'
            else:
                cls._current_device = 'cuda'

            print(f"[DeviceManager] Device set to: {cls._current_device}")
            return True

        elif device == 'cpu':
            cls._current_device = 'cpu'
            print(f"[DeviceManager] Device set to: CPU")
            return True

        else:
            print(f"[DeviceManager] Warning: Unknown device '{device}', using CPU")
            cls._current_device = 'cpu'
            return False

    @classmethod
    def get_device(cls) -> str:
        """Get current device setting."""
        if cls._current_device is None:
            cls.set_device('auto')
        return cls._current_device

    @classmethod
    def get_providers(cls, device: Optional[str] = None) -> List[str]:
        """
        Get ONNX Runtime providers for the specified device.

        Args:
            device: Device to use, or None to use current device

        Returns:
            List of provider names in priority order
        """
        if device is None:
            device = cls.get_device()

        device = device.lower()

        if device.startswith('cuda'):
            if cls.is_cuda_available():
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print(f"[DeviceManager] Warning: CUDA not available, using CPU provider")
                return ['CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']

    @classmethod
    def get_provider_options(cls, device: Optional[str] = None, memory_limit_gb: Optional[float] = None) -> Tuple[List[dict], dict]:
        """
        Get ONNX Runtime provider options for the specified device.

        Args:
            device: Device to use, or None to use current device
            memory_limit_gb: GPU memory limit in GB (optional)

        Returns:
            Tuple of (provider_options_list, session_options)
        """
        if device is None:
            device = cls.get_device()

        device = device.lower()

        # Session options
        try:
            import onnxruntime as ort
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except Exception:
            session_options = None

        if device.startswith('cuda'):
            if cls.is_cuda_available():
                # Extract device ID
                device_id = 0
                if ':' in device:
                    try:
                        device_id = int(device.split(':')[1])
                    except:
                        device_id = 0

                # CUDA provider options
                cuda_options = {
                    'device_id': device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'do_copy_in_default_stream': True,
                }

                # Add memory limit if specified
                if memory_limit_gb is not None:
                    cuda_options['gpu_mem_limit'] = int(memory_limit_gb * 1024 * 1024 * 1024)

                # Try to add advanced options (may not be supported in all versions)
                try:
                    cuda_options['cudnn_conv_algo_search'] = 'EXHAUSTIVE'
                except:
                    pass

                # Return providers and options
                provider_options = [
                    cuda_options,  # CUDA options
                    {}             # CPU options (empty = default)
                ]

                return provider_options, session_options
            else:
                print(f"[DeviceManager] Warning: CUDA not available, using CPU provider")
                return [{}], session_options
        else:
            # CPU provider
            return [{}], session_options

    @classmethod
    def get_device_info(cls) -> str:
        """Get information about available devices."""
        info = []

        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                info.append(f"CUDA available: Yes")
                info.append(f"  - GPU Count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    info.append(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                info.append(f"CUDA available: No")
        except:
            info.append(f"PyTorch CUDA: Not available")

        # Check ONNX Runtime providers
        providers = cls.get_available_providers()
        info.append(f"ONNX Runtime providers: {providers}")

        # Current device
        current = cls.get_device()
        info.append(f"Current device: {current}")

        return '\n'.join(info)


# Convenience functions
def set_device(device: str) -> bool:
    """Set the device for inference."""
    return DeviceManager.set_device(device)


def get_device() -> str:
    """Get current device."""
    return DeviceManager.get_device()


def get_providers(device: Optional[str] = None) -> List[str]:
    """Get ONNX Runtime providers."""
    return DeviceManager.get_providers(device)


def get_provider_options(device: Optional[str] = None, memory_limit_gb: Optional[float] = None):
    """Get ONNX Runtime provider options."""
    return DeviceManager.get_provider_options(device, memory_limit_gb)


def print_device_info():
    """Print device information."""
    print(DeviceManager.get_device_info())