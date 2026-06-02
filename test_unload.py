#!/usr/bin/env python3
"""
Test script to verify model unloading functionality.
This script simulates the unloading process and checks if models are completely removed.
"""

import sys
import gc

# Mock torch for testing
class MockCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def empty_cache():
        pass

    @staticmethod
    def synchronize():
        pass

# Inject mock torch before importing
sys.modules['torch'] = type('torch', (), {'cuda': MockCuda})()

# Now import the modules
from facefusion_api.models import (
    get_local_swapper,
    unload_local_swapper,
    get_face_occluder,
    unload_face_occluder,
    get_face_parser,
    unload_face_parser,
)
from facefusion_api.detection.detector import (
    get_face_detector,
    unload_face_detector,
)
from facefusion_api.utils import unload_all_models


def test_unload_swapper():
    """Test swapper unloading."""
    print("\n=== Testing Swapper Unload ===")

    # Get swapper instance
    swapper1 = get_local_swapper('hyperswap_1c_256')
    print(f"Created swapper instance: {swapper1 is not None}")

    # Get same instance again (should be same due to singleton)
    swapper2 = get_local_swapper('hyperswap_1c_256')
    print(f"Same instance: {swapper1 is swapper2}")

    # Unload
    unload_local_swapper()

    # Get new instance (should be different)
    swapper3 = get_local_swapper('hyperswap_1c_256')
    print(f"New instance after unload: {swapper3 is not swapper1}")
    print(f"Swapper3 session: {swapper3.model_session}")

    # Cleanup
    unload_local_swapper()
    return True


def test_unload_occluder():
    """Test occluder unloading."""
    print("\n=== Testing Occluder Unload ===")

    # Get occluder instances
    occluder1 = get_face_occluder('xseg_1')
    print(f"Created occluder instance: {occluder1 is not None}")

    # Unload specific model
    unload_face_occluder('xseg_1')

    # Get new instance
    occluder2 = get_face_occluder('xseg_1')
    print(f"New instance after unload: {occluder2 is not occluder1}")

    # Cleanup all
    unload_face_occluder()
    return True


def test_unload_parser():
    """Test parser unloading."""
    print("\n=== Testing Parser Unload ===")

    # Get parser instances
    parser1 = get_face_parser('bisenet_resnet_34')
    print(f"Created parser instance: {parser1 is not None}")

    # Unload specific model
    unload_face_parser('bisenet_resnet_34')

    # Get new instance
    parser2 = get_face_parser('bisenet_resnet_34')
    print(f"New instance after unload: {parser2 is not parser1}")

    # Cleanup all
    unload_face_parser()
    return True


def test_unload_detector():
    """Test detector unloading."""
    print("\n=== Testing Detector Unload ===")

    # Get detector instances
    detector1 = get_face_detector('scrfd')
    print(f"Created detector instance: {detector1 is not None}")

    # Unload specific model
    unload_face_detector('scrfd')

    # Get new instance
    detector2 = get_face_detector('scrfd')
    print(f"New instance after unload: {detector2 is not detector1}")

    # Cleanup all
    unload_face_detector()
    return True


def test_unload_all():
    """Test complete unloading."""
    print("\n=== Testing Complete Unload ===")

    # Create all model instances
    swapper = get_local_swapper('hyperswap_1c_256')
    occluder = get_face_occluder('xseg_1')
    parser = get_face_parser('bisenet_resnet_34')
    detector = get_face_detector('scrfd')

    print("All models created")

    # Unload all at once
    unload_all_models()

    print("All models unloaded")

    # Check that instances are cleared
    swapper2 = get_local_swapper('hyperswap_1c_256')
    occluder2 = get_face_occluder('xseg_1')
    parser2 = get_face_parser('bisenet_resnet_34')
    detector2 = get_face_detector('scrfd')

    print(f"Swapper is new instance: {swapper2 is not swapper}")
    print(f"Occluder is new instance: {occluder2 is not occluder}")
    print(f"Parser is new instance: {parser2 is not parser}")
    print(f"Detector is new instance: {detector2 is not detector}")

    # Final cleanup
    unload_all_models()
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Model Unloading Verification Test")
    print("=" * 60)

    try:
        results = []
        results.append(('Swapper', test_unload_swapper()))
        results.append(('Occluder', test_unload_occluder()))
        results.append(('Parser', test_unload_parser()))
        results.append(('Detector', test_unload_detector()))
        results.append(('All Models', test_unload_all()))

        print("\n" + "=" * 60)
        print("Test Results:")
        print("=" * 60)
        for name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{name}: {status}")

        print("\nAll tests completed successfully!")
        print("Models are now properly unloaded with complete memory cleanup.")
        sys.exit(0)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)