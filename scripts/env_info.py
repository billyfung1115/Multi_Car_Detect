#!/usr/bin/env python3
"""Print environment info useful for debugging/install checks.

Outputs: Python version, torch/torchvision/torchaudio versions, CUDA availability/version,
         ultralytics version, OpenCV version, and the python executable path.
"""
import sys
import platform


def _print(msg):
    print(msg)


def main():
    _print("=== Environment Info ===")
    _print(f"Python: {platform.python_version()}")
    _print(f"Executable: {sys.executable}")

    # torch + cuda
    try:
        import torch
        _print(f"torch: {torch.__version__}")
        try:
            _print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        except Exception:
            _print("torch.cuda.is_available: <failed to query>")
        try:
            _print(f"torch.version.cuda: {torch.version.cuda}")
        except Exception:
            _print("torch.version.cuda: <unknown>")
        try:
            if torch.cuda.is_available():
                _print(f"CUDA device count: {torch.cuda.device_count()}")
                try:
                    idx = torch.cuda.current_device()
                    _print(f"Current CUDA device index: {idx}")
                    _print(f"Current CUDA device name: {torch.cuda.get_device_name(idx)}")
                except Exception:
                    _print("CUDA device info: <failed to query device name>")
        except Exception:
            pass
    except Exception as e:
        _print(f"torch: not installed ({e})")

    # torchvision
    try:
        import torchvision
        _print(f"torchvision: {torchvision.__version__}")
    except Exception as e:
        _print(f"torchvision: not installed ({e})")

    # torchaudio
    try:
        import torchaudio
        _print(f"torchaudio: {torchaudio.__version__}")
    except Exception as e:
        _print(f"torchaudio: not installed ({e})")

    # ultralytics
    try:
        import ultralytics
        _print(f"ultralytics: {ultralytics.__version__}")
    except Exception as e:
        _print(f"ultralytics: not installed ({e})")

    # opencv
    try:
        import cv2
        _print(f"opencv: {cv2.__version__}")
    except Exception as e:
        _print(f"opencv: not installed ({e})")

    _print("=== End ===")


if __name__ == "__main__":
    main()
