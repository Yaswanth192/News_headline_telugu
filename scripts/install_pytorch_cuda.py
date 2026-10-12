#!/usr/bin/env python3
"""
scripts/install_pytorch_cuda.py
================================
Phase 0 — Install PyTorch with CUDA 12.1 support.

This script detects the installed CUDA driver and installs the appropriate
PyTorch CUDA build into the current virtual environment.

Run (with venv activated):
    python scripts/install_pytorch_cuda.py

The RTX 3050 driver 592.82 supports CUDA 13.1.
PyTorch CUDA 12.1 wheels are compatible (backward compatibility).
"""

import subprocess
import sys
import shutil


def main():
    print("=" * 60)
    print("PyTorch CUDA Installation Script")
    print("=" * 60)

    # Verify we're in the project venv
    python_exec = sys.executable
    print(f"\nPython executable: {python_exec}")
    print("Verify this is inside your project venv before continuing.")
    answer = input("\nProceed with installation? [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    # Install PyTorch with CUDA 12.1 (compatible with driver 592.82 / CUDA 13.1)
    torch_cmd = [
        python_exec, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "--upgrade"
    ]

    print("\nInstalling PyTorch + torchvision + torchaudio (CUDA 12.1 build)...")
    print(f"Command: {' '.join(torch_cmd)}")
    print("This may take several minutes (~2.5 GB download).\n")

    result = subprocess.run(torch_cmd)
    if result.returncode != 0:
        print("\n[ERROR] PyTorch installation failed.")
        sys.exit(1)

    print("\n[OK] PyTorch installed.")

    # Verify
    verify_cmd = [
        python_exec, "-c",
        "import torch; print('Version:', torch.__version__); "
        "print('CUDA:', torch.cuda.is_available()); "
        "print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    ]
    print("\nVerifying installation...")
    subprocess.run(verify_cmd)


if __name__ == "__main__":
    main()
