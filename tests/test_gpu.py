"""
tests/test_gpu.py
=================
Phase 0 — GPU visibility and CUDA smoke test.

Run:
    python tests/test_gpu.py

Expected output on success:
    [PASS] PyTorch imported successfully. Version: X.Y.Z
    [PASS] CUDA is available. Device: NVIDIA GeForce RTX 3050 Laptop GPU
    [PASS] VRAM: 4096 MiB
    [PASS] GPU tensor operation successful. Result: 6.0
    [PASS] Mixed precision (FP16) supported.
    All GPU checks passed.
"""

import sys
import json
import datetime

def main():
    results = {}
    all_passed = True

    # 1. Import PyTorch
    try:
        import torch
        results["pytorch_version"] = torch.__version__
        print(f"[PASS] PyTorch imported successfully. Version: {torch.__version__}")
    except ImportError as e:
        print(f"[FAIL] Cannot import PyTorch: {e}")
        print("       Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    # 2. CUDA availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_mib = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        cuda_version = torch.version.cuda
        results["cuda_available"] = True
        results["device_name"] = device_name
        results["vram_mib"] = vram_mib
        results["cuda_version"] = cuda_version
        print(f"[PASS] CUDA is available.")
        print(f"       Device:       {device_name}")
        print(f"       VRAM:         {vram_mib} MiB")
        print(f"       CUDA version: {cuda_version}")
    else:
        print("[FAIL] CUDA is NOT available. PyTorch cannot see the GPU.")
        print("       Check: (1) installed torch with CUDA build, (2) driver version, (3) venv activation.")
        results["cuda_available"] = False
        all_passed = False

    # 3. GPU tensor operation
    if results.get("cuda_available"):
        try:
            a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            b = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            c = (a + b).sum().item()
            assert abs(c - 12.0) < 1e-4, f"Expected 12.0, got {c}"
            results["gpu_tensor_test"] = True
            print(f"[PASS] GPU tensor operation successful. Result: {c}")
        except Exception as e:
            print(f"[FAIL] GPU tensor operation failed: {e}")
            results["gpu_tensor_test"] = False
            all_passed = False

    # 4. Mixed precision check
    if results.get("cuda_available"):
        try:
            props = torch.cuda.get_device_properties(0)
            # Ampere (RTX 30xx) = compute capability 8.x
            cc_major = props.major
            cc_minor = props.minor
            fp16_ok = (cc_major >= 7)  # Volta+ supports FP16 Tensor Cores
            results["compute_capability"] = f"{cc_major}.{cc_minor}"
            results["fp16_supported"] = fp16_ok
            if fp16_ok:
                print(f"[PASS] Mixed precision (FP16) supported. Compute capability: {cc_major}.{cc_minor}")
            else:
                print(f"[WARN] FP16 Tensor Cores may not be optimal. Compute: {cc_major}.{cc_minor}")
        except Exception as e:
            print(f"[WARN] Could not check compute capability: {e}")

    # 5. Memory summary
    if results.get("cuda_available"):
        allocated = torch.cuda.memory_allocated(0) // (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) // (1024 ** 2)
        free = results["vram_mib"] - reserved
        results["vram_free_mib"] = free
        print(f"\n[INFO] VRAM Summary:")
        print(f"       Total:     {results['vram_mib']} MiB")
        print(f"       Allocated: {allocated} MiB")
        print(f"       Reserved:  {reserved} MiB")
        print(f"       Free est.: {free} MiB")

    # 6. Save results
    results["timestamp"] = datetime.datetime.now().isoformat()
    results["all_passed"] = all_passed

    import pathlib
    log_dir = pathlib.Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    report_path = log_dir / "phase0_gpu_test.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to: {report_path}")

    if all_passed:
        print("\n[ALL PASSED] All GPU checks passed.")
        return 0
    else:
        print("\n[FAILED] Some GPU checks failed. See above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
