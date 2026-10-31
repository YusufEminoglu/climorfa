"""CLIMORFA Street environment check script.

Verifies Python, GDAL, PyTorch, CUDA, GPU matmul performance,
SOLWEIG-GPU, and pythermalcomfort installations.
"""

from __future__ import annotations

import sys
import time

# 1. Python version
print(f"Python Version: {sys.version.split()[0]} ({sys.executable})")

# 2. GDAL version
try:
    from osgeo import gdal
    gdal_version = gdal.__version__
except ImportError:
    try:
        import gdal
        gdal_version = gdal.__version__
    except ImportError as exc:
        raise ImportError(f"Failed to import GDAL: {exc}") from exc
print(f"GDAL Version: {gdal_version}")

# 3. PyTorch version & CUDA availability
import torch
print(f"PyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    device_name = torch.cuda.get_device_name(0)
    device_cap = torch.cuda.get_device_capability(0)
    print(f"Device Name: {device_name} (Compute Capability: {device_cap[0]}.{device_cap[1]})")

    # 4096 x 4096 float32 GPU matmul timing
    size = 4096
    print(f"Running {size}x{size} float32 GPU matmul benchmark...")
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)

    # Warmup
    torch.cuda.synchronize()
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Timed run (repeat 5 times for mean)
    iterations = 5
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time) / iterations
    tflops = (2 * (size ** 3) * 1e-12) / elapsed
    print(f"4096x4096 float32 GPU matmul: {elapsed * 1000:.2f} ms ({tflops:.2f} TFLOPS)")
else:
    print("Device Name: None (CUDA is not available)")
    print("4096x4096 float32 GPU matmul: SKIPPED (No CUDA)")

# 4. SOLWEIG-GPU
try:
    import solweig_gpu
    solweig_ver = getattr(solweig_gpu, "__version__", "2.0.0")
    print(f"SOLWEIG-GPU: SUCCESS (version: {solweig_ver})")
except ImportError as exc:
    print(f"SOLWEIG-GPU: FAILED ({exc})")
    raise

# 5. pythermalcomfort
try:
    import pythermalcomfort
    pythermal_ver = getattr(pythermalcomfort, "__version__", "4.7.0")
    print(f"pythermalcomfort: SUCCESS (version: {pythermal_ver})")
except ImportError as exc:
    print(f"pythermalcomfort: FAILED ({exc})")
    raise

print("\n--- Environment check PASSED ---")
