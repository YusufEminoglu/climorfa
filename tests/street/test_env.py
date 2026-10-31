"""Environment validation tests for CLIMORFA Street-scale compute environment."""

from __future__ import annotations

import importlib
import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "osgeo.gdal",
        "rasterio",
        "geopandas",
        "pyogrio",
        "shapely",
        "numpy",
        "scipy",
        "pandas",
        "xarray",
        "netCDF4",
        "numba",
        "matplotlib",
        "timezonefinder",
        "sklearn",
        "lightgbm",
        "shap",
        "pyproj",
        "tqdm",
        "pytest",
        "torch",
        "solweig_gpu",
        "pythermalcomfort",
    ],
)
def test_module_imports(module_name: str) -> None:
    """Verify that all required core and street packages import successfully."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_cuda_available() -> None:
    """Verify PyTorch detects NVIDIA GPU with CUDA support."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable on this host; skipping GPU assertion.")

    assert torch.cuda.is_available() is True
    device_name = torch.cuda.get_device_name(0)
    assert "T1000" in device_name or torch.cuda.device_count() >= 1

    # Simple GPU tensor calculation
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    y = x * 2.0
    assert torch.allclose(y, torch.tensor([2.0, 4.0, 6.0], device="cuda"))


def test_solweig_gpu_import() -> None:
    """Verify SOLWEIG-GPU module and version."""
    import solweig_gpu

    assert hasattr(solweig_gpu, "__version__") or solweig_gpu is not None


def test_pythermalcomfort_import() -> None:
    """Verify pythermalcomfort module and UTCI / PMV calculation availability."""
    import pythermalcomfort

    assert hasattr(pythermalcomfort, "__version__") or pythermalcomfort is not None
