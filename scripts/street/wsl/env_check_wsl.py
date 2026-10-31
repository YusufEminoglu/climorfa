#!/usr/bin/env python3
"""
Environment check and SuPy smoke test for climorfa-wsl.
Work Package: WP0.2
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import importlib.metadata
import supy
import xarray
import cdsapi

def get_pkg_version(mod_name, mod_obj):
    if hasattr(mod_obj, "__version__"):
        return mod_obj.__version__
    try:
        return importlib.metadata.version(mod_name)
    except Exception:
        return "installed"

def main():
    print("=== ENVIRONMENT CHECK (climorfa-wsl) ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"supy version:   {get_pkg_version('supy', supy)}")
    print(f"xarray version: {get_pkg_version('xarray', xarray)}")
    print(f"cdsapi version: {get_pkg_version('cdsapi', cdsapi)}")
    print()

    print("=== SUPY SMOKE TEST ===")
    print("Loading bundled sample data...")
    df_state_init, df_forcing = supy.load_SampleData()

    # Slice for 1 day: 2012-01-01
    df_forcing_1d = df_forcing.loc["2012-01-01"]
    print(f"Forcing sample slice: {len(df_forcing_1d)} timesteps ({df_forcing_1d.index[0]} to {df_forcing_1d.index[-1]})")

    print("Running SuPy simulation for 1 day...")
    df_output, df_state_final = supy.run_supy(df_forcing_1d, df_state_init)

    qh = df_output[("SUEWS", "QH")]
    t2 = df_output[("SUEWS", "T2")]

    qh_mean = float(qh.mean())
    t2_mean = float(t2.mean())

    print("\n=== SMOKE TEST RESULTS ===")
    print(f"Mean Sensible Heat Flux (QH): {qh_mean:.4f} W/m^2")
    print(f"Mean 2-meter Air Temperature (T2): {t2_mean:.4f} degC")
    print("SuPy smoke test: SUCCESS (exit 0)")

    # Clean up / relocate SuPy.log created in cwd to data/street/wsl/
    cwd_log = Path("SuPy.log")
    if cwd_log.exists():
        target_dir = Path("data/street/wsl")
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            cwd_log.rename(target_dir / "SuPy.log")
        except Exception:
            pass

if __name__ == "__main__":
    main()
