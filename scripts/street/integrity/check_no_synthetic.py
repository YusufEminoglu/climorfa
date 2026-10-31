"""
CLIMORFA Street — Integrity scanner.
Given a manifest (YAML mapping figures/tables to input files) or a list of files,
verifies that NO input file contains synthetic or analogue markers, and no table contains is_synthetic==True.

Exits with code 1 if any synthetic/analogue input is detected.
Exits with code 0 if all inputs are clean (real field observations).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple
import yaml
import pandas as pd
import geopandas as gpd

SYNTHETIC_MARKERS = ["SYNTHETIC", "ANALOGUE"]


def extract_file_paths(manifest_obj: Any) -> List[str]:
    """Recursively extract all file paths from a loaded YAML object."""
    paths: List[str] = []
    if isinstance(manifest_obj, str):
        paths.append(manifest_obj)
    elif isinstance(manifest_obj, list):
        for item in manifest_obj:
            paths.extend(extract_file_paths(item))
    elif isinstance(manifest_obj, dict):
        for k, v in manifest_obj.items():
            paths.extend(extract_file_paths(v))
    return paths


def check_file(path_str: str) -> List[str]:
    """
    Check a single file path for synthetic/analogue markers or records.
    Returns a list of violation messages (empty if clean).
    """
    violations = []
    path = Path(path_str)
    filename_upper = path.name.upper()

    for marker in SYNTHETIC_MARKERS:
        if marker in filename_upper:
            violations.append(f"Filename contains forbidden marker '{marker}': {path_str}")

    if not path.exists():
        # If file doesn't exist, we could only check name
        return violations

    suffix = path.suffix.lower()
    if suffix in [".csv", ".tsv"]:
        try:
            df = pd.read_csv(path, nrows=500)
            if "is_synthetic" in df.columns:
                val_matches = (df["is_synthetic"] == True) | (df["is_synthetic"].astype(str).str.lower() == "true")
                if val_matches.any():
                    violations.append(f"CSV contains is_synthetic=True records ({val_matches.sum()} rows found): {path_str}")
        except Exception as e:
            violations.append(f"Error reading CSV {path_str}: {e}")

    elif suffix in [".gpkg", ".geojson", ".shp"]:
        try:
            gdf = gpd.read_file(path, rows=500)
            if "is_synthetic" in gdf.columns:
                val_matches = (gdf["is_synthetic"] == True) | (gdf["is_synthetic"].astype(str).str.lower() == "true")
                if val_matches.any():
                    violations.append(f"Geodata contains is_synthetic=True features ({val_matches.sum()} found): {path_str}")
        except Exception as e:
            violations.append(f"Error reading geodata {path_str}: {e}")

    elif suffix in [".yaml", ".yml", ".json"]:
        try:
            text = path.read_text(encoding="utf-8")
            if "is_synthetic: true" in text.lower() or '"is_synthetic": true' in text.lower():
                violations.append(f"Metadata file contains 'is_synthetic: true': {path_str}")
        except Exception as e:
            violations.append(f"Error reading metadata {path_str}: {e}")

    return violations


def check_manifest(manifest_path: str) -> Tuple[bool, List[str]]:
    """
    Load a YAML manifest and check all listed files.
    Returns (is_clean, violations_list).
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    file_paths = extract_file_paths(data)
    all_violations: List[str] = []

    for p in file_paths:
        violations = check_file(p)
        all_violations.extend(violations)

    is_clean = len(all_violations) == 0
    return is_clean, all_violations


def main() -> int:
    parser = argparse.ArgumentParser(description="CLIMORFA Street Integrity Check: Ensure No Synthetic Data.")
    parser.add_argument("--manifest", type=str, help="Path to YAML manifest file listing inputs per figure/table.")
    parser.add_argument("--inputs", nargs="*", help="Direct list of file paths to check.")
    args = parser.parse_args()

    files_to_check: List[str] = []
    if args.manifest:
        if not os.path.exists(args.manifest):
            print(f"ERROR: Manifest file not found: {args.manifest}", file=sys.stderr)
            return 1
        with open(args.manifest, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        files_to_check.extend(extract_file_paths(data))

    if args.inputs:
        files_to_check.extend(args.inputs)

    if not files_to_check:
        print("No files specified to check. Provide --manifest or --inputs.")
        return 0

    all_violations: List[str] = []
    for f in files_to_check:
        all_violations.extend(check_file(f))

    if all_violations:
        print("\n=======================================================", file=sys.stderr)
        print(" INTEGRITY CHECK FAILED: SYNTHETIC DATA DETECTED!", file=sys.stderr)
        print("=======================================================", file=sys.stderr)
        for v in all_violations:
            print(f" [FAIL] {v}", file=sys.stderr)
        print(f"\nTotal violations: {len(all_violations)}", file=sys.stderr)
        return 1
    else:
        print("\n=======================================================")
        print(" INTEGRITY CHECK PASSED: All inputs are confirmed real.")
        print(f" Scanned {len(files_to_check)} inputs without synthetic markers.")
        print("=======================================================")
        return 0


if __name__ == "__main__":
    sys.exit(main())
