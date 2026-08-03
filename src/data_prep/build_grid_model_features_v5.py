"""Merge v4 model features with exact building-footprint morphometrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUTS = [
    "data/03_processed/grid_250m_model_features_v4.csv",
    "data/03_processed/grid_250m_building_morphometrics.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_model_features_v5.csv")
    return parser.parse_args()


def read_csv_by_grid_id(path: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"No header in {path}")
        if "grid_id" not in reader.fieldnames:
            raise RuntimeError(f"grid_id not found in {path}")
        rows = {row["grid_id"]: row for row in reader}
    return list(reader.fieldnames), rows


def main() -> None:
    args = parse_args()
    all_fields: list[str] = []
    merged: dict[str, dict[str, str]] = {}
    base_ids: list[str] = []
    missing_by_input: dict[str, int] = {}

    for input_index, path in enumerate(args.inputs):
        fields, rows = read_csv_by_grid_id(path)
        if input_index == 0:
            base_ids = list(rows.keys())
            merged = {grid_id: dict(row) for grid_id, row in rows.items()}
            all_fields = list(fields)
            continue
        for field in fields:
            if field != "grid_id" and field not in all_fields:
                all_fields.append(field)
        missing = 0
        for grid_id in base_ids:
            row = rows.get(grid_id)
            if row is None:
                missing += 1
                continue
            for field in fields:
                if field != "grid_id":
                    merged[grid_id][field] = row.get(field, "")
        missing_by_input[path] = missing

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        for grid_id in base_ids:
            writer.writerow({field: merged[grid_id].get(field, "") for field in all_fields})

    print(f"rows={len(base_ids)}")
    print(f"columns={len(all_fields)}")
    for path, missing in missing_by_input.items():
        print(f"missing_from_{path}={missing}")
    print(f"out_csv={out_path}")


if __name__ == "__main__":
    main()
