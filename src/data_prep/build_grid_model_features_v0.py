"""Merge current grid-level feature tables into a first model matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prelim-csv", default="data/03_processed/grid_250m_prelim_features.csv")
    parser.add_argument("--lcz-csv", default="data/03_processed/grid_250m_lcz_weak_labels.csv")
    parser.add_argument("--rs-csv", default="data/03_processed/grid_250m_remote_sensing_features.csv")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_model_features_v0.csv")
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
    prelim_fields, prelim = read_csv_by_grid_id(args.prelim_csv)
    lcz_fields, lcz = read_csv_by_grid_id(args.lcz_csv)
    rs_fields, rs = read_csv_by_grid_id(args.rs_csv)

    out_fields = list(prelim_fields)
    for fields in (lcz_fields, rs_fields):
        for field in fields:
            if field != "grid_id" and field not in out_fields:
                out_fields.append(field)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing_lcz = 0
    missing_rs = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        for grid_id, base_row in prelim.items():
            out_row = {field: base_row.get(field, "") for field in out_fields}
            lcz_row = lcz.get(grid_id)
            rs_row = rs.get(grid_id)
            if lcz_row is None:
                missing_lcz += 1
            else:
                for field in lcz_fields:
                    if field != "grid_id":
                        out_row[field] = lcz_row.get(field, "")
            if rs_row is None:
                missing_rs += 1
            else:
                for field in rs_fields:
                    if field != "grid_id":
                        out_row[field] = rs_row.get(field, "")
            writer.writerow(out_row)

    print(f"rows={len(prelim)}")
    print(f"columns={len(out_fields)}")
    print(f"missing_lcz={missing_lcz}")
    print(f"missing_remote_sensing={missing_rs}")
    print(f"out_csv={out_path}")


if __name__ == "__main__":
    main()
