"""Merge v5 model features with street-facing morphology proxy metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_INPUTS = [
    "data/03_processed/grid_250m_model_features_v5.csv",
    "data/03_processed/grid_250m_street_facing_morphology.csv",
]


DERIVED_FIELDS = [
    "street_frontage_road_presence_flag",
    "street_frontage_building_presence_flag",
    "street_frontage_continuity_available_flag",
    "block_permeability_network_proxy",
    "block_permeability_street_open_network_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_model_features_v6.csv")
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


def as_float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def fmt_float(value: float | None) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.10g}"


def bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


def add_derived_fields(row: dict[str, str]) -> None:
    road_length = as_float_or_none(row.get("street_frontage_road_length_m")) or 0.0
    building_count = as_float_or_none(row.get("street_frontage_building_count")) or 0.0
    continuity = as_float_or_none(row.get("street_frontage_continuity_proxy"))
    open_share = as_float_or_none(row.get("street_frontage_open_buffer_share"))

    row["street_frontage_road_presence_flag"] = "1" if road_length > 0 else "0"
    row["street_frontage_building_presence_flag"] = "1" if building_count > 0 else "0"
    row["street_frontage_continuity_available_flag"] = "1" if continuity is not None else "0"

    endpoint_nodes = as_float_or_none(row.get("network_endpoint_node_count")) or 0.0
    intersection_nodes = as_float_or_none(row.get("network_intersection_node_count")) or 0.0
    dead_end_share = as_float_or_none(row.get("network_dead_end_share"))

    if endpoint_nodes > 0:
        intersection_share = bounded(intersection_nodes / endpoint_nodes)
        dead_end_penalty = bounded(1.0 - dead_end_share) if dead_end_share is not None else 0.0
        network_proxy = bounded((intersection_share + dead_end_penalty) / 2.0)
    elif road_length > 0:
        network_proxy = 0.0
    else:
        network_proxy = 0.0

    row["block_permeability_network_proxy"] = fmt_float(network_proxy)
    if open_share is None:
        row["block_permeability_street_open_network_proxy"] = "0" if road_length <= 0 else ""
    else:
        row["block_permeability_street_open_network_proxy"] = fmt_float(bounded(open_share) * network_proxy)


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

    for derived in DERIVED_FIELDS:
        if derived not in all_fields:
            all_fields.append(derived)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        for grid_id in base_ids:
            row = {field: merged[grid_id].get(field, "") for field in all_fields}
            add_derived_fields(row)
            writer.writerow(row)

    print(f"rows={len(base_ids)}")
    print(f"columns={len(all_fields)}")
    for path, missing in missing_by_input.items():
        print(f"missing_from_{path}={missing}")
    print(f"out_csv={out_path}")


if __name__ == "__main__":
    main()
