"""Triangulate OSM park/garden 2SFCA supply with the official IZBB inventory.

The official CSVs have district, neighbourhood, address, name, type and area,
but no coordinates. They support district/name audits, not polygon ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import unicodedata
import urllib.request
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
from osgeo import ogr

ogr.UseExceptions()
PACKAGE_ID = "72a1e584-ac48-41b1-b993-12c6f1cd1a63"
PACKAGE_API = f"https://acikveri.bizizmir.com/api/3/action/package_show?id={PACKAGE_ID}"
RESOURCE_IDS = {
    "south": "ab95e4b0-8db3-40b3-8de8-494d57a3135e",
    "north": "ce856a4d-b898-4876-afe5-ecfaba74bdcf",
}
EXTENDED_TYPES = {"park", "yesil alan", "yesil alani", "rekreasyon alani", "kiyi seridi"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/01_raw/green_space")
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--model-csv", default="data/03_processed/grid_250m_model_features_v8.csv")
    parser.add_argument("--supply-gpkg", default="data/03_processed/green_space_2sfca_supply_audit.gpkg")
    parser.add_argument("--supply-layer", default="green_space_2sfca_supply_audit")
    parser.add_argument("--out-dir", default=f"outputs/diagnostics/green_space_official_validation_{date.today().isoformat()}")
    return parser.parse_args()


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "CLIMORFA-research-audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def immutable_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite changed raw source: {path}")
        return
    path.write_bytes(payload)


def decode(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1254"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise RuntimeError("Official CSV encoding could not be decoded")


def norm(value: Any) -> str:
    text = str(value or "").strip().lower().replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def norm_name(value: Any) -> str:
    text = norm(value)
    stop = {"park", "parki", "yesil", "alani", "mahalle", "mahallesi"}
    core = " ".join(token for token in text.split() if token not in stop)
    return core or text


def norm_mahalle_key(value: Any) -> str:
    parts = str(value or "").split("|", 1)
    if len(parts) != 2:
        return ""
    return f"{norm(parts[0])}|{norm_name(parts[1])}"


def numeric_area(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0).clip(lower=0.0)


def load_official(package: dict[str, Any], raw_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    resources = {item["id"]: item for item in package["resources"]}
    frames: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    for branch, resource_id in RESOURCE_IDS.items():
        resource = resources[resource_id]
        raw_path = raw_dir / f"izbb_parks_{branch}_modified_2023-04-26.csv"
        payload = raw_path.read_bytes() if raw_path.exists() else fetch(resource["url"])
        immutable_write(raw_path, payload)
        frame = pd.read_csv(io.StringIO(decode(payload)), sep=";", dtype=str).fillna("")
        frame.insert(0, "branch", branch)
        frame.insert(1, "source_row", range(1, len(frame) + 1))
        frames.append(frame)
        provenance.append({
            "branch": branch, "resource_id": resource_id, "url": resource["url"],
            "created": resource.get("created"), "last_modified": resource.get("last_modified"),
            "raw_path": raw_path.as_posix(), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    source = pd.concat(frames, ignore_index=True)
    clean = pd.DataFrame({
        "official_id": source["branch"] + "_" + source["source_row"].astype(str).str.zfill(4),
        "branch": source["branch"], "green_type": source["YESIL_ALAN_TURU"].str.strip(),
        "district": source["ILCE"].str.strip(), "neighbourhood": source["MAHALLE"].str.strip(),
        "address": source["ADRES"].str.strip(), "park_name": source["PARK_ADI"].str.strip(),
        "reported_green_area_m2": numeric_area(source["YESIL_ALAN_MIKTARI"]),
        "fitness_equipment": source["KONDISYON_TAKIMI"].str.strip(),
        "playground": source["OYUN_GRUBU"].str.strip(),
    })
    clean["green_type_norm"] = clean["green_type"].map(norm)
    clean["district_norm"] = clean["district"].map(norm)
    clean["neighbourhood_norm"] = clean["neighbourhood"].map(norm_name)
    clean["mahalle_key_norm"] = clean["district_norm"] + "|" + clean["neighbourhood_norm"]
    clean["park_name_norm"] = clean["park_name"].map(norm_name)
    fallback = (clean["neighbourhood"] + " " + clean["address"]).map(norm_name)
    clean["site_name_key"] = clean["park_name_norm"].where(clean["park_name_norm"].ne(""), fallback)
    missing = clean["site_name_key"].eq("")
    clean.loc[missing, "site_name_key"] = clean.loc[missing, "official_id"]
    clean["site_key"] = clean["district_norm"] + "|" + clean["site_name_key"]
    clean["park_comparable_flag"] = clean["green_type_norm"].eq("park").astype(int)
    clean["extended_green_flag"] = clean["green_type_norm"].isin(EXTENDED_TYPES).astype(int)
    return clean, provenance


def collapse_sites(clean: pd.DataFrame) -> pd.DataFrame:
    first_fields = ["branch", "green_type", "green_type_norm", "district", "district_norm", "neighbourhood", "neighbourhood_norm", "mahalle_key_norm", "address", "park_name", "park_name_norm", "site_name_key"]
    aggregation: dict[str, Any] = {field: "first" for field in first_fields}
    aggregation.update({"reported_green_area_m2": "sum", "park_comparable_flag": "max", "extended_green_flag": "max", "official_id": "size"})
    sites = clean.groupby("site_key", as_index=False).agg(aggregation)
    return sites.rename(columns={"official_id": "source_record_count"})


def open_layer(path: Path, layer_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    datasource = ogr.Open(str(path))
    if datasource is None:
        raise RuntimeError(f"Could not open {path}")
    layer = datasource.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Could not find {layer_name} in {path}")
    return datasource, layer


def load_osm(args: argparse.Namespace) -> pd.DataFrame:
    model = pd.read_csv(args.model_csv, usecols=["grid_id", "district", "population_mahalle_key"], dtype=str).fillna("")
    district_by_grid = model.set_index("grid_id")["district"].to_dict()
    model["mahalle_key_norm"] = model["population_mahalle_key"].map(norm_mahalle_key)
    mahalle_key_by_grid = model.set_index("grid_id")["mahalle_key_norm"].to_dict()
    grid_ds, grid = open_layer(Path(args.grid_gpkg), args.grid_layer)
    supply_ds, supplies = open_layer(Path(args.supply_gpkg), args.supply_layer)
    rows: list[dict[str, Any]] = []
    for feature in supplies:
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue
        point = geometry.Centroid()
        grid.SetSpatialFilter(point)
        grid_id = ""
        district = ""
        mahalle_key = ""
        for cell in grid:
            cell_geometry = cell.GetGeometryRef()
            if cell_geometry and (cell_geometry.Contains(point) or cell_geometry.Intersects(point)):
                grid_id = str(cell.GetField("grid_id"))
                district = district_by_grid.get(grid_id, "")
                mahalle_key = mahalle_key_by_grid.get(grid_id, "")
                break
        grid.SetSpatialFilter(None)
        grid.ResetReading()
        name = str(feature.GetField("name") or "").strip()
        rows.append({
            "supply_id": str(feature.GetField("supply_id")), "green_type": str(feature.GetField("green_type") or ""),
            "name": name, "name_norm": norm_name(name), "supply_area_m2": float(feature.GetField("supply_area_m2") or 0.0),
            "network_available_flag": int(feature.GetField("network_available_flag") or 0), "grid_id": grid_id,
            "district": district, "district_norm": norm(district), "mahalle_key_norm": mahalle_key,
        })
    grid_ds = None
    supply_ds = None
    return pd.DataFrame(rows)


def load_fua_neighbourhood_keys(model_csv: str) -> set[str]:
    model = pd.read_csv(model_csv, usecols=["population_mahalle_key"], dtype=str).fillna("")
    return {key for key in model["population_mahalle_key"].map(norm_mahalle_key) if key}


def build_comparison(sites: pd.DataFrame, osm: pd.DataFrame, scope_keys: set[str] | None = None) -> pd.DataFrame:
    working = sites.copy()
    osm_working = osm.copy()
    if scope_keys is not None:
        working = working[working["mahalle_key_norm"].isin(scope_keys)]
        osm_working = osm_working[osm_working["mahalle_key_norm"].isin(scope_keys)]
    working["park_area_m2"] = working["reported_green_area_m2"] * working["park_comparable_flag"]
    working["extended_area_m2"] = working["reported_green_area_m2"] * working["extended_green_flag"]
    official = working.groupby(["district_norm", "district"], as_index=False).agg(
        official_park_sites=("park_comparable_flag", "sum"), official_park_area_m2=("park_area_m2", "sum"),
        official_extended_sites=("extended_green_flag", "sum"), official_extended_area_m2=("extended_area_m2", "sum"),
    )
    official = official[official["official_park_sites"] > 0]
    osm_summary = osm_working[osm_working["district_norm"].ne("")].groupby("district_norm", as_index=False).agg(
        osm_polygon_count=("supply_id", "size"), osm_named_polygon_count=("name_norm", lambda x: int(x.ne("").sum())),
        osm_area_m2=("supply_area_m2", "sum"), osm_network_available_count=("network_available_flag", "sum"),
    )
    return official.merge(osm_summary, on="district_norm", how="left").fillna(0)


def build_matches(sites: pd.DataFrame, osm: pd.DataFrame) -> pd.DataFrame:
    named = osm[(osm["district_norm"].ne("")) & (osm["name_norm"].ne(""))]
    by_district = {key: frame.to_dict("records") for key, frame in named.groupby("district_norm")}
    by_neighbourhood = {key: frame.to_dict("records") for key, frame in named[named["mahalle_key_norm"].ne("")].groupby("mahalle_key_norm")}
    rows: list[dict[str, Any]] = []
    for official in sites[sites["park_comparable_flag"].eq(1)].to_dict("records"):
        best: dict[str, Any] | None = None
        score = 0.0
        candidates = by_neighbourhood.get(official["mahalle_key_norm"], []) if official["fua_neighbourhood_flag"] else by_district.get(official["district_norm"], [])
        for candidate in candidates:
            candidate_score = SequenceMatcher(None, official["park_name_norm"], candidate["name_norm"]).ratio()
            if candidate_score > score:
                score, best = candidate_score, candidate
        status = "exact_normalized" if score >= 0.999 else "strong_candidate" if score >= 0.82 else "possible_candidate" if score >= 0.68 else "unmatched"
        rows.append({
            "official_site_key": official["site_key"], "district": official["district"], "neighbourhood": official["neighbourhood"],
            "fua_neighbourhood_flag": int(official["fua_neighbourhood_flag"]), "official_name": official["park_name"],
            "official_area_m2": official["reported_green_area_m2"], "best_osm_supply_id": best["supply_id"] if best else "",
            "best_osm_name": best["name"] if best else "", "best_osm_area_m2": best["supply_area_m2"] if best else "",
            "name_similarity": round(score, 6), "match_status": status, "manual_review_required": int(status != "exact_normalized"),
        })
    return pd.DataFrame(rows)


def log_correlation(frame: pd.DataFrame, left: str, right: str) -> float | None:
    if len(frame) < 3:
        return None
    value = frame[[left, right]].apply(lambda column: column.map(math.log1p)).corr().iloc[0, 1]
    return None if pd.isna(value) else float(value)


def rank_correlation(frame: pd.DataFrame, left: str, right: str) -> float | None:
    if len(frame) < 3:
        return None
    value = frame[[left, right]].rank(method="average").corr().iloc[0, 1]
    return None if pd.isna(value) else float(value)


def leave_one_out_range(frame: pd.DataFrame, left: str, right: str) -> dict[str, float | None]:
    values = [log_correlation(frame.drop(index), left, right) for index in frame.index]
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return {"minimum": min(finite) if finite else None, "maximum": max(finite) if finite else None}


def render_report(summary: dict[str, Any], comparison: pd.DataFrame) -> str:
    matches = summary["name_matches"]
    lines = [
        "# Official IZBB Green-Space Inventory Audit of OSM 2SFCA Supply", "", f"Run date: {summary['run_date']}  ",
        "Status: external triangulation completed; polygon-ground-truth validation remains unavailable", "", "## Decision", "",
        "The official IZBB inventory supports an independent district-level and name-based audit, but it cannot replace OSM supply geometry because it contains no coordinates. It is limited to parks under the responsibility of the north and south maintenance branches and was last modified in 2023. OSM park/garden area therefore remains a proxy input to 2SFCA. Strong distributive-equity claims remain out of scope until a current geocoded municipal inventory or WFS is supplied.", "", "## Source and Coverage", "",
        f"- Official dataset: [{summary['official_dataset_title']}](https://acikveri.bizizmir.com/dataset/{PACKAGE_ID})",
        f"- Official organization: {summary['official_organization']}", f"- Metadata modified: {summary['official_metadata_modified']}",
        f"- Official source records / distinct sites: {summary['official_source_records']:,} / {summary['official_distinct_sites']:,}",
        f"- All official Park sites and reported area: {summary['official_park_sites']:,} / {summary['official_park_area_m2']:,.0f} m2",
        f"- FUA-neighbourhood-matched official Park sites: {summary['official_park_sites_fua_neighbourhood_matched']:,}",
        f"- Comparable neighbourhoods / OSM polygons in them: {summary['comparable_neighbourhoods']:,} / {summary['osm_polygons_in_comparable_neighbourhoods']:,}",
        f"- All OSM park/garden polygons / named polygons: {summary['osm_supply_polygons']:,} / {summary['osm_named_polygons']:,}", "",
        "## Name-Match Screening", "", f"- Exact normalized: {matches['exact_normalized']:,}",
        f"- Strong candidate: {matches['strong_candidate']:,}", f"- Possible candidate: {matches['possible_candidate']:,}",
        f"- Unmatched: {matches['unmatched']:,}", f"- Exact + strong share: {matches['exact_or_strong_share']:.1%}", "",
        "Name similarity is a screening aid, not identity proof. All non-exact candidates require review.", "", "## Visual Triangulation", "", "![Official IZBB versus OSM district triangulation](district_official_osm_triangulation.png)", "", "The labelled scatter uses log1p axes and descriptive least-squares fits. It is a supplement candidate, not a calibration or completeness plot.", "", "## District-Level Triangulation", "",
        "| District | Official park sites | Official park area m2 | OSM polygons | OSM area m2 |", "|---|---:|---:|---:|---:|",
    ]
    for row in comparison.to_dict("records"):
        lines.append(f"| {row['district']} | {int(row['official_park_sites'])} | {row['official_park_area_m2']:.0f} | {int(row['osm_polygon_count'])} | {row['osm_area_m2']:.0f} |")
    lines.extend([
        "", f"Across {summary['comparable_districts']} districts, log1p Pearson correlation: counts = {summary['district_log_count_correlation']:.3f}; areas = {summary['district_log_area_correlation']:.3f}. Spearman rank correlation: counts = {summary['district_rank_count_correlation']:.3f}; areas = {summary['district_rank_area_correlation']:.3f}. Leave-one-district-out log1p ranges: counts = {summary['district_log_count_leave_one_out']['minimum']:.3f}-{summary['district_log_count_leave_one_out']['maximum']:.3f}; areas = {summary['district_log_area_leave_one_out']['minimum']:.3f}-{summary['district_log_area_leave_one_out']['maximum']:.3f}.", "",
        "## Interpretation Lock", "", "1. The official CSV is not a complete municipal green-space census.",
        "2. The main correlation is restricted to official neighbourhood names matched to FUA population keys; it is triangulation across 12 districts, not a completeness coefficient.",
        "3. Reported municipal area is not assumed equivalent to OSM polygon area.",
        "4. Keep 800 m 2SFCA primary; retain 400/1200 m as threshold sensitivities.",
        "5. Use this audit to calibrate source-completeness claims, not to auto-relabel OSM polygons.",
        "6. Require current geocoded municipal green-space data before neighbourhood-level equity conclusions.", "", "## Outputs", "",
        "- official_inventory_clean.csv", "- official_distinct_sites.csv", "- osm_supply_with_district.csv",
        "- district_comparison.csv", "- district_comparison_raw_unscoped.csv", "- official_osm_name_match_candidates.csv", "- summary.json", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_response_path = raw_dir / "izbb_parks_package_api_response_2023-04-26.json"
    if api_response_path.exists():
        api_payload = json.loads(api_response_path.read_text(encoding="utf-8"))
    else:
        api_payload = json.loads(fetch(PACKAGE_API).decode("utf-8"))
        immutable_write(api_response_path, json.dumps(api_payload, ensure_ascii=False, indent=2).encode("utf-8"))
    package_payload = api_payload["result"] if "result" in api_payload else api_payload
    clean, provenance = load_official(package_payload, raw_dir)
    metadata = json.dumps(package_payload, ensure_ascii=False, indent=2).encode("utf-8")
    immutable_write(raw_dir / "izbb_parks_dataset_metadata_2023-04-26.json", metadata)
    sites = collapse_sites(clean)
    osm = load_osm(args)
    fua_keys = load_fua_neighbourhood_keys(args.model_csv)
    sites["fua_neighbourhood_flag"] = (sites["mahalle_key_norm"].isin(fua_keys) & sites["neighbourhood_norm"].ne("")).astype(int)
    comparable_sites = sites[(sites["park_comparable_flag"].eq(1)) & (sites["fua_neighbourhood_flag"].eq(1))]
    comparable_keys = set(comparable_sites["mahalle_key_norm"])
    raw_comparison = build_comparison(sites, osm)
    comparison = build_comparison(sites, osm, comparable_keys)
    matches = build_matches(sites, osm)
    fua_matches = matches[matches["fua_neighbourhood_flag"].eq(1)]
    statuses = ("exact_normalized", "strong_candidate", "possible_candidate", "unmatched")
    match_counts = {status: int(fua_matches["match_status"].eq(status).sum()) for status in statuses}
    all_match_counts = {status: int(matches["match_status"].eq(status).sum()) for status in statuses}
    park_area = float((sites["reported_green_area_m2"] * sites["park_comparable_flag"]).sum())
    extended_area = float((sites["reported_green_area_m2"] * sites["extended_green_flag"]).sum())
    osm_in_scope = osm[osm["mahalle_key_norm"].isin(comparable_keys)]
    summary = {
        "schema_version": "climorfa.green_space_official_validation.v2", "run_date": date.today().isoformat(),
        "official_dataset_title": package_payload["title"], "official_metadata_created": package_payload.get("metadata_created"),
        "official_metadata_modified": package_payload.get("metadata_modified"), "official_organization": package_payload.get("organization", {}).get("title"),
        "official_license": package_payload.get("license_title"), "official_source_records": len(clean), "official_distinct_sites": len(sites),
        "official_park_sites": int(sites["park_comparable_flag"].sum()), "official_park_area_m2": park_area,
        "official_extended_sites": int(sites["extended_green_flag"].sum()), "official_extended_area_m2": extended_area,
        "official_park_sites_fua_neighbourhood_matched": len(comparable_sites),
        "official_park_sites_outside_or_unresolved": int(sites["park_comparable_flag"].sum()) - len(comparable_sites),
        "comparable_neighbourhoods": len(comparable_keys),
        "official_green_type_counts": clean["green_type_norm"].value_counts().sort_index().to_dict(),
        "osm_supply_polygons": len(osm), "osm_district_assigned": int(osm["district_norm"].ne("").sum()),
        "osm_named_polygons": int(osm["name_norm"].ne("").sum()),
        "osm_polygons_in_comparable_neighbourhoods": len(osm_in_scope), "comparable_districts": len(comparison),
        "district_log_count_correlation": log_correlation(comparison, "official_park_sites", "osm_polygon_count"),
        "district_log_area_correlation": log_correlation(comparison, "official_park_area_m2", "osm_area_m2"),
        "district_rank_count_correlation": rank_correlation(comparison, "official_park_sites", "osm_polygon_count"),
        "district_rank_area_correlation": rank_correlation(comparison, "official_park_area_m2", "osm_area_m2"),
        "district_log_count_leave_one_out": leave_one_out_range(comparison, "official_park_sites", "osm_polygon_count"),
        "district_log_area_leave_one_out": leave_one_out_range(comparison, "official_park_area_m2", "osm_area_m2"),
        "name_matches": {**match_counts, "scope": "official Park sites matched to FUA neighbourhood keys", "rows": len(fua_matches), "exact_or_strong_share": (match_counts["exact_normalized"] + match_counts["strong_candidate"]) / len(fua_matches) if len(fua_matches) else 0.0},
        "name_matches_all_inventory": {**all_match_counts, "rows": len(matches)},
        "limitations": ["Official inventory has no coordinates and cannot serve as polygon ground truth.", "Inventory is limited to north/south maintenance-branch responsibility.", "Official and OSM scopes are compared only for official neighbourhood names matched to FUA keys; unresolved or blank neighbourhoods are excluded from the main correlation.", "Official metadata was last modified in 2023.", "OSM name matching requires manual confirmation."],
        "provenance": provenance,
    }
    clean.to_csv(out_dir / "official_inventory_clean.csv", index=False, encoding="utf-8-sig")
    sites.to_csv(out_dir / "official_distinct_sites.csv", index=False, encoding="utf-8-sig")
    osm.to_csv(out_dir / "osm_supply_with_district.csv", index=False, encoding="utf-8-sig")
    raw_comparison.to_csv(out_dir / "district_comparison_raw_unscoped.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(out_dir / "district_comparison.csv", index=False, encoding="utf-8-sig")
    matches.to_csv(out_dir / "official_osm_name_match_candidates.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(summary, comparison), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()








