#!/usr/bin/env python3
"""
Fetch ERA5 and ERA5-Land hourly data via Google Earth Engine for selected design days:
- IMBAT: 2021-08-09
- CALM HEAT: 2023-07-26
Extracts at Izmir centre (38.42 N, 27.14 E) and computes 3x3 neighbourhood spatial spread.
Part of WP1.4 (Summer design days + SOLWEIG forcing).
"""
import argparse
import datetime
import json
from pathlib import Path
import ee
import numpy as np
import pandas as pd

PROJECT = "ee-geophilo"
IZMIR_LAT = 38.42
IZMIR_LON = 27.14
TIMEZONE = "Europe/Istanbul"  # UTC+3

OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "street" / "forcing_summer"

# Default design days selected under locked rule & master tie-break
DEFAULT_DESIGN_DAYS = {
    "IMBAT": "2022-07-05",
    "CALM": "2023-07-26"
}

def init_gee():
    print(f"Initializing Earth Engine with project '{PROJECT}'...")
    ee.Initialize(project=PROJECT)
    print("Earth Engine initialized successfully.")

def fetch_era5_data_for_day(date_str: str, day_type: str):
    print(f"\n=======================================================")
    print(f"Fetching GEE ERA5 & ERA5-Land for {day_type}: {date_str}")
    print(f"=======================================================")

    pt_center = ee.Geometry.Point([IZMIR_LON, IZMIR_LAT])

    # Date range for local 00:00 to 23:00 (UTC+3)
    # 00:00 local = 21:00 UTC previous day
    # 23:00 local = 20:00 UTC target day
    dt_target = pd.to_datetime(date_str)
    dt_prev = dt_target - pd.Timedelta(days=1)
    dt_next = dt_target + pd.Timedelta(days=1)

    t_start_utc = f"{dt_prev.strftime('%Y-%m-%d')}T20:00:00"
    t_end_utc = f"{dt_next.strftime('%Y-%m-%d')}T02:00:00"

    print(f"Query window (UTC): {t_start_utc} to {t_end_utc}")

    # 1. Query ECMWF/ERA5/HOURLY
    col_era5 = ee.ImageCollection("ECMWF/ERA5/HOURLY").filterDate(t_start_utc, t_end_utc)
    era5_bands = [
        "temperature_2m", "dewpoint_temperature_2m",
        "u_component_of_wind_10m", "v_component_of_wind_10m",
        "surface_pressure",
        "surface_solar_radiation_downwards",
        "total_sky_direct_solar_radiation_at_surface",
        "surface_thermal_radiation_downwards"
    ]

    # 3x3 neighbourhood coordinates for ERA5 (0.25 deg)
    era5_lats = [38.25, 38.50, 38.75]
    era5_lons = [27.00, 27.25, 27.50]
    era5_pts = [ee.Feature(ee.Geometry.Point([lon, lat]), {"lon": lon, "lat": lat, "pt_id": f"{lat}_{lon}"})
                for lat in era5_lats for lon in era5_lons]
    fc_era5 = ee.FeatureCollection(era5_pts)

    def extract_era5_img(img):
        t_ms = img.get("system:time_start")
        # Center point
        c_vals = img.select(era5_bands).reduceRegion(ee.Reducer.first(), pt_center, 5000)
        # 3x3 neighbourhood
        nb_vals = img.select(era5_bands).reduceRegions(
            collection=fc_era5,
            reducer=ee.Reducer.first(),
            scale=25000
        )
        return ee.Feature(None, {
            "time_start": t_ms,
            "center": c_vals,
            "neighbourhood": nb_vals.toList(10)
        })

    print("Extracting ERA5 HOURLY features...")
    era5_features = col_era5.map(extract_era5_img).getInfo()["features"]
    print(f"Retrieved {len(era5_features)} ERA5 hourly images.")

    # 2. Query ECMWF/ERA5_LAND/HOURLY
    col_land = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(t_start_utc, t_end_utc)
    land_bands = [
        "temperature_2m", "dewpoint_temperature_2m",
        "u_component_of_wind_10m", "v_component_of_wind_10m",
        "surface_pressure",
        "surface_solar_radiation_downwards_hourly",
        "surface_thermal_radiation_downwards_hourly"
    ]

    # 3x3 neighbourhood coordinates for ERA5-Land (0.1 deg)
    land_lats = [38.30, 38.40, 38.50]
    land_lons = [27.00, 27.10, 27.20]
    land_pts = [ee.Feature(ee.Geometry.Point([lon, lat]), {"lon": lon, "lat": lat, "pt_id": f"{lat}_{lon}"})
                for lat in land_lats for lon in land_lons]
    fc_land = ee.FeatureCollection(land_pts)

    # For center point in ERA5-Land: pasaport/gulf cell (38.4, 27.1) is water-masked,
    # so we use 10km buffer mean or nearest land cell to avoid NaN
    def extract_land_img(img):
        t_ms = img.get("system:time_start")
        c_vals = img.select(land_bands).reduceRegion(ee.Reducer.mean(), pt_center.buffer(8000), 1000)
        nb_vals = img.select(land_bands).reduceRegions(
            collection=fc_land,
            reducer=ee.Reducer.first(),
            scale=10000
        )
        return ee.Feature(None, {
            "time_start": t_ms,
            "center": c_vals,
            "neighbourhood": nb_vals.toList(10)
        })

    print("Extracting ERA5_LAND HOURLY features...")
    land_features = col_land.map(extract_land_img).getInfo()["features"]
    print(f"Retrieved {len(land_features)} ERA5-Land hourly images.")

    # 3. Parse and align to 24 local hours of target date
    # Local hours 0..23 of target date
    local_grid = pd.date_range(f"{date_str} 00:00:00", f"{date_str} 23:00:00", freq="1h", tz=TIMEZONE)

    era5_rows = []
    era5_nb_records = []
    for f in era5_features:
        t_ms = f["properties"]["time_start"]
        dt_utc = datetime.datetime.fromtimestamp(t_ms / 1000.0, datetime.timezone.utc)
        dt_local = dt_utc.astimezone(datetime.timezone(datetime.timedelta(hours=3)))
        if dt_local.date() == dt_target.date():
            c = f["properties"]["center"]
            u10 = c.get("u_component_of_wind_10m")
            v10 = c.get("v_component_of_wind_10m")
            wind_spd = float(np.sqrt(u10**2 + v10**2)) if u10 is not None and v10 is not None else np.nan
            wind_dir = float((np.degrees(np.arctan2(-u10, -v10))) % 360.0) if u10 is not None and v10 is not None else np.nan

            ghi_wm2 = float(c.get("surface_solar_radiation_downwards") / 3600.0) if c.get("surface_solar_radiation_downwards") is not None else 0.0
            dir_horiz_wm2 = float(c.get("total_sky_direct_solar_radiation_at_surface") / 3600.0) if c.get("total_sky_direct_solar_radiation_at_surface") is not None else 0.0
            diff_horiz_wm2 = max(0.0, ghi_wm2 - dir_horiz_wm2)
            ldown_wm2 = float(c.get("surface_thermal_radiation_downwards") / 3600.0) if c.get("surface_thermal_radiation_downwards") is not None else 0.0

            t2m_c = float(c.get("temperature_2m") - 273.15) if c.get("temperature_2m") is not None else np.nan
            d2m_c = float(c.get("dewpoint_temperature_2m") - 273.15) if c.get("dewpoint_temperature_2m") is not None else np.nan
            # RH from Ta and Td
            es = 6.112 * np.exp((17.67 * t2m_c) / (t2m_c + 243.5))
            ea = 6.112 * np.exp((17.67 * d2m_c) / (d2m_c + 243.5))
            rh_pct = float(np.clip((ea / es) * 100.0, 0.0, 100.0))
            sp_kpa = float(c.get("surface_pressure") / 1000.0) if c.get("surface_pressure") is not None else np.nan

            era5_rows.append({
                "local_time": dt_local.strftime("%Y-%m-%d %H:%M:%S"),
                "utc_time": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "local_hour": dt_local.hour,
                "t2m_c": round(t2m_c, 2),
                "d2m_c": round(d2m_c, 2),
                "rh_pct": round(rh_pct, 2),
                "press_kpa": round(sp_kpa, 2),
                "wind_spd_ms": round(wind_spd, 2),
                "wind_dir_deg": round(wind_dir, 1),
                "u10_ms": round(float(u10), 2) if u10 is not None else None,
                "v10_ms": round(float(v10), 2) if v10 is not None else None,
                "ghi_wm2": round(ghi_wm2, 2),
                "dir_horiz_wm2": round(dir_horiz_wm2, 2),
                "diff_horiz_wm2": round(diff_horiz_wm2, 2),
                "ldown_wm2": round(ldown_wm2, 2)
            })

            # Process neighbourhood spread
            nb = f["properties"]["neighbourhood"]
            for nb_feat in nb:
                props = nb_feat["properties"]
                pt_id = props.get("pt_id")
                t_val = props.get("temperature_2m")
                s_val = props.get("surface_solar_radiation_downwards")
                l_val = props.get("surface_thermal_radiation_downwards")
                era5_nb_records.append({
                    "date": date_str,
                    "local_hour": dt_local.hour,
                    "product": "ERA5",
                    "pt_id": pt_id,
                    "t2m_c": t_val - 273.15 if t_val is not None else None,
                    "ghi_wm2": s_val / 3600.0 if s_val is not None else None,
                    "ldown_wm2": l_val / 3600.0 if l_val is not None else None
                })

    era5_df = pd.DataFrame(era5_rows).sort_values(by="local_hour").reset_index(drop=True)

    # Process ERA5-Land rows
    land_rows = []
    land_nb_records = []
    for f in land_features:
        t_ms = f["properties"]["time_start"]
        dt_utc = datetime.datetime.fromtimestamp(t_ms / 1000.0, datetime.timezone.utc)
        dt_local = dt_utc.astimezone(datetime.timezone(datetime.timedelta(hours=3)))
        if dt_local.date() == dt_target.date():
            c = f["properties"]["center"]
            u10 = c.get("u_component_of_wind_10m")
            v10 = c.get("v_component_of_wind_10m")
            wind_spd = float(np.sqrt(u10**2 + v10**2)) if u10 is not None and v10 is not None else np.nan
            wind_dir = float((np.degrees(np.arctan2(-u10, -v10))) % 360.0) if u10 is not None and v10 is not None else np.nan

            ghi_wm2 = float(c.get("surface_solar_radiation_downwards_hourly") / 3600.0) if c.get("surface_solar_radiation_downwards_hourly") is not None else 0.0
            ldown_wm2 = float(c.get("surface_thermal_radiation_downwards_hourly") / 3600.0) if c.get("surface_thermal_radiation_downwards_hourly") is not None else 0.0

            t2m_c = float(c.get("temperature_2m") - 273.15) if c.get("temperature_2m") is not None else np.nan
            d2m_c = float(c.get("dewpoint_temperature_2m") - 273.15) if c.get("dewpoint_temperature_2m") is not None else np.nan
            es = 6.112 * np.exp((17.67 * t2m_c) / (t2m_c + 243.5))
            ea = 6.112 * np.exp((17.67 * d2m_c) / (d2m_c + 243.5))
            rh_pct = float(np.clip((ea / es) * 100.0, 0.0, 100.0))
            sp_kpa = float(c.get("surface_pressure") / 1000.0) if c.get("surface_pressure") is not None else np.nan

            land_rows.append({
                "local_time": dt_local.strftime("%Y-%m-%d %H:%M:%S"),
                "utc_time": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "local_hour": dt_local.hour,
                "t2m_c": round(t2m_c, 2),
                "d2m_c": round(d2m_c, 2),
                "rh_pct": round(rh_pct, 2),
                "press_kpa": round(sp_kpa, 2),
                "wind_spd_ms": round(wind_spd, 2),
                "wind_dir_deg": round(wind_dir, 1),
                "u10_ms": round(float(u10), 2) if u10 is not None else None,
                "v10_ms": round(float(v10), 2) if v10 is not None else None,
                "ghi_wm2": round(ghi_wm2, 2),
                "ldown_wm2": round(ldown_wm2, 2)
            })

            nb = f["properties"]["neighbourhood"]
            for nb_feat in nb:
                props = nb_feat["properties"]
                pt_id = props.get("pt_id")
                t_val = props.get("temperature_2m")
                s_val = props.get("surface_solar_radiation_downwards_hourly")
                l_val = props.get("surface_thermal_radiation_downwards_hourly")
                land_nb_records.append({
                    "date": date_str,
                    "local_hour": dt_local.hour,
                    "product": "ERA5_LAND",
                    "pt_id": pt_id,
                    "t2m_c": t_val - 273.15 if t_val is not None else None,
                    "ghi_wm2": s_val / 3600.0 if s_val is not None else None,
                    "ldown_wm2": l_val / 3600.0 if l_val is not None else None
                })

    land_df = pd.DataFrame(land_rows).sort_values(by="local_hour").reset_index(drop=True)

    # Save to CSV
    era5_out_path = OUT_DIR / f"era5_hourly_{day_type}_{date_str}.csv"
    land_out_path = OUT_DIR / f"era5_land_hourly_{day_type}_{date_str}.csv"
    era5_df.to_csv(era5_out_path, index=False)
    land_df.to_csv(land_out_path, index=False)
    print(f"Saved ERA5 ({len(era5_df)} hours) to {era5_out_path}")
    print(f"Saved ERA5-Land ({len(land_df)} hours) to {land_out_path}")

    return {
        "date": date_str,
        "day_type": day_type,
        "era5_df": era5_df,
        "land_df": land_df,
        "era5_nb": pd.DataFrame(era5_nb_records),
        "land_nb": pd.DataFrame(land_nb_records)
    }

def main():
    parser = argparse.ArgumentParser(description="Fetch ERA5 and ERA5-Land hourly data via Google Earth Engine.")
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    parser.add_argument("--day-type", type=str, choices=["IMBAT", "CALM"], help="Design day type (IMBAT or CALM)")
    parser.add_argument("--all", action="store_true", help="Fetch both default design days")
    args = parser.parse_args()

    init_gee()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.date and args.day_type:
        target_days = {args.day_type: args.date}
    else:
        target_days = DEFAULT_DESIGN_DAYS

    print(f"Target design days to fetch: {target_days}")

    spread_json_path = OUT_DIR / "era5_neighbourhood_spread.json"
    if spread_json_path.exists():
        try:
            with open(spread_json_path, "r") as f:
                all_spreads = json.load(f)
        except Exception:
            all_spreads = {}
    else:
        all_spreads = {}

    for day_type, date_str in target_days.items():
        res = fetch_era5_data_for_day(date_str, day_type)

        # Compute 3x3 neighbourhood spread statistics at 14:00 local (peak heat/solar)
        era5_nb = res["era5_nb"]
        land_nb = res["land_nb"]

        era5_14 = era5_nb[era5_nb["local_hour"] == 14].dropna(subset=["t2m_c"])
        land_14 = land_nb[land_nb["local_hour"] == 14].dropna(subset=["t2m_c"])

        spread_summary = {
            "date": date_str,
            "day_type": day_type,
            "era5_14h_center_t2m_c": float(res["era5_df"].loc[res["era5_df"]["local_hour"] == 14, "t2m_c"].values[0]),
            "era5_14h_center_ghi_wm2": float(res["era5_df"].loc[res["era5_df"]["local_hour"] == 14, "ghi_wm2"].values[0]),
            "era5_14h_center_dir_horiz_wm2": float(res["era5_df"].loc[res["era5_df"]["local_hour"] == 14, "dir_horiz_wm2"].values[0]),
            "era5_14h_center_diff_horiz_wm2": float(res["era5_df"].loc[res["era5_df"]["local_hour"] == 14, "diff_horiz_wm2"].values[0]),
            "era5_14h_center_ldown_wm2": float(res["era5_df"].loc[res["era5_df"]["local_hour"] == 14, "ldown_wm2"].values[0]),
            "era5_3x3_t2m_mean": round(float(era5_14["t2m_c"].mean()), 2),
            "era5_3x3_t2m_std": round(float(era5_14["t2m_c"].std()), 2),
            "era5_3x3_t2m_min": round(float(era5_14["t2m_c"].min()), 2),
            "era5_3x3_t2m_max": round(float(era5_14["t2m_c"].max()), 2),
            "era5_3x3_ghi_mean": round(float(era5_14["ghi_wm2"].mean()), 2),
            "era5_3x3_ghi_std": round(float(era5_14["ghi_wm2"].std()), 2),
            "era5_land_14h_center_t2m_c": float(res["land_df"].loc[res["land_df"]["local_hour"] == 14, "t2m_c"].values[0]),
            "era5_land_3x3_t2m_mean": round(float(land_14["t2m_c"].mean()), 2),
            "era5_land_3x3_t2m_std": round(float(land_14["t2m_c"].std()), 2),
            "era5_land_3x3_t2m_min": round(float(land_14["t2m_c"].min()), 2),
            "era5_land_3x3_t2m_max": round(float(land_14["t2m_c"].max()), 2),
        }
        all_spreads[day_type] = spread_summary

        print(f"\n--- 3x3 Neighbourhood Spread at 14:00 Local ({day_type} {date_str}) ---")
        print(f"ERA5 center Ta: {spread_summary['era5_14h_center_t2m_c']} °C | 3x3 mean: {spread_summary['era5_3x3_t2m_mean']} ± {spread_summary['era5_3x3_t2m_std']} °C (range [{spread_summary['era5_3x3_t2m_min']}, {spread_summary['era5_3x3_t2m_max']}])")
        print(f"ERA5-Land center Ta: {spread_summary['era5_land_14h_center_t2m_c']} °C | 3x3 land mean: {spread_summary['era5_land_3x3_t2m_mean']} ± {spread_summary['era5_land_3x3_t2m_std']} °C (range [{spread_summary['era5_land_3x3_t2m_min']}, {spread_summary['era5_land_3x3_t2m_max']}])")
        print(f"ERA5 solar GHI: {spread_summary['era5_14h_center_ghi_wm2']} W/m2 (Direct horiz: {spread_summary['era5_14h_center_dir_horiz_wm2']}, Diffuse: {spread_summary['era5_14h_center_diff_horiz_wm2']}) | 3x3 GHI std: {spread_summary['era5_3x3_ghi_std']} W/m2")

    with open(spread_json_path, "w") as f:
        json.dump(all_spreads, f, indent=2)
    print(f"\nSaved neighbourhood spread summary to {spread_json_path}")

if __name__ == "__main__":
    main()
