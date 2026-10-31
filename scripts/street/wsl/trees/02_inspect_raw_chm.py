"""
WP1.3: Inspect Meta CHM raw files and extract acquisition date range
"""

import os
import json
import rasterio
import glob

def main():
    raw_dir = "data/street/wsl/meta_chm/raw"
    tif_files = sorted(glob.glob(f"{raw_dir}/*_chm.tif"))
    print("=== Raw Meta CHM Rasters ===")
    for f in tif_files:
        with rasterio.open(f) as src:
            print(f"File: {os.path.basename(f)}")
            print(f"  Shape: {src.shape}, Bands: {src.count}, Dtype: {src.dtypes[0]}")
            print(f"  CRS: {src.crs}, Res: {src.res}")
            print(f"  Bounds: {src.bounds}")
            print(f"  Nodata: {src.nodata}")

    geojson_files = sorted(glob.glob(f"{raw_dir}/*.geojson"))
    print("\n=== Imagery Metadata & Acquisition Dates ===")
    all_dates = []
    for f in geojson_files:
        print(f"Inspecting {os.path.basename(f)}...")
        with open(f) as fp:
            data = json.load(fp)
        features = data.get('features', [])
        print(f"  Total feature polygons: {len(features):,}")
        sample_props = features[0]['properties'] if features else {}
        print(f"  Sample properties: {sample_props}")
        
        # Extract dates
        dates = []
        for feat in features:
            props = feat.get('properties', {})
            # Check common keys for date: date, observation_date, acquisition_date, start_date, etc.
            for k, v in props.items():
                if 'date' in k.lower() or 'time' in k.lower():
                    if v:
                        dates.append(str(v))
        if dates:
            dates = sorted(dates)
            print(f"  Date range for {os.path.basename(f)}: min={dates[0]}, max={dates[-1]}")
            all_dates.extend(dates)
        else:
            print(f"  No explicit date keys found in properties: {list(sample_props.keys())}")

    if all_dates:
        all_dates = sorted(all_dates)
        print(f"\nOVERALL ACQUISITION DATE RANGE FOR IZMIR: {all_dates[0]} to {all_dates[-1]}")

if __name__ == "__main__":
    main()
