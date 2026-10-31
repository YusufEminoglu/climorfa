"""
WP1.3 — Step 1: Download Meta / WRI Canopy Height Maps for Izmir FUA
Bucket: s3://dataforgood-fb-data/forests/v1/ (public, no credentials needed)
Target: data/street/wsl/meta_chm/raw/
"""

import os
import sys
import time
import rasterio
from rasterio.warp import transform_bounds
import mercantile
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def check_ram(min_free_gb=3.0):
    with open('/proc/meminfo') as f:
        meminfo = {line.split(':')[0]: int(line.split(':')[1].split()[0]) for line in f}
    avail_gb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0)) / (1024 * 1024)
    if avail_gb < min_free_gb:
        raise RuntimeError(f"Available RAM ({avail_gb:.2f} GB) is below minimum threshold ({min_free_gb} GB).")
    return avail_gb

def main():
    t0 = time.time()
    print("=== Step 1: Query & Download Meta / WRI CHM Tiles for Izmir FUA ===")
    check_ram()

    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    raw_dir = "data/street/wsl/meta_chm/raw"
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Determine FUA bounding box in EPSG:4326
    print(f"Reading DSM extent from {dsm_path}...")
    with rasterio.open(dsm_path) as src:
        crs_dsm = src.crs
        bounds_dsm = src.bounds
        bounds_4326 = transform_bounds(crs_dsm, "EPSG:4326", *bounds_dsm)

    west, south, east, north = bounds_4326
    print(f"FUA Bounds (EPSG:5253): {bounds_dsm}")
    print(f"FUA Bounds (EPSG:4326): lon [{west:.6f}, {east:.6f}], lat [{south:.6f}, {north:.6f}]")

    # 2. Determine intersecting zoom 9 tiles
    tiles = list(mercantile.tiles(west, south, east, north, zooms=9))
    print(f"Found {len(tiles)} intersecting tile(s) at zoom level 9:")
    quadkeys = []
    for t in tiles:
        qk = mercantile.quadkey(t)
        quadkeys.append(qk)
        tb = mercantile.bounds(t)
        print(f"  Tile {t}: QuadKey = {qk}, Bounds = [{tb.west:.4f}, {tb.south:.4f}, {tb.east:.4f}, {tb.north:.4f}]")

    # 3. Setup anonymous S3 client
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'dataforgood-fb-data'

    # Download readme.txt if not present
    readme_path = "data/street/wsl/meta_chm/readme.txt"
    if not os.path.exists(readme_path):
        print("Fetching bucket readme.txt...")
        try:
            s3.download_file(bucket, 'forests/v1/readme.txt', readme_path)
            print(f"Saved {readme_path}")
        except Exception as e:
            print(f"Could not fetch readme: {e}")

    # 4. Download CHM rasters, masks, and metadata geojsons
    download_files = []
    for qk in quadkeys:
        download_files.append((f"forests/v1/alsgedi_global_v6_float/chm/{qk}.tif", f"{raw_dir}/{qk}_chm.tif"))
        download_files.append((f"forests/v1/alsgedi_global_v6_float/msk/{qk}.tif.msk", f"{raw_dir}/{qk}_chm.tif.msk"))
        download_files.append((f"forests/v1/alsgedi_global_v6_float/metadata/{qk}.geojson", f"{raw_dir}/{qk}.geojson"))

    total_download_bytes = 0
    print("\n--- Starting Downloads ---")
    for s3_key, local_path in download_files:
        check_ram()
        try:
            head = s3.head_object(Bucket=bucket, Key=s3_key)
            expected_size = head['ContentLength']
        except Exception as e:
            print(f"Error accessing s3://{bucket}/{s3_key}: {e}")
            continue

        if os.path.exists(local_path) and os.path.getsize(local_path) == expected_size:
            print(f"Already exists (complete): {os.path.basename(local_path)} ({expected_size / 1e6:.2f} MB)")
            total_download_bytes += expected_size
            continue

        print(f"Downloading s3://{bucket}/{s3_key} -> {local_path} ({expected_size / 1e6:.2f} MB)...")
        t_file0 = time.time()
        s3.download_file(bucket, s3_key, local_path)
        actual_size = os.path.getsize(local_path)
        dt = time.time() - t_file0
        speed = (actual_size / 1e6) / dt if dt > 0 else 0
        print(f"  Done in {dt:.1f} s ({speed:.2f} MB/s). Verified size: {actual_size:,} bytes")
        total_download_bytes += actual_size

    print(f"\nAll downloads finished in {time.time()-t0:.1f} s.")
    print(f"Total size downloaded/verified: {total_download_bytes / 1e6:.2f} MB")

if __name__ == "__main__":
    main()
