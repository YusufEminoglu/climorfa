"""
WP1.1 DTM Validation Across 8 Windows (3 Required + 5 Diverse LCZ Types)
Author: Yusuf Eminoglu / Antigravity
Date: 2026-10-29

Specifications:
- 3 Required windows:
  1. Alsancak (512350, 4255850, 512850, 4256350)
  2. Bornova (517500, 4257500, 519000, 4259000)
  3. Karabaglar (508500, 4248000, 510000, 4249500)
- 5 Diverse LCZ windows:
  4. LCZ 8 Industrial (Çiğli AOSB: 502375, 4262625, 503375, 4263625)
  5. LCZ 6 Open Low-Rise (Gaziemir: 512375, 4239875, 513375, 4240875)
  6. LCZ 3 Compact Low-Rise (Buca: 514375, 4249125, 515375, 4250125)
  7. LCZ 2 Compact Mid-Rise (Konak: 511125, 4253625, 512125, 4254625)
  8. LCZ 9 Sparsely Built (Güzelbahçe: 486625, 4240875, 487625, 4241875)
"""

import os
import sys
import time
import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from scipy.ndimage import sobel
from shapely.ops import unary_union
import pyogrio

def calc_slope_deg(dem, cellsize=5.0):
    dx = sobel(dem, axis=1) / (8.0 * cellsize)
    dy = sobel(dem, axis=0) / (8.0 * cellsize)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

def main():
    t_start = time.time()
    print("=== Step 3: DTM Validation Across 8 Izmir Windows ===")
    
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    dtm_path = "data/street/terrain/dtm_5m_epsg5253.tif"
    ndsm_path = "data/street/terrain/ndsm_5m_epsg5253.tif"
    gpkg_path = "data/02_interim/akilli_sehir_fua.gpkg"
    
    windows = {
        "Alsancak (Dense core)": (512350, 4255850, 512850, 4256350, "LCZ 1/2"),
        "Bornova (Mid-rise)": (517500, 4257500, 519000, 4259000, "LCZ 2/5"),
        "Karabaglar (Hillside)": (508500, 4248000, 510000, 4249500, "LCZ 3"),
        "Cigli AOSB (Industrial)": (502375, 4262625, 503375, 4263625, "LCZ 8"),
        "Gaziemir (Open low-rise)": (512375, 4239875, 513375, 4240875, "LCZ 6"),
        "Buca (Compact low-rise)": (514375, 4249125, 515375, 4250125, "LCZ 3"),
        "Konak (Compact mid-rise)": (511125, 4253625, 512125, 4254625, "LCZ 2"),
        "Guzelbahce (Sparse built)": (486625, 4240875, 487625, 4241875, "LCZ 9"),
    }
    
    results = []
    
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(dtm_path) as src_dtm, rasterio.open(ndsm_path) as src_ndsm:
        for name, (minx, miny, maxx, maxy, lcz_tag) in windows.items():
            bbox = (minx, miny, maxx, maxy)
            win = from_bounds(*bbox, transform=src_dsm.transform).round_offsets().round_lengths()
            win_trans = rasterio.windows.transform(win, src_dsm.transform)
            
            dsm_crop = src_dsm.read(1, window=win).astype(np.float32)
            dtm_crop = src_dtm.read(1, window=win).astype(np.float32)
            ndsm_crop = src_ndsm.read(1, window=win).astype(np.float32)
            nodata_val = src_dsm.nodata
            valid = (dsm_crop != nodata_val) & (dtm_crop != nodata_val)
            
            # Buildings
            bld = pyogrio.read_dataframe(gpkg_path, layer="buildings_fua", bbox=bbox)
            n_bld = len(bld)
            
            # Inside vs ring on DSM and DTM
            dsm_per_bld_diff = []
            dtm_per_bld_diff = []
            
            # Also pooled masks
            valid_bld_geoms = [g for g in bld.geometry if g is not None and not g.is_empty]
            if len(valid_bld_geoms) > 0:
                all_bld_union = unary_union(valid_bld_geoms)
                shrunk_union = unary_union([g.buffer(-2.5) for g in valid_bld_geoms if not g.buffer(-2.5).is_empty])
                ring_union = all_bld_union.buffer(7.5).difference(all_bld_union)
                
                shrunk_mask = features.rasterize([(shrunk_union, 1)], out_shape=dsm_crop.shape, transform=win_trans, dtype=np.uint8)
                ring_mask = features.rasterize([(ring_union, 1)], out_shape=dsm_crop.shape, transform=win_trans, dtype=np.uint8)
                
                dsm_shrunk_med = np.median(dsm_crop[(shrunk_mask == 1) & valid]) if np.any((shrunk_mask == 1) & valid) else np.nan
                dsm_ring_med = np.median(dsm_crop[(ring_mask == 1) & valid]) if np.any((ring_mask == 1) & valid) else np.nan
                dsm_pooled_diff = dsm_shrunk_med - dsm_ring_med
                
                dtm_shrunk_med = np.median(dtm_crop[(shrunk_mask == 1) & valid]) if np.any((shrunk_mask == 1) & valid) else np.nan
                dtm_ring_med = np.median(dtm_crop[(ring_mask == 1) & valid]) if np.any((ring_mask == 1) & valid) else np.nan
                dtm_pooled_diff = dtm_shrunk_med - dtm_ring_med
            else:
                dsm_pooled_diff = np.nan
                dtm_pooled_diff = np.nan
                
            for g in valid_bld_geoms:
                sg = g.buffer(-2.5)
                rg = g.buffer(7.5).difference(g)
                if sg.is_empty or rg.is_empty:
                    continue
                ms = features.rasterize([(sg, 1)], out_shape=dsm_crop.shape, transform=win_trans, dtype=np.uint8)
                mr = features.rasterize([(rg, 1)], out_shape=dsm_crop.shape, transform=win_trans, dtype=np.uint8)
                ds, dr = dsm_crop[(ms == 1) & valid], dsm_crop[(mr == 1) & valid]
                ts, tr = dtm_crop[(ms == 1) & valid], dtm_crop[(mr == 1) & valid]
                if len(ds) > 0 and len(dr) > 0:
                    dsm_per_bld_diff.append(np.median(ds) - np.median(dr))
                if len(ts) > 0 and len(tr) > 0:
                    dtm_per_bld_diff.append(np.median(ts) - np.median(tr))
                    
            dsm_per_bld_med = np.median(dsm_per_bld_diff) if len(dsm_per_bld_diff) > 0 else np.nan
            dtm_per_bld_med = np.median(dtm_per_bld_diff) if len(dtm_per_bld_diff) > 0 else np.nan
            
            # Roads residual: DSM - DTM
            roads = pyogrio.read_dataframe(gpkg_path, layer="roads_fua", bbox=bbox)
            valid_roads = [g for g in roads.geometry if g is not None and not g.is_empty]
            if len(valid_roads) > 0:
                road_mask = features.rasterize(((g, 1) for g in valid_roads), out_shape=dsm_crop.shape, transform=win_trans, dtype=np.uint8)
                road_valid = (road_mask == 1) & valid
                road_resids = np.abs(dsm_crop[road_valid] - dtm_crop[road_valid])
                road_resid_med = np.median(road_resids) if len(road_resids) > 0 else np.nan
                road_resid_p95 = np.percentile(road_resids, 95) if len(road_resids) > 0 else np.nan
                road_resid_mean = np.mean(road_resids) if len(road_resids) > 0 else np.nan
            else:
                road_resid_med = np.nan
                road_resid_p95 = np.nan
                road_resid_mean = np.nan
                
            # Slopes
            s_dsm = calc_slope_deg(dsm_crop)[2:-2, 2:-2]
            s_dtm = calc_slope_deg(dtm_crop)[2:-2, 2:-2]
            v_s = valid[2:-2, 2:-2]
            
            dsm_med_slope = np.median(s_dsm[v_s])
            dtm_med_slope = np.median(s_dtm[v_s])
            dtm_p99_slope = np.percentile(s_dtm[v_s], 99)
            dtm_pct_gt_60 = np.mean(s_dtm[v_s] > 60.0) * 100.0
            
            # Capping compliance
            cap_ok = np.all(dtm_crop[valid] <= dsm_crop[valid] + 1e-4)
            cap_share = np.mean(dtm_crop[valid] <= dsm_crop[valid] + 1e-4) * 100.0
            
            res = {
                "Window": name,
                "LCZ": lcz_tag,
                "Buildings": n_bld,
                "DSM_pooled_diff_m": dsm_pooled_diff,
                "DSM_bld_diff_m": dsm_per_bld_med,
                "DTM_bld_diff_m": dtm_per_bld_med,
                "DTM_pooled_diff_m": dtm_pooled_diff,
                "Road_resid_med_m": road_resid_med,
                "Road_resid_mean_m": road_resid_mean,
                "Road_resid_p95_m": road_resid_p95,
                "DSM_slope_med_deg": dsm_med_slope,
                "DTM_slope_med_deg": dtm_med_slope,
                "DTM_slope_p99_deg": dtm_p99_slope,
                "DTM_pct_gt_60": dtm_pct_gt_60,
                "Cap_compliance_pct": cap_share
            }
            results.append(res)
            
            print(f"[{name}] {lcz_tag}: bldgs={n_bld}")
            print(f"   DSM diff: pooled={dsm_pooled_diff:+.2f} m, per-bld={dsm_per_bld_med:+.2f} m")
            print(f"   DTM diff: per-bld={dtm_per_bld_med:+.2f} m (Acceptance <= 1.5m: {abs(dtm_per_bld_med) <= 1.5})")
            print(f"   Road residual: med={road_resid_med:.2f} m, p95={road_resid_p95:.2f} m")
            print(f"   Slope: DSM med={dsm_med_slope:.1f} deg -> DTM med={dtm_med_slope:.1f} deg, % > 60 deg={dtm_pct_gt_60:.2f}%")
            print(f"   DTM <= DSM compliance: {cap_share:.2f}%\n")
            
    # Print Markdown Table
    print("\n" + "="*80)
    print("### DTM VALIDATION TABLE ACROSS 8 EVALUATION WINDOWS")
    print("="*80)
    headers = [
        "Window", "LCZ", "Bldgs", "DSM Bld Diff (m)", "DTM Bld Diff (m)",
        "|Diff| <= 1.5m?", "Road Resid Med (m)", "DTM Med Slope", "DTM % > 60 deg", "DTM <= DSM"
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"]*len(headers)) + "|")
    for r in results:
        pass_crit = "**PASS**" if abs(r['DTM_bld_diff_m']) <= 1.5 else "**FAIL**"
        line = [
            r['Window'],
            r['LCZ'],
            f"{r['Buildings']:,}",
            f"{r['DSM_bld_diff_m']:+.2f} m",
            f"{r['DTM_bld_diff_m']:+.2f} m",
            pass_crit,
            f"{r['Road_resid_med_m']:.2f} m",
            f"{r['DTM_slope_med_deg']:.1f}°",
            f"{r['DTM_pct_gt_60']:.2f}%",
            f"{r['Cap_compliance_pct']:.1f}%"
        ]
        print("| " + " | ".join(line) + " |")
        
    print(f"\nStep 3 validation completed in {time.time()-t_start:.2f} s")

if __name__ == "__main__":
    main()
