# CLIMORFA v2 — Analysis data contract (authoritative; all WP3 workers read and write exactly this)

Definitions come from docs/climorfa_street/pre_analysis_lock.md (PS, segment, LCZ, metrics M1–M5). CRS EPSG:5253.
Tiles and grid per docs/climorfa_street/grid_spec.md. Analysis rasters use the SOLWEIG output grid: the INNER 1000 m of
each tile, 500 × 500 px at 2 m (same transform as data/street/solweig/<DAY>/tmrt/<tile>.tif).

## A. Produced by WP3.0 (street units; desktop)
1. `data/street/analysis/ps_index/<tile>.tif` — int32, 500 × 500, value = seg_id of the nearest segment for PS cells,
   0 = not PS. A seg_id is globally unique across tiles (segments are split at tile borders).
2. `data/street/analysis/segments.gpkg`, layer `segments`, one row per segment:
   seg_id (int64), tile_id, district, lcz_primary (int), lcz_weak250 (int), length_m, n_ps, azimuth_deg (0–180,
   undirected street axis), hw_ratio, bh_mean30 (m), bcover30 (0–1), tcover15 (0–1), tht15 (m, mean canopy height of
   tree cells), dist_coast_m, elev_m, slope_deg, lst_med (°C), g050_id, g100_id, g250_id, g500_id, geometry (LineString).
   NaN allowed with a documented reason.
3. `data/street/analysis/seg_metrics_<DAY>.parquet` (DAY ∈ IMBAT, CALM), one row per segment present in finished tiles:
   seg_id, n_ps_valid, utci_b06 … utci_b21 (segment medians of UTCI_street per band), utci_unif_b15, tmrt_b15 (median),
   tmrt_b15_p90, v10eff_b15 (median clip(U10 × ratio)), ratio_med_b15, svf_med, shade_frac_10_18, m2_strong (count of
   bands 10–18 with utci ≥ 32), m2_vstrong (≥ 38), partial_run (bool), n_tiles_at_extraction (int).
4. `data/street/analysis/WP3.0_tiles_done.csv` (tile_id, day, extracted_at).

## B. Produced by WP3.0c (grid ladder and context; laptop)
`data/street/analysis/grid_<L>.gpkg` for L ∈ {050, 100, 250, 500}, layer `grid`:
g<L>_id, tile_id, lcz_primary_major (majority class over PS cells; and share), lcz_weak250, lst_med, pop (people, from
data/01_raw/population/mahalle_population_controls_2024.gpkg — areal weighting by built-up area; method documented),
reach (network-reach centrality, field chosen and named in the report, from data/03_processed/grid_250m_road_network_reach_features.csv
for L = 250; for the other levels: area-weighted from 250 m), district, geometry.
Grid cell ids: g<L>_id = f"{x0//L}_{y0//L}" from the lower-left corner in EPSG:5253 (integers), identical across workers.

## C. Conventions
- Bands: data/street/solweig/band_hour_table.csv (band "15" = 14:00–15:00 local, hour-ending).
- Nodata float -9999; never write NaN into GeoTIFFs.
- Every table produced from a partial run carries partial_run = True and n_tiles_at_extraction; any number reported from
  such a table is labelled "PARTIAL (n tiles)".
- Synthetic test tables must carry is_synthetic = True and SYNTHETIC in the filename (rule 6).
