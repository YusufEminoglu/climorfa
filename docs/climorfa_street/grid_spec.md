# CLIMORFA v2 street-scale tiling grid (authoritative, 2026-10-29)

All street-scale rasters (building DSM, DEM, tree DSM, land cover, SOLWEIG outputs) use this grid. Do not invent another.

- CRS: EPSG:5253 (TUREF / TM27). Units m.
- Origin: snap to the 5 m DSM grid (data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif): tile corners at integer
  multiples of 1000 m in x and y (the DSM origin is a multiple of 5 m; 1000 m multiples are therefore aligned).
- Tiles: 1000 m × 1000 m inner extent + 100 m buffer on every side (1200 m × 1200 m written).
  Tile id `T_{x0/1000:04d}_{y0/1000:04d}` from the inner lower-left corner, e.g. T_0512_4255.
- Resolutions: 2 m primary (600 × 600 px incl. buffer); 1 m deep-dive (1200 × 1200 px incl. buffer).
  Both fit the T1000 8 GB comfortably (WP0.4: 1200 × 1200 px ≈ 2.6 GB).
- Urban-core mask: 250 m grid cells with `building_coverage_exact > 0.05` (data/03_processed/grid_250m_model_features_v8.csv
  joined to data/03_processed/analysis_grids.gpkg; WP0.4 found 3,647 cells = 227.94 km²). A tile is in scope if its inner
  extent intersects the mask. Write the tile index as data/street/grid/tiles_core.gpkg (inner polygons + id + core share).
- Nodata: -9999 (float32). Sea cells: DEM = 0, building/tree = 0, flagged in a separate land mask.
- Writing: GeoTIFF, float32, DEFLATE + predictor 3, tiled 256 × 256; one file per tile per layer:
  data/street/tiles/<res>m/<layer>/<tile_id>.tif with layers dem, building_dsm, tree_dsm, landcover, landmask;
  plus a VRT per layer and resolution.
