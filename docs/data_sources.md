# Data sources

This repository does **not** redistribute third-party geospatial data. Each
dataset must be obtained from its original provider.

| Dataset | Provider | Access method | License | Validation |
|---|---|---|---|---|
| Surface model (5 m) | Local administration | Local administration | Provider terms | Cross-checked against independent municipal floor-count fields |
| OpenStreetMap | OSM Foundation | [Overpass API](https://overpass-api.de/) / [Geofabrik](https://download.geofabrik.de/) | ODbL | Validated against the official IMM park inventory: 189 official sites, 180 matched across 112 FUR neighbourhoods vs. 644 OSM polygons in 12 districts; log-count Pearson *r* = 0.93 (leave-one-district-out range 0.88-0.95), log-area *r* = 0.91 (range 0.81-0.93). Rank correlations are lower (0.75/0.78) and only 10.6% of matched sites had a strong OSM name candidate, so this validates district-level supply geography, not object-level completeness or public access. |
| Landsat 8/9 LST | USGS / NASA | [Google Earth Engine](https://earthengine.google.com/) `LANDSAT/LC08/C02/T1_L2` | Public domain | Used as the climate-response variable, not itself cross-validated against another source in this pipeline |
| Sentinel-2 | ESA | [Google Earth Engine](https://earthengine.google.com/) `COPERNICUS/S2_SR_HARMONIZED` | ESA terms | Indirect: contributes to the primary model, which is checked against the 569-cell manual field audit |
| Dynamic World | Google / WRI | [Google Earth Engine](https://earthengine.google.com/) `GOOGLE/DYNAMICWORLD/V1` | CC BY 4.0 | Indirect: contributes to the primary model, which is checked against the 569-cell manual field audit |
| ESA WorldCover 2021 | ESA | [esa-worldcover.org](https://esa-worldcover.org/) | CC BY 4.0 | Indirect: contributes to the primary model, which is checked against the 569-cell manual field audit |
| ETH Global Canopy Height | ETH Zurich | [Google Earth Engine](https://earthengine.google.com/) `users/nlang/ETH_GlobalCanopyHeight_10m_2020` | CC BY 4.0 | Indirect: contributes to the primary model, which is checked against the 569-cell manual field audit |
| WUDAPT LCZ | WUDAPT | [wudapt.org](https://www.wudapt.org/) | Provider terms | Direct: the weak label itself is the subject of the 569-cell manual field audit (primary-tier exact agreement 8.3%) |
| Building footprints | Local administration | Local administration | Provider terms | Cross-checked against independent municipal floor-count fields |
| Izmir FUA boundary | Derived from administrative data | Local administration | Provider terms | Matches the 18-district Izmir Metropolitan Municipality administrative extent |
| Population (2024) | TURKSTAT | Local administration | Provider terms | — |

## Earth Engine setup

1. Register a [Google Cloud project](https://console.cloud.google.com/)
   with the Earth Engine API enabled.
2. Authenticate: `earthengine authenticate`
3. Copy `.env.example` to `.env` and set `EE_PROJECT`.

## Data pipeline order

The `src/data_prep/` scripts are numbered `v0` through `v8` in execution
order. Intermediate outputs land in `data/02_interim/` and `data/03_processed/`;
final feature tables are in `data/03_processed/grid_250m_model_features_v8.csv`.
