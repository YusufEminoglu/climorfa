# Data sources

This repository does **not** redistribute third-party geospatial data. Each
dataset must be obtained from its original provider.

| Dataset | Provider | Access method | License |
|---|---|---|---|
| Surface model (5 m) | Local administration | Local administration | Provider terms |
| OpenStreetMap | OSM Foundation | [Overpass API](https://overpass-api.de/) / [Geofabrik](https://download.geofabrik.de/) | ODbL |
| Landsat 8/9 LST | USGS / NASA | [Google Earth Engine](https://earthengine.google.com/) `LANDSAT/LC08/C02/T1_L2` | Public domain |
| Sentinel-2 | ESA | [Google Earth Engine](https://earthengine.google.com/) `COPERNICUS/S2_SR_HARMONIZED` | ESA terms |
| Dynamic World | Google / WRI | [Google Earth Engine](https://earthengine.google.com/) `GOOGLE/DYNAMICWORLD/V1` | CC BY 4.0 |
| ESA WorldCover 2021 | ESA | [esa-worldcover.org](https://esa-worldcover.org/) | CC BY 4.0 |
| ETH Global Canopy Height | ETH Zurich | [Google Earth Engine](https://earthengine.google.com/) `users/nlang/ETH_GlobalCanopyHeight_10m_2020` | CC BY 4.0 |
| WUDAPT LCZ | WUDAPT | [wudapt.org](https://www.wudapt.org/) | Provider terms |
| Building footprints | Local administration | Local administration | Provider terms |
| Izmir FUA boundary | Derived from administrative data | Local administration | Provider terms |
| Population (2024) | TURKSTAT | Local administration | Provider terms |

## Earth Engine setup

1. Register a [Google Cloud project](https://console.cloud.google.com/)
   with the Earth Engine API enabled.
2. Authenticate: `earthengine authenticate`
3. Copy `.env.example` to `.env` and set `EE_PROJECT`.

## Data pipeline order

The `src/data_prep/` scripts are numbered `v0` through `v8` in execution
order. Intermediate outputs land in `data/02_interim/` and `data/03_processed/`;
final feature tables are in `data/03_processed/grid_250m_model_features_v8.csv`.
