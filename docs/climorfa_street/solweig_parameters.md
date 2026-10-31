# SOLWEIG / UTCI parameters actually used (WP3.7, for the methods text)

Read from source on 2026-10-30. Nothing was executed on a GPU; values are the constants in the code that runs.

- Package: `solweig_gpu` 2.0.0, `C:\Users\YE\miniforge3\envs\climorfa-street\Lib\site-packages\solweig_gpu\` (abbreviated `SG/` below).
  Main routine `SG/utci_process.py::compute_utci`; radiation model `SG/solweig.py::Solweig_2022a_calc`.
- Driver: `scripts/street/solweig/run_solweig_citywide.py` (abbreviated `DRV`).
- UTCI: `scripts/street/utci/compute_street_utci.py` (abbreviated `UTCI`), pythermalcomfort 4.7.0.
- Cited literature for the defaults (SOLWEIG papers etc.) is **not verified in this session** and is deliberately not named here.

## 0. What the driver does and does not override

`DRV` passes no physical parameters. It stages three rasters per tile (`DRV:109-133`, `stage_tile_inputs`: `Building_DSM`,
`DEM`, `Trees`), calls `run_walls_aspect` (`DRV:412`), `calculate_svf(patch_option=2)` (`DRV:419`) and
`run_utci_tiles(... save_tmrt=True, save_shadow=True ...)` (`DRV:453-462`). A grep of `scripts/street` finds no assignment to
`albedo_g`, `absK`, `absL`, `ewall`, `firstdayleaf`, `transVeg` or `utci_process`. Therefore **all values below are the package defaults**.

Two consequences of what is *not* staged:
1. No `Landcover` raster is staged, so `landcover_path is None` and `SG/utci_process.py:417` is skipped. The `else` branch at
   `utci_process.py:534-542` applies: one uniform ground class for every cell (see section 1).
2. No `WindCoeff` rasters are staged, so `windcoeff = ones` (`utci_process.py:510-511`). SOLWEIG's own UTCI layer (`utci/` rasters)
   therefore has no street-wind structure and is **not used** for M1; M1/M5 come from `UTCI` (section 5).

## 1. Ground and walls

| Quantity | Value | Where | Note |
|---|---|---|---|
| Ground albedo (all ground cells, all land covers) | 0.15 | `utci_process.py:51` (`albedo_g`), applied at `:537` (`alb_grid = Knight + albedo_g`) | uniform; no land-cover dependence in this run |
| Ground emissivity | 0.95 | `utci_process.py:53` (`eground`), applied at `:538` | uniform |
| Ground thermal parameters (Tg) | TgK = 0.37, Tstart = -3.41, TmaxLST = 15 | `utci_process.py:535,536,541` | uniform; these are the "cobblestone" row of `SG/landcoverclasses_2016a.txt` (alb 0.20, TgK 0.37, Tstart -3.41), but albedo is overridden by 0.15 |
| Wall albedo | 0.20 | `utci_process.py:50` (`albedo_b`); passed into `Solweig_2022a_calc` at `:672` and used for wall reflection in `solweig.py:213-215` and `Kup_veg_2015a` (`solweig.py:490-506`) | also named "building albedo" in the code |
| Wall emissivity | 0.90 | `utci_process.py:52` (`ewall`); used in `solweig.py:2187-2189` (`Lside_veg_v2022a`) | |
| Wall thermal parameters | TgK_wall = 0.37, Tstart_wall = -3.41, TmaxLST_wall = 15 | `utci_process.py:539-542` | |
| Roof albedo / emissivity | not separately modelled | roofs are in `buildings` (`:506-508`) and masked out of UTCI (`valid_mask = buildings == 1`, `:509`) | roofs matter only as wall tops and shaders |

If a land-cover raster had been staged, the table in `SG/landcoverclasses_2016a.txt` would apply (asphalt 0.18/0.95, cobble 0.20/0.95,
grass 0.16/0.94, bare soil 0.25/0.94, water 0.05/0.98, walls 0.20/0.90). It was not.

## 2. Human body

| Quantity | Value | Where |
|---|---|---|
| Absorption of shortwave, `absK` | 0.70 | `utci_process.py:54` |
| Absorption of longwave, `absL` (= emissivity of the body) | 0.95 | `utci_process.py:55` |
| Posture | standing, cylinder model (`cyl = True`) | `utci_process.py:57-62` (comment "Standing position") |
| Projected-area form factors | Fside = 0.22, Fup = 0.06, Fcyl = 0.28 | `utci_process.py:58-60` |
| Sky radiation | anisotropic sky (Perez-type diffuse, Martin-Berdahl-type longwave per the code comment) | `anisotropic_sky = 1`, `utci_process.py:551`; comment at `solweig.py:2203-2204` |
| SVF patches | 153 (`patch_option = 2`) | `utci_process.py:552`, `DRV:419` |
| Tmrt equation | `Sstr = absK*(Kside*Fcyl + (Kdown+Kup)*Fup + (Knorth+Keast+Ksouth+Kwest)*Fside) + absL*((Ldown+Lup)*Fup + Lside*Fcyl + (Lnorth+Least+Lsouth+Lwest)*Fside)`; `Tmrt = (Sstr/(absL*sigma))^0.25 - 273.2` | `solweig.py:2205-2213` |
| `height` | 1.1 | `utci_process.py:594` |

Body height: the only height parameter is `height = 1.1` (`:594`). It is used only to derive `first = round(height) = 1` and
`second = round(height*20) = 22` (`:597-600`), the sampling distances of the ground-view factor (`sunonsurface_2018a`,
`solweig.py:143-145, 218-251`). I found no docstring in `solweig_gpu` that calls it body height or centre height, so describe it as "the
SOLWEIG default sensor height parameter 1.1 m" unless it is checked against the SOLWEIG reference. The Tmrt is therefore a
standing-person Tmrt, not a 1.5 m or 2 m volume average that we parameterised ourselves.

## 3. Trees

| Quantity | Value | Where |
|---|---|---|
| Tree raster | canopy height above ground (m); negatives set to 0 | `utci_process.py:496` |
| Trunk-zone ratio | 0.25 of canopy height (`vegdem2 = temp1*0.25 + DEM`, `vegdsm2 = temp1*0.25 + building DSM`) | `utci_process.py:498, 502` |
| Canopy shortwave transmissivity `transVeg` | 0.03 (3 %) | `utci_process.py:545` |
| Used in | `psi = leafon * transVeg` (`:591`), `svfbuveg` (`:631`), `diffsh` (`:633-635`) | |
| Leaf-off transmissivity | `psi = 0.5` when leaf is off | `utci_process.py:592` |
| Leaf-on period | day of year > 97 and < 300 (exclusive), deciduous (`conifer_bool = False`) | `utci_process.py:67-69, 581-590`; the met reader has the same 97/300 constants at `sun_position.py:940-941` |
| Design days | 2022-07-05 = DOY 186, 2023-07-26 = DOY 207 (`date +%j`; also column `id` of the met files), both inside 98-299, so **leaf-on** on both days | |

For the H5 tree template (crown 8 m, trunk zone 2 m) the 0.25 ratio gives 0.25 x 8 m = 2 m, consistent with the lock.
Trees are a single class: no species, no LAI, no conifer option used. The wind-screening model treats trees >= 2 m as 50 % porous
(`scripts/street/wind/compute_street_wind.py:14, 102-105`), a separate parameter from the radiative 3 %.

## 4. Sun position, forcing columns, location

- Sun position uses the **midpoint** of each hourly stamp: `YMDHM = date + hour + minute - half` with `half = (dectime[1]-dectime[0])/2`
  (`SG/sun_position.py:935, 977-980`). The band labelled 15 (stamp it = 15) is therefore evaluated at 14:30 local, which is the basis of lock amendment A8(1).
- Local time -> UTC uses the time zone found from the tile centre (`TimezoneFinder`, `utci_process.py:489-493`); for Izmir this
  gives UTC+3 on the design days (the project's own check: 14:30 local = 11:30 UTC reproduces the SOLWEIG azimuth to < 0.5 deg in `test_wp37.py`, an independent formula).
- Forcing columns read: Kdn (global, col 14), Kdiff (col 21), Kdir (col 22), Ta (col 11), RH (col 10), pressure (col 12), wind (col 9),
  wind direction (col 23) (`utci_process.py:553-570`). No urban-heat-island increment (`uhii = 0`, `:569-572`; the met files have 24 columns).
- Site altitude: 3 m if the median DEM > 0, else the DEM median (`:482-485`).

## 5. UTCI wind-height convention

- UTCI (pythermalcomfort 4.7.0) expects wind speed **10 m above ground** (`pythermalcomfort/models/utci.py:34-35, 44`; same convention in
  `SG/calculate_utci.py:271`).
- Our UTCI_street (M1) uses `v10_eff = clip(U10 * ratio, 0.5, 17.0)` m/s (`UTCI:10, 403`) with `limit_inputs=False, round_output=False` (`UTCI:425-426`),
  where `U10` = the met-file "Wind" column (METAR speed, knots x 0.514444, interpolated hourly; `scripts/street/forcing_summer/04_build_forcing_and_compare.py:66,294`)
  and `ratio` = the 02Geo3D screening ratio of local to open-terrain 10 m wind (16 sectors, `data/street/wind/ratio16`). So the wind entering UTCI
  is a "10 m-equivalent" wind: the local sheltering ratio is applied but the open-terrain 10 m -> 1.5 m logarithmic reduction is **not**.
- UTCI_uniform (M5) uses `clip(U10, 0.5, 17.0)` (`UTCI:13, 404`).
- `OPEN_PEDESTRIAN_FACTOR = ln(1.5/0.03)/ln(10/0.03) = 0.6734` (`scripts/street/wind/wind_screening.py:35-36`) is used only for the separate 1.5 m pedestrian-wind rasters
  (`compute_street_wind.py`, docstring lines 23-27 of `wind_screening.py`). `UTCI` imports it (`UTCI:75`) but never applies it; `validate_utci_sanity.py` uses it for a cross-check.
- Clipping: 0.5 m/s lower bound is the pythermalcomfort validity limit (`UTCI:166, 185`), 17 m/s the upper; SOLWEIG's own UTCI layer clamps at 0.15 m/s (`utci_process.py:691`) and is not used.
- VRB hours: ratio = mean of the 16 sector ratios (lock A7.3).

## 6. Points worth stating (or checking) in the methods text

1. Ground is radiatively uniform (albedo 0.15, emissivity 0.95, one Tg set) for asphalt, grass, bare ground alike; spatial Tmrt differences come from geometry
   (shade, SVF, walls, trees) only. A land-cover-specific run was not made.
2. Wall albedo 0.20 / emissivity 0.90, no façade materials.
3. Tree transmissivity 3 % in leaf, trunk zone 25 % of canopy height, deciduous leaf-on 97-300; both design days are leaf-on.
4. Body: standing cylinder, absK 0.70, absL 0.95, Fside 0.22, Fup 0.06, Fcyl 0.28; "height" parameter 1.1 (meaning not documented in the package).
5. UTCI wind: 10 m-equivalent, `U10 x street ratio`, bounds 0.5-17 m/s.
