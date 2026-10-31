# CLIMORFA v2 — Pre-analysis lock (H1–H5)

> Locked 2026-10-30 by the master, BEFORE any hypothesis-level result was computed. The only outputs seen so far are model
> sanity values (pilot tiles, 15:00 sunlit/shaded medians) and one UTCI wind-sensitivity check. Changes after this date
> need a dated amendment below, with the reason, and the original analysis is still reported.

## 1. Units, labels, metrics

**Pedestrian space (PS):** 2 m cells that are on land (landmask = 1), not building (building_dsm − dem ≤ 0.5 m), and within
15 m of a `roads_fua` centreline. All road classes are included. Sensitivity runs use 10 m and 20 m buffers and exclude
motorway-class roads (if a class field exists).

**Street segment (primary unit):** `roads_fua` lines split at intersections (nodes where ≥ 3 segment ends meet) and at tile
borders. Segments are merged when < 20 m, and dropped if they have < 25 PS cells. Each PS cell goes to its nearest segment.
Segment metrics are summaries over its PS cells.

**Grid units (scale ladder):** 50, 100, 250 and 500 m squares aligned to the 1 km tiles. The 250 m level is the old
CLIMORFA grid.

**LCZ label:**
- Primary: global LCZ map (data/01_raw/gee/lcz/izmir_fua_rub_lcz_latest.tif, Demuzere et al., 100 m), sampled as the
  majority class over the segment's PS cells.
- Sensitivity: the old 250 m weak labels (data/03_processed/grid_lcz_weak_labels.gpkg).
- Classes with < 30 segments are pooled into "other built" or "other natural" and reported separately.

**District:** from data/03_processed/analysis_grids.gpkg.

**Metrics** (design day d ∈ {IMBAT 2022-07-05, CALM 2023-07-26}). Hour labels follow data/street/solweig/band_hour_table.csv:
band "15" = interval 14:00–15:00 local, hour-ending.
- M1 primary: UTCI_street at band 15, median over the segment's PS cells.
- M2: daytime strong heat-stress hours = number of bands 10–18 with segment-median UTCI_street ≥ 32 °C (UTCI category
  "strong heat stress" or worse); also ≥ 38 °C ("very strong").
- M3: Tmrt at band 15, median.
- M4: shade fraction = mean over bands 10–18 of (1 − sunlit share).
- M5: UTCI_uniform at band 15 (no street-wind structure).

**Uncertainty:** spatial block bootstrap, 2 km blocks, 1000 resamples, 95 % CI on every effect. Report effect sizes with
CIs, not bare p-values. The five hypotheses are a family; report all of them, whatever their direction.

## 2. Hypotheses and pre-specified analyses

**H1 Scale mismatch.** On segments, per day:
- Fit the LMM M1 ~ 1 + (1 | district / LCZ). Report the ICC for LCZ-within-district and district, and the residual
  (within-LCZ) share.
- Compare R² of an LCZ-only fixed-effect model with a street-morphology model (LightGBM; the H2 features minus LCZ), both
  under 5-fold spatial-block CV (2 km blocks).
- Scale ladder: share of M1 variance explained by LCZ (η²) at the segment, 50, 100, 250 and 500 m levels.
- **H1 supported** if the within-LCZ residual share is > 50 % of total M1 variance AND CV R² of the LCZ-only model is
  < 0.30 on both days. Otherwise report it as not supported.

**H2 Mechanisms.** LightGBM on segments, M1 per day, same CV folds. Features grouped in families:
- shade/geometry: SVF median, H/W, mean building height and building cover within 30 m
- vegetation: tree cover and mean canopy height within 15 m
- orientation: sin and cos of street azimuth; angle between street axis and sun azimuth at band 15; angle to the day's
  mean 14–17 h wind direction
- ventilation: median wind ratio
- context: distance to coast, elevation, slope
- LCZ as categorical, in a separate model
Report TreeSHAP family shares, partial dependence for H/W, SVF, tree cover and orientation, and the within-LCZ residual
explained (R² of a model on M1 minus its LCZ-mean).

**H3 Surface ≠ pedestrian.**
- Landsat summer LST (data/02_interim/rasters/izmir_fua_landsat_summer_2021_2025_lst_epsg5253_30m.tif): median over PS
  cells per segment, and per 250 m cell.
- Spearman ρ(LST, M3) and ρ(LST, M1) overall and by LCZ.
- Bivariate local Moran (LST vs M1) at 250 m.
- "Decoupled" segments = top LST tercile and bottom M1 tercile (and vice versa). Report their share by LCZ and by audit
  class (data from the manual LCZ audit, where available).
- Report ρ whatever its value. The "compact paradox" counts as observed only if ρ(LST, M1) < 0 within LCZ 2/3 with a 95 %
  CI that excludes 0.

**H4 Shade vs breeze.** For each segment and day, Shapley two-factor decomposition of M1 relative to the city median
reference (Tmrt_ref, v_ref) = (median Tmrt, median v10_eff) over all PS cells at band 15:
- shade term = ½[UTCI(Tmrt_i, v_ref) − UTCI(Tmrt_ref, v_ref) + UTCI(Tmrt_i, v_i) − UTCI(Tmrt_ref, v_i)]
- breeze term = the same expression with the roles of Tmrt and v swapped. Ta and RH are fixed at the forcing values.
Report:
- the variance of each term across segments, per day
- the IMBAT − CALM contrast
- the interaction with orientation class (aligned ≤ 22.5°, oblique, perpendicular ≥ 67.5° to the 14–17 h wind) × H/W tercile
**H4 is "shade-dominated"** if Var(shade term) > 4 × Var(breeze term) on both days. A known physical caveat (UTCI wind
response is weak near Ta ≈ 35 °C) is reported as a mechanism, not hidden.

**H5 Targeting.**
- Tree budget N = the number of trees needed to plant the hottest 10 % of segment length (by M2 on IMBAT) at 8 m spacing.
- Scenario S (street-targeted): plant those segments.
- Scenario Z (zone-targeted): the same N trees spread proportionally over segments of the LCZ class with the highest
  mean M2, choosing within it by a random draw (seed 20261030, 20 draws).
- Tree template: crown height 8 m, crown diameter 6 m, trunk zone 2 m, placed in PS cells 2.5–4 m from the segment axis on
  the side with more PS cells; SOLWEIG reruns only for the affected tiles, both days.
- Outcome: Δ(M2 × pedestrian weight) summed over all segments. The pedestrian weight is the 250 m network-reach centrality
  (data/03_processed/grid_250m_road_network_reach_features.csv; the field is chosen and named in WP3.5 before the
  outcome is computed) × the population of the 250 m cell.
- Also report the Jaccard overlap of the priority sets (S vs Z top 10 %).
- **H5 supported** if S reduces the weighted heat-stress hours by ≥ 1.5 × Z (median over the Z draws, 95 % CI from the draws).

## 3. Sensitivity (reported for H1, H2 headline numbers)
- Radiation: RADSENS forcing.
- Building height: h26 / h34 variants. Only if GPU time allows; otherwise on a stratified 40-tile subset, stated.
- PS buffer: 10 / 20 m.
- LCZ label source.
- Segment metric median vs p90.

## 4. Data status at lock
The SOLWEIG and UTCI runs are in progress. Analysis code is developed and tested on the completed tiles. Final numbers come
only from the complete 427-tile run. Any number in a report from a partial run must be labelled PARTIAL (n tiles).

## Amendments
**A1 (2026-10-30, data-source clarification, no result involved).** `district` is not a field of
data/03_processed/analysis_grids.gpkg. It is taken from data/03_processed/grid_lcz_weak_labels.gpkg (same 250 m grid).
224 segments without a match get district = "unknown". They are kept in all analyses except the district random
effect, where they form their own level.

**A2 (2026-10-30, label clarification).** The LCZ raster has 3 unnamed bands, identified from gee/scripts/download_lcz_weak_labels.py
as LCZ, LCZ_Filter and LCZ_Probability. Primary lcz_primary = band 1 "LCZ", the same band the original CLIMORFA pipeline
used (src/data_prep/aggregate_lcz_to_grid.py), for continuity with the weak labels and the manual audit. Sensitivity
analysis (§3): band 2 "LCZ_Filter". Bands 1 and 2 agree on 89.7 % of pixels. The choice was the literal default of the
WP3.0 worker, confirmed by the master before any H1/H2 LCZ comparison was computed. The H3 PARTIAL run (17 peripheral
tiles) used band 1; it played no role in the choice.

**A3 (2026-10-30, implementation detail).** PS cells whose nearest segment belongs to a different tile are set to 0
(0.75 % of the 15 m-buffer cells). Otherwise per-tile extraction would require cross-tile segment ids. A segment split at a
tile border is a separate segment on each side.

**A4 (2026-10-30, data construction for the H5 weight; decided before any H5 outcome on the final data).** The population
term of the H5 pedestrian weight is residential population. The mahalle population is distributed dasymetrically onto
residential building cells only (buildings_fua YAPITIP = MESKEN), not onto all buildings. The WP3.0c grids
(all-building dasymetry) put residents into industrial halls (e.g. Çiğli AOSB). Both variants are kept; the residential
one is primary for H5 and the all-building one is reported as a sensitivity. The reach field is fixed as
network_reach_400m_road_length_m (chosen in WP3.0c before use).

**A5 (2026-10-30, H5 Scenario Z wording, decided before any final H5 outcome).** "The same N trees spread proportionally
over segments of the LCZ class with the highest mean M2, choosing within it by a random draw" is made operational as: each
draw samples segments of that LCZ class at random, without replacement and with probability ∝ segment length, until their
cumulative length at 8 m spacing absorbs N trees (the last segment partially). The 20 draws therefore select different
segment sets. The WP3.5 PARTIAL run had identical Jaccard values across draws, which shows the earlier wording was not
implemented as a random selection.

**A6 (2026-10-30, label hygiene; found in WP3.5b, before any final result).** lcz_primary = 0 (and lcz_weak250 = 0) is
the no-data value of the LCZ raster (sea, outside the map), not a class. Such segments are excluded from every
LCZ-based statistic (H1 LMM and η², H2 LCZ model, H3 by-LCZ results, H5 target class) and reported as a count. They stay
in analyses that do not use LCZ (H2 morphology model, H4). The WP3.5 PARTIAL run had picked LCZ 0 as the Z target class.

**A7 (2026-10-30, master rulings on WP3.124 ambiguities and a forcing error; made on PARTIAL 17-tile data only, without
changing any threshold).**
1. LMM fallback, if MixedLM fails to converge: nested-ANOVA (method-of-moments) variance components with the same
   hierarchy. Report which estimator was used per day.
2. H1 "on both days" applies to BOTH clauses (residual share > 50 % AND CV R² of LCZ-only < 0.30, each on both days).
3. VRB winds. The CALM-day METAR reports VRB (variable direction) at 13:50, 14:50 and 15:20 local. The met builder took
   the missing direction as 0° (north), giving spurious Wd at it = 14 (34.3°) and it = 15 (0.0°). Rule: an hour is
   VRB-flagged if any METAR report in its hour-ending window is VRB. For VRB hours the street-wind ratio is the mean of the
   16 sector ratios (direction-averaged sheltering), and Wd is not used. U10 is kept. UTCI_street and the segment metrics
   are recomputed for the affected hours on both days (check IMBAT for VRB too). SOLWEIG Tmrt is unaffected (it does not use
   wind direction). Orientation classes relative to wind (H2 angle-to-wind feature, H4 interaction) use the day's 14–17 h
   circular mean of non-VRB reports only. Its resultant length R is reported; if R < 0.5 the CALM orientation results are
   descriptive only, and the H4 orientation interaction is interpreted on IMBAT.
4. The H1 residual share is reported as "within district × LCZ" (the LMM residual), plus the one-way segment-level
   between-LCZ η² from the scale ladder.
5. The H4 interaction is operationalised as the orientation-class × H/W-tercile table of mean shade and breeze terms, plus
   the IMBAT − CALM difference-in-differences, with block-bootstrap CIs.
6. Sensitivities: the LCZ_Filter (A2), PS buffer 10/20 m and p90 variants are computed from re-extraction once all tiles are
   done (WP3.6). The RADSENS and h26/h34 variants need SOLWEIG reruns and are run on a stratified 40-tile subset after the
   main run (as §3 allows), stated as such.
   A7 follow-up (2026-10-30, after WP2.3b). The rule above flags CALM hours it = 6, 7, 8, 9, 14, 15, 16. That is intended:
   it is the literal rule (VRB reports at 08:20, 08:50, 13:50, 14:50, 15:20). The corrected CALM 14–17 h reference is 193.0°,
   R = 0.872, from only 4 non-VRB reports; IMBAT is 335.0°, R = 0.989. Since R ≥ 0.5, the "descriptive only" clause does
   not trigger, but every CALM orientation result must state n = 4 reports and R. Pending before the sensitivity runs:
   fix the root cause (fillna(0.0) of missing drct in scripts/street/forcing_summer/04_build_forcing_and_compare.py) and
   regenerate the RADSENS met files with the v2 Wd.

**A8 (2026-10-30, master rulings on methods-drafting questions; made before the full run was complete, no threshold
changed).** (1) The H2 sun-azimuth feature uses the sun azimuth at 14:30 local, the midpoint of the label-15 interval
(hour-ending 14:00–15:00), matching SOLWEIG-GPU's midpoint sun position. (2) H/W is undefined for segments without
enclosing buildings on either side (24 %). LightGBM uses NaN natively. In the H4 orientation × H/W table these segments
form a separate class, "no enclosing buildings". (3) The motorway-exclusion sensitivity listed in §1 is run in Phase B
(road_class / YOLTIP = OTOYOL), as written.

**A9 (2026-10-30, ADDITIONAL sensitivity; the primary analysis is unchanged; added before the full run was complete).**
WP3.7 found that the primary SOLWEIG run had no land-cover raster, so SOLWEIG-GPU defaults applied uniformly to the
ground (albedo 0.15, emissivity 0.95; docs/climorfa_street/solweig_parameters.md). An extra sensitivity variant
`landcover` reruns the 40-tile subset with a SOLWEIG land-cover raster: paved from roads_fua and the ESA WorldCover
built/bare classes, grass/vegetation from WorldCover, water, buildings. It reports the change in the H1/H2 headline numbers
and in M1/M3 on those tiles. The primary results stay those of the uniform-ground run; the uniform ground is stated as a
limitation.



**A10 (2026-10-30, H5 operationalisation; approved by the author; made before any H5 outcome was computed; the H5
decision rule, budget definition, tree template, weights and the 20 draws are unchanged).** Found by the master on
descriptive properties of the partial (363-tile) data, without any scenario outcome:
(i) M2 at the locked threshold is saturated on the design days: on IMBAT 79 % of segments reach 8 or 9 of the 9 bands
10–18 at UTCI ≥ 32 °C. The sunlit-to-shaded UTCI difference at 15:00 is about 5.6 °C, so tree shade cannot move most
street cells below 32 °C and the locked outcome cannot register a tree effect. At ≥ 38 °C (the "very strong" category)
the hours range from 0 to 6.
(ii) The LCZ class with the highest mean M2 is LCZ 16 (129 segments, capacity about 2,800 trees at 8 m spacing), which
cannot absorb the budget (about 62,000 trees). Scenario Z as locked is not executable.
(iii) The H5 evaluation code used a placeholder effect per tree instead of the SOLWEIG reruns (a code defect, fixed;
not an amendment).
(iv) Running SOLWEIG for every Z draw touches about 250 tiles per draw (about 5,000 tile runs), which is not feasible.
Rulings:
1. Primary H5 outcome and budget ranking use M2_38 = the number of bands 10–18 with segment-median UTCI_street ≥ 38 °C
   (column m2_vstrong). The locked M2 (≥ 32 °C) version is computed with the same machinery and reported.
2. Ties in the budget ranking are broken by M1 (descending), then by seg_id.
3. The Z target class is the LCZ class with the highest mean M2 among the classes whose planting capacity (sum of
   floor(length / 8 m)) is at least N. A6 still applies.
4. Scenario effects are computed additively. One city-wide SOLWEIG + UTCI rerun (scenario P) plants the tree template on
   every segment. The per-segment effect is ΔM2_seg = M2_seg(baseline) − M2_seg(P), on that segment's own PS cells. A
   scenario's outcome is Σ ΔM2_seg × pedestrian weight over its segments; the last Z segment is prorated by planted
   length.
5. Validation, pre-specified. Scenario S is also run exactly (trees on S segments only) on 40 tiles drawn from the S tile
   set (seed 20261030). If the additive estimate of S's weighted ΔM2_38 on those tiles differs from the exact value by
   more than 15 %, the H5 verdict is reported as descriptive only, not as a test. The relative difference is reported
   in either case.
A10 clarifications (2026-10-30, master, during implementation, before any H5 outcome):
- Trees per segment = floor(length / 8 m), the same as the capacity rule.
- The exact value in ruling 5 sums the weighted ΔM2_38 over ALL segments on the 40 validation tiles, so that the shade
  falling on neighbouring segments is included. The S-only sum is reported as well.
- The H5 verdict is taken on IMBAT, the day that defines the budget. CALM is reported with the same machinery.
- Ties in the class means go to the lowest LCZ code.

**A11 (2026-10-30, ADDITIONAL descriptive H5 outputs; the primary outcome, decision rule and A10 are unchanged; added before
any H5 outcome was computed).** Implementation checks on synthetic data and on tree geometry (no scenario result) showed
two properties of the A10 outcome. (i) A segment with M2_38 = 0 cannot lose hours, so S ≥ Z is partly built in by
selecting hot segments. (ii) M2 uses the segment median over a band up to 30 m wide, while the single-sided crowns cover a
median 18 % of a segment's PS cells, so many ΔM2_seg may be 0. These are reported with the verdict, in every case:
1. baseline mean M2_38 of the S and Z segments, and the share of their segments with M2_38 = 0;
2. the share of segments with ΔM2_seg = 0 in scenario P;
3. a cell-based companion outcome: Σ w × Δ(mean over the segment's PS cells of the hours 10–18 with cell UTCI_street ≥ 38 °C),
   and the S/Z ratio on it;
4. a continuous companion: Σ w × Δ(mean UTCI_street over the segment's PS cells at band 15), and the S/Z ratio on it.
The companions are descriptive and do not change the H5 verdict.

A10 correction (2026-10-31, master, before any H5 outcome). The A10 heading says the budget definition and the outcome are
unchanged in form; two consequences must be stated. (1) The budget size rule (hottest 10 % of length, 8 m spacing) is kept,
but its ranking metric moves from M2 to M2_38. (2) The locked outcome is summed over all segments; under A10 ruling 4 it is
summed over the scenario's own segments, so shade falling on neighbouring segments enters only through the ruling 5
validation. The H5 decision rule (S >= 1.5 x Z) is unchanged.
