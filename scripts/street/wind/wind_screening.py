"""Street-level geometric wind screening model ported from 02Geo3D.

Source reference:
  Plugin file: plugin/wind.js
  Origin: qgis_plugins/zero2geospatial3d/web/src/core/wind.js
  Software: 02Geo3D v1.2.3 (MIT License)
  Author: Yusuf Eminoğlu

MODEL DISCLAIMER:
  This is a geometric screening model (wake deficit, channelling, corner factor),
  NOT CFD (computational fluid dynamics). It provides a fast, city-wide screening
  estimate of sheltered and exposed areas at pedestrian level, suitable for
  macro-urban thermal comfort calculations and prioritisation.

Physics & Formulation:
  For each open ground cell, the ratio to the open-terrain pedestrian wind is:
      ratio = max(0.05, min(1.6, (1 - wake_deficit) * channelling * corner_factor))
  where:
    - wake_deficit: strongest upwind obstacle deficit: A * (1 - porosity) * exp(-x / (k * h))
    - channelling: flow acceleration in street corridors aligned with wind
    - corner_factor: localized downwash and corner acceleration around tall buildings

Pedestrian speed conversion (runners.js):
  U_ped = ratio * U10 * OPEN_PEDESTRIAN_FACTOR
  where OPEN_PEDESTRIAN_FACTOR scales open 10 m wind to 1.5 m pedestrian height
  via a logarithmic profile over open terrain (z0 = 0.03 m):
  ln(1.5 / 0.03) / ln(10 / 0.03) ≈ 0.673425.
"""

from __future__ import annotations
import math
import numpy as np
from numba import njit, prange

# Open-terrain pedestrian scaling factor (10 m to 1.5 m with z0 = 0.03 m)
OPEN_PEDESTRIAN_FACTOR: float = float(np.log(1.5 / 0.03) / np.log(10.0 / 0.03))


@njit(inline="always")
def _js_round(x: float) -> int:
    """Emulate JavaScript Math.round(x): floor(x + 0.5)."""
    return int(math.floor(x + 0.5))


@njit(parallel=True, fastmath=False)
def _wind_ratio_numba(
    heights: np.ndarray,
    porosity: np.ndarray | None,
    has_poro: bool,
    cols: int,
    rows: int,
    cell: float,
    wind_from_deg: float,
    wake_amplitude: float,
    wake_length: float,
) -> np.ndarray:
    """Core numba-compiled parallel implementation matching wind.js exactly."""
    n = cols * rows
    out = np.empty(n, dtype=np.float32)

    # Unit vector pointing upwind (meteorological: from where wind originates)
    # in grid coordinates: +x east = +col, +y north = -row.
    from_rad = wind_from_deg * math.pi / 180.0
    ux = math.sin(from_rad)
    uy = math.cos(from_rad)
    dc = ux
    dr = -uy

    # Perpendicular vector (for street canyon channelling)
    pc = -dr
    pr = dc

    # Global terrain / building height statistics
    h_max = 0.0
    h_sum = 0.0
    h_n = 0
    for i in range(n):
        h = float(heights[i])
        if h > h_max:
            h_max = h
        if h > 0.0:
            h_sum += h
            h_n += 1

    h_mean = h_sum / h_n if h_n > 0 else 0.0

    has_tall = False
    for i in range(n):
        h = float(heights[i])
        if h > 2.0 * h_mean and h > 20.0:
            has_tall = True
            break

    max_steps = int(math.ceil(min(600.0, max(60.0, wake_length * 5.0 * h_max)) / cell))
    side_steps = int(math.ceil(40.0 / cell))

    # Parallel loop over all rows
    for r in prange(rows):
        for c in range(cols):
            i = r * cols + c
            h_self = float(heights[i])
            p_self = float(porosity[i]) if has_poro else 0.0

            # Buildings (solid obstacles: height > 0 and porosity == 0) are NaN
            if h_self > 0.0 and (not has_poro or p_self <= 0.0):
                out[i] = np.nan
                continue

            # 1. Wake deficit from upwind obstacles
            deficit = 0.0
            fetch = max_steps * cell
            for s in range(1, max_steps + 1):
                cc = c + dc * s
                rr = r + dr * s
                if cc < 0.0 or rr < 0.0 or cc > (cols - 1) or rr > (rows - 1):
                    break
                ic = _js_round(cc)
                ir = _js_round(rr)
                if ic < 0 or ir < 0 or ic >= cols or ir >= rows:
                    h = 0.0
                else:
                    h = float(heights[ir * cols + ic])

                if h <= 0.0:
                    continue

                x = s * cell
                if x < fetch:
                    fetch = x

                p_val = 0.0
                if has_poro and ic >= 0 and ir >= 0 and ic < cols and ir < rows:
                    p_val = float(porosity[ir * cols + ic])

                d = wake_amplitude * (1.0 - p_val) * math.exp(-x / (wake_length * h))
                if d > deficit:
                    deficit = d
                if x > wake_length * 5.0 * h_max:
                    break

            # 2. Channelling: buildings close on both sides, open along the wind
            left = 0.0
            right = 0.0
            hl = 0.0
            hr = 0.0

            for s in range(1, side_steps + 1):
                if left > 0.0:
                    break
                ic = _js_round(c + pc * s)
                ir = _js_round(r + pr * s)
                if ic < 0 or ir < 0 or ic >= cols or ir >= rows:
                    h = 0.0
                else:
                    h = float(heights[ir * cols + ic])
                if h > 0.0:
                    left = s * cell
                    hl = h

            for s in range(1, side_steps + 1):
                if right > 0.0:
                    break
                ic = _js_round(c - pc * s)
                ir = _js_round(r - pr * s)
                if ic < 0 or ir < 0 or ic >= cols or ir >= rows:
                    h = 0.0
                else:
                    h = float(heights[ir * cols + ic])
                if h > 0.0:
                    right = s * cell
                    hr = h

            channel = 1.0
            if left > 0.0 and right > 0.0:
                width = left + right
                h_side = min(hl, hr)
                along = min(1.0, fetch / (3.0 * width))
                channel = 1.0 + 0.25 * along * min(1.0, h_side / width)

            # 3. Corner / downwash: a much taller building close by
            corner = 1.0
            if has_tall:
                rad = int(math.ceil(30.0 / cell))
                best = 0.0
                for dy in range(-rad, rad + 1, 2):
                    for dx in range(-rad, rad + 1, 2):
                        ic = c + dx
                        ir = r + dy
                        if ic < 0 or ir < 0 or ic >= cols or ir >= rows:
                            h = 0.0
                        else:
                            h = float(heights[ir * cols + ic])
                        if h > 2.0 * h_mean and h > 20.0:
                            dist = math.hypot(dx, dy) * cell
                            if dist < 0.6 * h:
                                candidate = (1.0 - dist / (0.6 * h)) * min(1.0, h / (4.0 * h_mean))
                                if candidate > best:
                                    best = candidate
                corner = 1.0 + 0.35 * best

            # Final ratio clamped between 0.05 and 1.6
            val = (1.0 - deficit) * channel * corner
            if val < 0.05:
                val = 0.05
            elif val > 1.6:
                val = 1.6
            out[i] = val

    return out


def wind_ratio(
    heights: np.ndarray,
    porosity: np.ndarray | None = None,
    cols: int | None = None,
    rows: int | None = None,
    cell: float = 2.0,
    wind_from_deg: float = 270.0,
    wake_amplitude: float = 0.75,
    wake_length: float = 3.5,
) -> np.ndarray:
    """Calculate pedestrian wind ratio for a 2D height grid.

    Parameters
    ----------
    heights : np.ndarray
        Obstacle heights above ground per cell (0 = open ground).
        Can be 2D (rows, cols) or 1D (rows * cols).
    porosity : np.ndarray, optional
        Optical porosity per cell in [0, 1] (trees ~ 0.5, buildings 0).
    cols : int, optional
        Number of grid columns (inferred if heights is 2D).
    rows : int, optional
        Number of grid rows (inferred if heights is 2D).
    cell : float
        Grid cell size in metres (default 2.0 m).
    wind_from_deg : float
        Meteorological wind direction in degrees (0 = North, 90 = East, etc.).
    wake_amplitude : float
        Wake deficit amplitude (default 0.75).
    wake_length : float
        Wake recovery length in obstacle heights (default 3.5).

    Returns
    -------
    np.ndarray
        Array of shape matching input heights with ratio values in [0.05, 1.6],
        with NaN inside solid buildings.
    """
    is_2d = heights.ndim == 2
    if is_2d:
        r_in, c_in = heights.shape
        if rows is None:
            rows = r_in
        if cols is None:
            cols = c_in
        h_flat = np.ascontiguousarray(heights.ravel(), dtype=np.float32)
    else:
        if rows is None or cols is None:
            raise ValueError("cols and rows must be provided when heights is 1D")
        h_flat = np.ascontiguousarray(heights, dtype=np.float32)

    has_poro = porosity is not None
    if has_poro:
        p_flat = np.ascontiguousarray(porosity.ravel(), dtype=np.float32)
    else:
        p_flat = np.zeros(1, dtype=np.float32)

    res_1d = _wind_ratio_numba(
        heights=h_flat,
        porosity=p_flat,
        has_poro=has_poro,
        cols=cols,
        rows=rows,
        cell=float(cell),
        wind_from_deg=float(wind_from_deg),
        wake_amplitude=float(wake_amplitude),
        wake_length=float(wake_length),
    )

    if is_2d:
        return res_1d.reshape((rows, cols))
    return res_1d


def lawson_class(speed: float) -> dict[str, str] | None:
    """Indicative pedestrian comfort class for a wind speed (m/s) at pedestrian height.

    Based on the LDDC (Lawson) criteria:
      - Sitting: < 4 m/s
      - Standing: < 6 m/s
      - Strolling: < 8 m/s
      - Business walking: < 10 m/s
      - Uncomfortable: >= 10 m/s
    """
    if not math.isfinite(speed):
        return None
    if speed < 4.0:
        return {"key": "sitting", "label": "Sitting", "color": "#2c7bb6"}
    if speed < 6.0:
        return {"key": "standing", "label": "Standing", "color": "#abd9e9"}
    if speed < 8.0:
        return {"key": "strolling", "label": "Strolling", "color": "#ffffbf"}
    if speed < 10.0:
        return {"key": "walking", "label": "Business walking", "color": "#fdae61"}
    return {"key": "uncomfortable", "label": "Uncomfortable", "color": "#d7191c"}
