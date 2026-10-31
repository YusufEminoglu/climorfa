"""SYNTHETIC unit test (is_synthetic=True) of the front-to-front width estimator hw_stations() used for hw_ratio.
Nothing is written to disk; results are printed only and are NOT observations."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import shapely
from build_street_units import hw_stations

is_synthetic = True
bx0, by1 = 0.0, 1200.0
H = W = 600


def raster(fn):
    cx = bx0 + 1 + 2 * np.arange(W)
    cy = by1 - 1 - 2 * np.arange(H)
    X, Y = np.meshgrid(cx, cy)
    return fn(X, Y)


res = []
for width in (10, 16, 24, 40):
    # E-W street along y = 600, buildings where |y-600| >= width/2 up to 50 m, symmetric
    b = raster(lambda X, Y: np.abs(Y - 600) >= width / 2)
    line = shapely.linestrings([(200, 600), (400, 600)])
    Wm, vs, nv = hw_stations(line, b, bx0, by1)
    res.append(("E-W symmetric", width, Wm, vs))
# asymmetric: front at +8 m north, -14 m south
b = raster(lambda X, Y: ((Y - 600) >= 8) | ((600 - Y) >= 14))
Wm, vs, nv = hw_stations(shapely.linestrings([(200, 600), (400, 600)]), b, bx0, by1)
res.append(("E-W asymmetric 8+14", 22, Wm, vs))
# diagonal street (45 deg), buildings beyond perpendicular distance width/2 from the axis y = x
for width in (16, 30):
    b = raster(lambda X, Y: np.abs((Y - X) / np.sqrt(2)) >= width / 2)
    Wm, vs, nv = hw_stations(shapely.linestrings([(300, 300), (500, 500)]), b, bx0, by1)
    res.append(("45deg symmetric", width, Wm, vs))
# open plaza: building only on the north side -> must be NaN
b = raster(lambda X, Y: (Y - 600) >= 8)
Wm, vs, nv = hw_stations(shapely.linestrings([(200, 600), (400, 600)]), b, bx0, by1)
res.append(("one side only (expect nan)", np.nan, Wm, vs))
print("SYNTHETIC test (is_synthetic=True): case, true width m, estimated mean W m, valid share")
ok = True
for c, w, e, v in res:
    print(f"  {c:28s} true={w!s:5} est={e:7.2f} valid={v:.2f}")
    if np.isfinite(w):
        ok &= abs(e - w) <= 1.5
    else:
        ok &= not np.isfinite(e)
print("PASS (|error| <= 1.5 m, NaN when a side is open)" if ok else "FAIL")
