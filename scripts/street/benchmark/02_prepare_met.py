"""
CLIMORFA Street — WP0.4 Meteorological Forcing Preparation
Generates synthetic clear-sky UMEP forcing text file for 2025-07-15 for Izmir (38.42 N, 27.14 E, UTC+3).
Marked as SYNTHETIC in accordance with Rule 6.
"""

import os
import json
import numpy as np
import pandas as pd
import pvlib

def generate_synthetic_met():
    project_root = r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas"
    out_dir = os.path.join(project_root, "data/street/benchmark")
    os.makedirs(out_dir, exist_ok=True)

    lat = 38.42
    lon = 27.14
    utc_offset = 3.0
    date_str = "2025-07-15"

    loc = pvlib.location.Location(lat, lon, tz="Etc/GMT-3")
    times = pd.date_range(f"{date_str} 00:00", f"{date_str} 23:00", freq="1h", tz=loc.tz)

    # Clear-sky radiation using Ineichen model
    cs = loc.get_clearsky(times, model="ineichen")

    rows = []
    # 2025-07-15 is day of year 196
    doy = times[0].dayofyear

    sigma = 5.670374419e-8  # Stefan-Boltzmann constant

    for it, t in enumerate(times):
        hour = t.hour
        # Ta sinusoid 26..36 C peaking at 15:00
        ta = 31.0 + 5.0 * np.cos((hour - 15.0) * 2.0 * np.pi / 24.0)

        # RH sinusoid 55..35 % (minimum at 15:00, peak at 03:00)
        rh = 45.0 - 10.0 * np.cos((hour - 15.0) * 2.0 * np.pi / 24.0)

        # Wind sinusoid 2..5 m/s (peaking afternoon at 15:00)
        wind = 3.5 + 1.5 * np.cos((hour - 15.0) * 2.0 * np.pi / 24.0)

        # Radiation components
        ghi = float(cs["ghi"].iloc[it])
        dhi = float(cs["dhi"].iloc[it])
        dni = float(cs["dni"].iloc[it])

        # Ensure non-negative and zero at night
        if ghi < 0.1:
            ghi = 0.0
            dhi = 0.0
            dni = 0.0

        press_kpa = 101.32

        # Downward longwave calculation (Prata 1996 / clear-sky formula)
        # Water vapor pressure in hPa
        e_sat_hpa = 6.112 * np.exp((17.67 * ta) / (ta + 243.5))
        ea_hpa = (rh / 100.0) * e_sat_hpa
        # Precipitable water proxy
        w_pw = 46.5 * (ea_hpa / (ta + 273.15))
        esky = 1.0 - (1.0 + w_pw) * np.exp(-((1.2 + 3.0 * w_pw) ** 0.5))
        esky = np.clip(esky, 0.65, 0.98)
        ldown = float(esky * sigma * ((ta + 273.15) ** 4))

        wd = 270.0  # Westerly imbat breeze

        row = {
            "iy": 2025,
            "id": doy,
            "it": hour,
            "imin": 0,
            "Q*": -999,
            "QH": -999,
            "QE": -999,
            "Qs": -999,
            "Qf": -999,
            "Wind": f"{wind:.2f}",
            "RH": f"{rh:.2f}",
            "Td": f"{ta:.2f}",
            "press": f"{press_kpa:.2f}",
            "rain": "0.00",
            "Kdn": f"{ghi:.2f}",
            "snow": -999,
            "ldown": f"{ldown:.2f}",
            "fcld": -999,
            "wuh": -999,
            "xsmd": -999,
            "lai_hr": -999,
            "Kdiff": f"{dhi:.2f}",
            "Kdir": f"{dni:.2f}",
            "Wd": f"{wd:.1f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    out_txt = os.path.join(out_dir, "met_SYNTHETIC_benchmark_2025-07-15.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        # Header line space-separated
        f.write(" ".join(df.columns) + "\n")
        for _, r in df.iterrows():
            f.write(" ".join(str(val) for val in r.values) + "\n")

    print(f"Generated SYNTHETIC metfile: {out_txt}")
    print(df[["it", "Wind", "RH", "Td", "Kdn", "Kdiff", "Kdir", "Wd"]].head(16))

    # Metadata json
    meta = {
        "is_synthetic": True,
        "date": date_str,
        "location": {"latitude": lat, "longitude": lon, "utc_offset": utc_offset},
        "description": "Synthetic clear-sky hot summer benchmark forcing for Izmir",
        "parameters": {
            "Ta_min_max_C": [26.0, 36.0],
            "Ta_peak_hour": 15,
            "RH_min_max_pct": [35.0, 55.0],
            "Wind_min_max_mps": [2.0, 5.0],
            "max_GHI_Wpm2": float(cs["ghi"].max()),
            "max_DNI_Wpm2": float(cs["dni"].max()),
        }
    }
    out_meta = os.path.join(out_dir, "met_SYNTHETIC_benchmark_2025-07-15.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {out_meta}")

if __name__ == "__main__":
    generate_synthetic_met()
