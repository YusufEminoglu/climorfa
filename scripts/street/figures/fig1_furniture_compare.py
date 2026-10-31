"""WP4.1: measure north arrow / scale bar / outline of the NEW Figure 1 against the ORIGINAL Fig 9 map (same functions,
different axes size).  Writes outputs/street/figures/fig1_qa/furniture_compare.json and furniture_compare.png."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[3]
QA = ROOT / "outputs/street/figures/fig1_qa"


def comps(mask, minpix=30):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        area = int((lab[sl] == i).sum())
        if area >= minpix:
            out.append((sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start, area))
    return out


def measure(img, region, name):
    x0, y0, x1, y1 = region
    a = np.asarray(img.convert("RGB"))[y0:y1, x0:x1].astype(int)
    black = (a.sum(axis=2) < 12)
    navy = (np.abs(a - np.array([44, 62, 80])).sum(axis=2) < 40)
    ink = (np.abs(a - np.array([0x22, 0x22, 0x22])).sum(axis=2) < 30)
    bars = [c for c in comps(black, 200) if c[2] > 3 * c[3]]
    nav = comps(navy, 60)
    res = dict(name=name, black_bar=bars[:1], navy_components=nav)
    return res, a


def run():
    orig = Image.open(ROOT / "paper/figures/main/fig_9_climate_validation.png")
    new = Image.open(ROOT / "paper/figures/v2/main/fig_1_study_area.png")
    # Fig 9: panel (a) 3900 px wide; furniture around fraction (0.09, 0.59) of the 3600 px axes starting at x=150
    ro, ao = measure(orig, (150, 0, 3900 - 150, 2364), "orig_fig9")
    # new: search the whole map panel region (top 52 % of the sheet, left 60 %)
    W, H = new.size
    rn, an = measure(new, (0, 0, int(0.58 * W), int(0.50 * H)), "new_fig1")
    for r in (ro, rn):
        print(json.dumps(r))
    return ro, rn, orig, new


def furniture_overlay():
    """Scale the ORIGINAL furniture crop to the new figure's km scale (bar length) and overlay."""
    orig = Image.open(ROOT / "paper/figures/main/fig_9_climate_validation.png").convert("RGB")
    new = Image.open(ROOT / "paper/figures/v2/main/fig_1_study_area.png").convert("RGB")
    ro, _ = measure(orig, (150, 0, 3750, 2364), "o")
    W, H = new.size
    rn, _ = measure(new, (0, 0, int(0.58 * W), int(0.50 * H)), "n")
    (bx_o, by_o, bw_o, bh_o, _), (bx_n, by_n, bw_n, bh_n, _) = ro["black_bar"][0], rn["black_bar"][0]
    bx_o += 150
    s = bw_n / bw_o
    # crop: 160 px left of bar, 330 px above, 60 px below, 120 px right (in ORIGINAL pixels), scaled by s
    cl, ca, cb, cr = 90, 380, 60, 90
    o = orig.crop((bx_o - cl, by_o - ca, bx_o + bw_o + cr, by_o + bh_o + cb))
    o = o.resize((int(round(o.width * s)), int(round(o.height * s))), Image.LANCZOS)
    n = new.crop((bx_n - int(round(cl * s)), by_n - int(round(ca * s)), bx_n + bw_n + int(round(cr * s)),
                  by_n + bh_n + int(round(cb * s))))
    w, h = min(o.width, n.width), min(o.height, n.height)
    o, n = o.crop((0, 0, w, h)), n.crop((0, 0, w, h))
    mo = np.asarray(o).astype(int); mn = np.asarray(n).astype(int)

    def furn(a):  # furniture pixels: navy, ink text, black bar
        navy = np.abs(a - np.array([44, 62, 80])).sum(axis=2) < 60
        txt = np.abs(a - np.array([0x22, 0x22, 0x22])).sum(axis=2) < 60
        blk = a.sum(axis=2) < 40
        return navy | txt | blk
    fo, fn = furn(mo), furn(mn)
    iou = (fo & fn).sum() / max((fo | fn).sum(), 1)
    ov = np.full((h, w, 3), 255, np.uint8)
    ov[fo & ~fn] = (220, 40, 40); ov[fn & ~fo] = (40, 90, 220); ov[fo & fn] = (30, 30, 30)
    side = np.concatenate([np.asarray(o), np.asarray(n), ov], axis=1)
    Image.fromarray(side.astype(np.uint8)).resize((side.shape[1] * 1, side.shape[0] * 1)).save(QA / "furniture_compare.png")
    # N glyph + arrow dims
    out = dict(scale_new_over_orig=float(s), bar_len_px=dict(orig=int(bw_o), new=int(bw_n)),
               bar_h_px=dict(orig=int(bh_o), new=int(bh_n)),
               navy=dict(orig=ro["navy_components"], new=rn["navy_components"]),
               overlay_mask_iou=float(iou))
    # outline: fully-inked pixels (#252526) per pixel of true polygon perimeter (robust to anti-aliasing offsets)
    import geopandas as gpd
    study = gpd.read_file(ROOT / "data/02_interim/study_area_fua.gpkg")
    per = study.geometry.iloc[0].boundary.length
    xmin, ymin, xmax, ymax = study.total_bounds

    def mass(img, region, pxm, excl):
        a = np.asarray(img.convert("RGB").crop(region)).astype(int)
        m = (np.abs(a - np.array([37, 37, 38])).sum(axis=2) < 24)
        for (x0_, y0_, x1_, y1_) in excl:
            m[y0_:y1_, x0_:x1_] = False
        lab, n_ = ndi.label(m, structure=np.ones((3, 3)))
        sz = ndi.sum(m, lab, range(1, n_ + 1))
        big = np.isin(lab, 1 + np.where(sz > 3000)[0])
        return float(big.sum() / (per * pxm))
    so = mass(orig, (150, 0, 3750, 2364), 3600 / (xmax - xmin), [(0, 0, 900, 1100)])
    sn = mass(new, (0, 0, int(0.58 * W), int(0.50 * H)), 6.7 * 300 / ((xmax - xmin) * 1.03), [(0, 0, 300, 400)])
    out["outline_inked_px_across_stroke"] = dict(orig=so, new=sn, nominal_lw_px=0.85 / 72 * 300)
    (QA / "furniture_compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ro, rn, orig, new = run()
    (QA / "furniture_compare_raw.json").write_text(json.dumps([ro, rn], indent=1), encoding="utf-8")
    furniture_overlay()

