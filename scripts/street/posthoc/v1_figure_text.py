"""Re-render the zone-scale figures (first-manuscript scripts) with larger type for the 13 in main-text width.

Runs an original figure script unchanged, but intercepts its savefig: every text artist is enlarged by a factor, the
saved file goes to paper/figures/v2/zone/ instead of the tracked paper/figures/main/ copy. Data, layout and colours stay
as the original script draws them.
Usage: python -m scripts.street.revision.v1_figure_text <script_module> <factor> [stem_filter]
"""
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/figures/v2/zone"
_orig = Figure.savefig


def main() -> None:
    mod, factor = sys.argv[1], float(sys.argv[2])
    want = sys.argv[3] if len(sys.argv) > 3 else None
    OUT.mkdir(parents=True, exist_ok=True)

    def savefig(self, fname, *a, **k):
        p = Path(str(fname))
        if p.suffix.lower() != ".png" or (want and want not in p.stem):
            return None
        if not getattr(self, "_type_scaled", False):
            for t in self.findobj(Text):
                if t.get_text():
                    t.set_fontsize(t.get_fontsize() * factor)
            self._type_scaled = True
        return _orig(self, OUT / p.name, *a, **k)

    Figure.savefig = savefig
    sys.argv = [mod]
    runpy.run_module(mod, run_name="__main__")


if __name__ == "__main__":
    main()
