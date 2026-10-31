"""WP2.3b helper: run an unchanged stats script (h1_scale.py / h4_shade_breeze.py, OLD code) against the pre-correction CALM
UTCI rasters. raster_ladder reads data/street/utci/CALM/utci_street under common.PROJECT_ROOT/data/street; this wrapper
points PROJECT_ROOT at outputs/street/utci/wp23b/fakeroot_before (junctions to the real IMBAT / SOLWEIG / wind data and to the
snapshot of the CALM utci_street rasters taken BEFORE the VRB correction). Nothing in data/ is modified.

    python wp23b_run_before.py h1_scale.py <script args...>
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm  # noqa: E402

cm.PROJECT_ROOT = cm.PROJECT_ROOT / "outputs/street/utci/wp23b/fakeroot_before"
script = HERE / sys.argv[1]
sys.argv = [str(script)] + sys.argv[2:]
runpy.run_path(str(script), run_name="__main__")
