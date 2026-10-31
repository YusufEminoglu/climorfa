"""Poll the real tables every 30 min (max 12 h); re-run run_all.py when the number of finished tiles grew by >= 25 %
since the last analysed state (or reached 427). Logs to outputs/street/analysis/h1/poll_log.txt. Read-only on data/.

    python scripts/street/stats/poll_and_run.py --last-tiles 17
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOG = ROOT / "outputs" / "street" / "analysis" / "h1" / "poll_log.txt"
MET = ROOT / "data" / "street" / "analysis" / "seg_metrics_IMBAT.parquet"


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def n_tiles():
    try:
        m = pd.read_parquet(MET, columns=["n_tiles_at_extraction"])
        return int(m["n_tiles_at_extraction"].max())
    except Exception as e:  # noqa: BLE001
        log(f"read error {e!r}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-tiles", type=int, default=0)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--interval-min", type=float, default=30.0)
    a = ap.parse_args()
    t_end = time.time() + a.hours * 3600
    last = a.last_tiles
    log(f"poll start, last analysed tiles={last}")
    while time.time() < t_end:
        n = n_tiles()
        log(f"tiles in seg_metrics_IMBAT: {n}")
        if n is not None and (n >= 427 or n >= 1.25 * max(last, 1)) and n > last:
            log(f"re-running run_all on {n} tiles")
            r = subprocess.run([sys.executable, str(HERE / "run_all.py")], cwd=str(ROOT))
            log(f"run_all exit {r.returncode}")
            last = n
            if n >= 427:
                log("final data analysed; stop")
                return
        time.sleep(a.interval_min * 60)
    log("poll window ended")


if __name__ == "__main__":
    main()
