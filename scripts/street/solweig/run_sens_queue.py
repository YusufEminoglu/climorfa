"""CLIMORFA Street — GPU queue for the four sensitivity variants, after the H5 GPU queue. NOT started automatically.

Steps (stops on the first failure; every step logged with timestamps to logs/wp/sens_gpu_queue.log):
  0. preflight: run_sensitivity_subset.py --dry-run for each variant, plus the GPU-hour estimate;
  1. wait for the PID in --pid-file (default logs/wp/H5_gpu_queue.pid; re-read while waiting, so a relaunched H5 queue
     keeps the GPU);
  2. check that the H5 queue finished: the last "H5 GPU queue" line of logs/wp/H5_gpu_queue.log must say
     "FINISHED: all steps OK". Otherwise stop, because H5 has priority for a restart on the GPU
     (--ignore-h5-status overrides);
  3. for each variant: run_sensitivity_subset.py --variant <v> --no-wait, then --validate-only.
Resumable: run_sensitivity_subset.py skips tiles whose outputs validate, so a relaunch continues where it stopped.

--dry-run: runs step 0, reports the PID and H5 status, starts no variant, waits for nothing.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PY = sys.executable
LOG = PROJECT_ROOT / "logs/wp/sens_gpu_queue.log"
H5_LOG = PROJECT_ROOT / "logs/wp/H5_gpu_queue.log"
SENS = PROJECT_ROOT / "scripts/street/solweig/run_sensitivity_subset.py"
VARIANTS = ("radsens", "h26", "h34", "landcover")


class QueueError(RuntimeError):
    pass


def log(msg: str):
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def must(cmd: list[str], step: str):
    log(f"[{step}] $ {' '.join(cmd)}")
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    for line in p.stdout:
        log(f"[{step}]   {line.rstrip()}")
    rc = p.wait()
    log(f"[{step}] exit code {rc} after {(time.time() - t0) / 60:.1f} min")
    if rc != 0:
        raise QueueError(f"step {step} failed with exit code {rc}: {' '.join(cmd)}")


def pid_alive(pid: int | None) -> bool:
    import psutil
    if not pid or not psutil.pid_exists(pid):
        return False
    try:
        return "python" in psutil.Process(pid).name().lower()      # guards against PID reuse by another program
    except psutil.Error:
        return False


def h5_status() -> str:
    """The last 'H5 GPU queue' line of the H5 log, or '' if there is none."""
    if not H5_LOG.exists():
        return ""
    last = ""
    with open(H5_LOG, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if " H5 GPU queue " in ln or " QUEUE STOPPED" in ln:
                last = ln.strip()
    return last


def wait_for_pid(pid_file: Path, poll_s: int):
    from scripts.street.solweig.run_sensitivity_subset import read_pid_file
    t_last = 0.0
    while True:
        pid = read_pid_file(pid_file)
        if pid is None:
            raise QueueError(f"step 1 failed: no PID in {pid_file}")
        if not pid_alive(pid):
            log(f"[1 wait] PID {pid} ({pid_file.name}) is not running")
            return
        if time.time() - t_last > 3600:
            log(f"[1 wait] PID {pid} ({pid_file.name}) still running; polling every {poll_s} s")
            t_last = time.time()
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pid-file", default="logs/wp/H5_gpu_queue.pid")
    ap.add_argument("--poll-s", type=int, default=120)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--ignore-h5-status", action="store_true", help="run even if the H5 queue did not finish cleanly")
    a = ap.parse_args()
    pid_file = PROJECT_ROOT / a.pid_file
    variants = [v.strip() for v in a.variants.split(",")]
    log("=" * 100)
    log(f"Sensitivity GPU queue {'DRY-RUN' if a.dry_run else 'START'} (variants {variants}; pid file {pid_file.name})")
    try:
        must([PY, str(SENS), "--estimate"], "0 estimate")
        for v in variants:
            must([PY, str(SENS), "--variant", v, "--dry-run"], f"0 preflight {v}")
        if a.dry_run:
            from scripts.street.solweig.run_sensitivity_subset import read_pid_file
            pid = read_pid_file(pid_file)
            log(f"[1 wait] PID {pid} ({pid_file.name}) {'is running' if pid_alive(pid) else 'is not running'}")
            log(f"[2 H5 status] {h5_status() or '(no status line)'}")
            log("Sensitivity GPU queue DRY-RUN finished: nothing started, nothing waited for")
            return
        wait_for_pid(pid_file, a.poll_s)
        st = h5_status()
        log(f"[2 H5 status] {st or '(no status line)'}")
        if "FINISHED: all steps OK" not in st and not a.ignore_h5_status:
            raise QueueError("step 2 failed: the H5 queue did not finish cleanly; H5 has priority on the GPU "
                             "(--ignore-h5-status overrides)")
        for v in variants:
            must([PY, str(SENS), "--variant", v, "--no-wait"], f"3 {v}")
            must([PY, str(SENS), "--variant", v, "--validate-only"], f"3 {v} validate")
        log("Sensitivity GPU queue FINISHED: all steps OK")
    except QueueError as e:
        log(f"QUEUE STOPPED: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"QUEUE STOPPED (unexpected error): {e!r}")
        raise


if __name__ == "__main__":
    main()
