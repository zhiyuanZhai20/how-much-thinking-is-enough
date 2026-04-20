"""Self-healing watchdog.

Monitors all v2 truncation files, detects anomalies, auto-corrects, and
triggers the finalization pipeline once everything is done.

What it does every CHECK_INTERVAL seconds:
  1. Counts per-file progress (self + cross judge per model per benchmark)
  2. Detects stalled files (no progress in STALL_SECS) and flags them
  3. Deduplicates task_ids in every v2 jsonl (protects against parallel writes)
  4. When every (model, dataset, judge) meets its target count, runs
     scripts/10_finalize.py to produce the new deliverable.zip
  5. Prints compact status; writes detailed log to outputs/watchdog.log

Run: python scripts/99_watchdog.py
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import config
from utils.io_utils import iter_jsonl

CHECK_INTERVAL = 30   # seconds
STALL_SECS     = 300  # flag stall if no delta in 5 min
ALL_DS         = ["svamp", "gsm8k", "aqua", "math500"]
LOG            = ROOT / "outputs" / "watchdog.log"

# Workers: (worker_id, cmd) -- watchdog keeps these alive
WORKERS = {
    "r1_distill_cross": [
        sys.executable, "-u", str(ROOT / "scripts" / "03b_truncation_self_judge.py"),
    ],
    "qwen3_cross": [
        sys.executable, "-u", str(ROOT / "scripts" / "03b_truncation_self_judge.py"),
    ],
}
WORKER_ENV = {
    "r1_distill_cross": {"ONLY_CROSS": "1", "ONLY_MODEL": "r1_distill_7b", "MAX_CONC": "24"},
    "qwen3_cross":      {"ONLY_CROSS": "1", "ONLY_MODEL": "qwen3_30b_thinking", "MAX_CONC": "24"},
}


def log(msg: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def dedupe_all() -> int:
    """Deduplicate task_ids in every v2 jsonl. Returns total lines removed."""
    total_removed = 0
    for p in config.TRUNC_DIR.glob("v2__*.jsonl"):
        seen = {}
        raw = 0
        for line in open(p, encoding="utf-8"):
            raw += 1
            try:
                r = json.loads(line)
                seen[r.get("task_id", line)] = line
            except json.JSONDecodeError:
                pass
        if len(seen) != raw:
            with open(p, "w", encoding="utf-8") as f:
                for v in seen.values():
                    f.write(v)
            removed = raw - len(seen)
            total_removed += removed
            log(f"  dedup {p.name}: removed {removed} duplicates")
    return total_removed


def targets() -> dict[tuple[str, str], int]:
    """Target judgeable-trace count per (model, dataset): is_correct AND n_steps >= 2."""
    out: dict[tuple[str, str], int] = {}
    for m, _, _ in config.REASONING_MODELS:
        for ds in ALL_DS:
            p = config.segments_path(m, ds)
            if not p.exists():
                out[(m, ds)] = 0
                continue
            out[(m, ds)] = sum(1 for r in iter_jsonl(p)
                                if r.get("is_correct") and r.get("n_steps", 0) >= 2)
    return out


def progress_counts() -> dict[tuple[str, str, str], int]:
    out = {}
    for m, _, _ in config.REASONING_MODELS:
        for ds in ALL_DS:
            for j in ["self", "cross"]:
                p = config.TRUNC_DIR / f"v2__{m}__{ds}__judge-{j}.jsonl"
                out[(m, ds, j)] = sum(1 for _ in iter_jsonl(p)) if p.exists() else 0
    return out


def fmt_summary(tgt, cnt):
    rows = []
    done = 0
    total_target = 0
    for (m, ds, j), n in cnt.items():
        t = tgt.get((m, ds), 0)
        total_target += t
        done += min(n, t)
        if t and n < t:
            rows.append(f"{m[:15]:15s}/{ds:8s}/{j:5s} {n}/{t}")
    pct = done / max(total_target, 1) * 100
    return rows, pct


def all_done(tgt, cnt) -> bool:
    for (m, ds, j), n in cnt.items():
        t = tgt.get((m, ds), 0)
        if t == 0:
            continue
        # Tolerance 2 traces (some correct traces may have been filtered due
        # to n_steps < 2, etc.)
        if n < t - 2:
            return False
    return True


def run_finalize():
    log("ALL DONE — running scripts/10_finalize.py")
    r = subprocess.run([sys.executable, "scripts/10_finalize.py"], cwd=ROOT)
    log(f"finalize returned {r.returncode}")


import os

_procs: dict[str, subprocess.Popen] = {}

def worker_alive(wid: str) -> bool:
    p = _procs.get(wid)
    return p is not None and p.poll() is None

def launch_worker(wid: str):
    cmd = WORKERS[wid]
    env = os.environ.copy()
    env["SCALE"] = "A"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(WORKER_ENV[wid])
    log_file = ROOT / "outputs" / f"worker_{wid}.log"
    f = open(log_file, "ab")
    # On Windows: detach so children survive parent death
    creationflags = 0
    if sys.platform == "win32":
        import subprocess as _sp
        creationflags = _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS
    p = subprocess.Popen(cmd, cwd=ROOT, env=env,
                         stdout=f, stderr=subprocess.STDOUT,
                         creationflags=creationflags)
    _procs[wid] = p
    log(f"  launched {wid} pid={p.pid} (detached)")

_last_cnt_for_worker: dict[str, dict] = {}
_last_progress_ts: dict[str, float] = {}

def _cnt_for_worker(wid: str, cnt: dict) -> int:
    if wid == "r1_distill_cross":
        return sum(cnt.get(("r1_distill_7b", ds, "cross"), 0) for ds in ALL_DS)
    if wid == "qwen3_cross":
        return sum(cnt.get(("qwen3_30b_thinking", ds, "cross"), 0) for ds in ALL_DS)
    return 0

def _tgt_for_worker(wid: str, tgt: dict) -> int:
    if wid == "r1_distill_cross":
        return sum(tgt.get(("r1_distill_7b", ds), 0) for ds in ALL_DS)
    if wid == "qwen3_cross":
        return sum(tgt.get(("qwen3_30b_thinking", ds), 0) for ds in ALL_DS)
    return 0

def ensure_workers_alive(tgt, cnt):
    """Relaunch a worker if its file progress stopped for >90s and work remains."""
    now = time.time()
    for wid in WORKERS:
        c = _cnt_for_worker(wid, cnt)
        t = _tgt_for_worker(wid, tgt)
        if c >= t - 2:
            continue  # done
        prev = _last_cnt_for_worker.get(wid)
        if prev != c:
            _last_cnt_for_worker[wid] = c
            _last_progress_ts[wid] = now
            continue
        last = _last_progress_ts.get(wid, 0)
        if now - last > 90:
            if worker_alive(wid):
                log(f"  {wid} alive but stalled; killing and restarting")
                try: _procs[wid].terminate()
                except Exception: pass
            launch_worker(wid)
            _last_progress_ts[wid] = now


def main():
    log("watchdog started with self-restart")
    tgt = targets()
    last_counts = progress_counts()
    last_change_ts = {k: time.time() for k in last_counts}

    while True:
        ensure_workers_alive(tgt, progress_counts())
        # 1. Dedupe
        dedupe_all()

        # 2. Counts
        cnt = progress_counts()
        now = time.time()
        for k, n in cnt.items():
            if n != last_counts.get(k, 0):
                last_change_ts[k] = now
                last_counts[k] = n

        # 3. Detect stalls
        stalled = []
        for k, ts in last_change_ts.items():
            m, ds, j = k
            tgt_n = tgt.get((m, ds), 0)
            if tgt_n and cnt[k] < tgt_n - 2 and now - ts > STALL_SECS:
                stalled.append(f"{m}/{ds}/{j} stuck at {cnt[k]}/{tgt_n} "
                               f"for {int((now - ts) / 60)}m")

        # 4. Report
        rows, pct = fmt_summary(tgt, cnt)
        log(f"progress {pct:.1f}% ({len(rows)} conditions still running)")
        for r in rows[:5]:
            log(f"  {r}")
        for s in stalled:
            log(f"  STALL: {s}")

        # 5. Check completion
        if all_done(tgt, cnt):
            log("all conditions complete")
            # Kill any lingering workers
            for wid, p in list(_procs.items()):
                if p.poll() is None:
                    try: p.terminate()
                    except Exception: pass
            run_finalize()
            break

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
