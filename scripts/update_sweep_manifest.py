"""Refresh the `status` column of sweep_manifest.csv.

Status resolution per row:
  1. `summary.json` on disk -> `completed` (the driver writes it only at the
     very end of main(), after the full AL loop has finished). This is the
     authoritative end-of-run marker.
       NOTE: `state.pt` is overwritten after EVERY iteration — its presence
       only means "at least one iteration finished", not that the run is done.
  2. Else consult sacct:
       - RUNNING                             -> running
       - PENDING                             -> pending
       - COMPLETED but no summary.json       -> missing (driver crashed at
                                                the finalise step, or the
                                                output dir is wrong)
       - FAILED / TIMEOUT / CANCELLED / ...  -> <state>.lower()
       - If state.pt exists but sacct says   -> running (partial progress;
         RUNNING/nothing                       useful signal for dashboards)
  3. If sacct returns nothing and no summary.json, leave the row alone.

Idempotent: safe to run repeatedly, or from cron.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import click
import pandas as pd


TERMINAL_FAILS = {"FAILED", "TIMEOUT", "CANCELLED", "CANCELLED+", "NODE_FAIL",
                  "OUT_OF_MEMORY", "BOOT_FAIL", "PREEMPTED"}


def _sacct_state(job_id: str) -> str:
    """Return the primary state string reported by sacct, or empty."""
    if not job_id or job_id in ("DRY", "nan", ""):
        return ""
    try:
        out = subprocess.check_output(
            ["sacct", "-j", str(job_id), "-n", "-o", "State", "-P"],
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).decode()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return ""
    # sacct may return multiple lines (one per step); take the first non-empty
    for line in out.splitlines():
        state = line.strip().split("|", 1)[0].strip()
        if state:
            return state
    return ""


def _resolve_status(row) -> str:
    """Compute the new status for a single manifest row."""
    current = str(row.get("status", "submitted")).strip().lower()
    if current == "completed":
        return current

    run_dir = row.get("expected_run_dir")
    run_path = Path(run_dir) if isinstance(run_dir, str) and run_dir else None
    summary_done = run_path is not None and (run_path / "summary.json").exists()
    state_pt = run_path is not None and (run_path / "state.pt").exists()

    # 1. summary.json is written only at the very end of main() -> authoritative.
    if summary_done:
        return "completed"

    # 2. Consult sacct.
    state = _sacct_state(row.get("job_id", ""))
    if state.startswith("RUNNING"):
        return "running"
    if state.startswith("PENDING"):
        return "pending"
    if state.startswith("COMPLETED"):
        # Job ended cleanly but didn't write summary.json -> finalise crashed
        # or the output dir is mis-pointed. Worth surfacing distinctly.
        return "missing"
    base = state.rstrip("+").upper() if state else ""
    if base in TERMINAL_FAILS:
        return base.lower()

    # 3. No sacct info: fall back to on-disk progress signal.
    if state_pt:
        return "running"  # at least one iteration has flushed state.pt
    return current


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True, help="Sweep manifest CSV to refresh in place.")
@click.option("--sweep-id", default=None,
              help="Only refresh rows with this sweep_id (default: all).")
@click.option("--summary/--no-summary", default=True,
              help="Print a per-status count summary at the end.")
def main(manifest: str, sweep_id: str, summary: bool) -> None:
    path = Path(manifest)
    if not path.exists():
        raise click.ClickException(f"Manifest not found: {path}")

    df = pd.read_csv(path)
    if sweep_id:
        mask = df["sweep_id"].astype(str) == str(sweep_id)
    else:
        mask = pd.Series([True] * len(df))

    updates = 0
    for i in df.index[mask]:
        new = _resolve_status(df.loc[i])
        old = str(df.at[i, "status"])
        if new != old:
            df.at[i, "status"] = new
            updates += 1
            print(f"[update] job_id={df.at[i, 'job_id']} "
                  f"{df.at[i, 'model']}/{df.at[i, 'strategy']}/{df.at[i, 'warm_start']}/"
                  f"seed{df.at[i, 'seed']}: {old} -> {new}")

    df.to_csv(path, index=False)
    print(f"[manifest] wrote {path} with {updates} status change(s)")

    if summary:
        counts = df["status"].value_counts().to_dict()
        print("[summary]", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
