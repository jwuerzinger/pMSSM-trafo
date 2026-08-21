"""Reclaim the simulator debris from completed active-learning iterations.

The problem, measured on 2026-08-21
-----------------------------------
Each AL iteration leaves its Run3ModelGen workspaces behind:
``iteration_NNN/worker_*`` and ``iteration_NNN/retry_*``, holding the SLHA
spectra, SPheno logs and snakemake metadata of every model generated that
iteration. On a benchmark run that is **145 MB and about 6,200 files per
iteration**, against **14 MB and 28 files** of everything worth keeping
(the two checkpoints, the training logs, selected_points.csv, plots).

Nothing in the loop ever removes it, so a 40-iteration run is 5.8 GB and
227,000 files. Extrapolated over the 200-iteration campaign, 41,600 iterations
across both targets, that is roughly 6 PB and 260 million files on a shared
filesystem with 7.9 PB free and no inode quota to stop it.

Why this PACKS rather than deletes
----------------------------------
``scripts/archive_runs.sh`` calls this subset "worker_*/retry_* simulation debris
(regenerable)" and declines to archive it. That comment is about what could in
principle be re-simulated, NOT about whether anything reads it, and four
analyses do (checked 2026-08-21):

  scripts/composition_fractions.py       iteration_*/**/ntuple.*.root
  scripts/best_analysis_arms.py          worker_*/scan/SModelS/*.slha.py
                                         retry_*/worker_*/scan/SModelS/*.slha.py
  scripts/best_analysis_from_smodels.py  the same *.slha.py, via rglob
  scripts/mode_switch_diagnostic.py      the same *.slha.py

composition_fractions.py is explicit that the loss would be permanent: "the
pooled state.pt holds only inputs and Omega, so these files are the only route
to the same composition definition the LSP-type figure uses", and "the direct
workers alone hold only ~45% of the evaluated points, and the retries are not a
random subset of them".

So the default mode is ``tar``: each completed iteration's workspaces are packed
into one ``debris.tar`` beside them and the trees removed. That turns ~6,200
inodes into 1 without losing a byte, and it is reversible with ``tar -xf``. It is
the same trick archive_runs.sh uses for the home filesystem's 261,120-file quota.
The consumers above need the tar expanded before they run.

``--mode delete`` is available but it is destructive and irreversible; it will
break all four scripts above for the pruned iterations. It requires
``--i-know-this-deletes-analysis-inputs``.

What it will not touch
----------------------
The highest ``--keep-last`` iteration directories of each run. The
in-progress iteration is always the highest-numbered one, and a killed
iteration may be retried against its existing workspace, so the newest ones are
left alone. Everything below them belongs to an iteration whose result is
already in ``state.pt``.

Usage
-----
    python scripts/prune_run_debris.py                       # dry run, reports only
    python scripts/prune_run_debris.py --apply               # pack into debris.tar
    python scripts/prune_run_debris.py --apply --mode delete \
        --i-know-this-deletes-analysis-inputs                # destructive
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import click


def _debris(iter_dir: Path):
    return sorted([p for p in iter_dir.glob("worker_*") if p.is_dir()] +
                  [p for p in iter_dir.glob("retry_*") if p.is_dir()])


def _measure(paths):
    """(bytes, files) without following symlinks; cheap enough at this scale."""
    nb = nf = 0
    for p in paths:
        for root, _dirs, files in os.walk(p):
            for f in files:
                try:
                    nb += os.stat(os.path.join(root, f), follow_symlinks=False).st_size
                    nf += 1
                except OSError:
                    pass
    return nb, nf


@click.command()
@click.option("--runs", default="/ptmp/jwuerzin/output/*", show_default=True,
              help="Glob over run directories.")
@click.option("--keep-last", default=3, show_default=True,
              help="Leave the debris of this many newest iterations alone. The "
                   "in-progress iteration is the highest-numbered one, and a "
                   "retried iteration may reuse its workspace.")
@click.option("--max-dirs", default=0, show_default=True,
              help="Stop after removing this many debris directories (0 = no cap). "
                   "Deleting thousands of small files on GPFS is slow, so a cap "
                   "keeps one invocation inside a wall clock.")
@click.option("--deadline-min", default=0, show_default=True,
              help="Stop after this many minutes (0 = no deadline).")
@click.option("--min-age-hours", default=6.0, show_default=True,
              help="Skip debris modified more recently than this, as a second "
                   "guard against touching live work.")
@click.option("--mode", default="tar", show_default=True,
              type=click.Choice(["tar", "delete"]),
              help="tar packs each iteration's workspaces into debris.tar beside "
                   "them and removes the trees: same bytes, one inode instead of "
                   "~6,200, reversible with tar -xf. delete is irreversible and "
                   "breaks composition_fractions.py, best_analysis_arms.py, "
                   "best_analysis_from_smodels.py and mode_switch_diagnostic.py "
                   "for the pruned iterations.")
@click.option("--i-know-this-deletes-analysis-inputs", "confirmed_delete",
              is_flag=True, default=False,
              help="Required for --mode delete.")
@click.option("--apply/--dry-run", default=False,
              help="Dry run by default: nothing is written or removed and the "
                   "reclaimable total is reported.")
def main(runs, keep_last, max_dirs, deadline_min, min_age_hours, mode,
         confirmed_delete, apply):
    if mode == "delete" and apply and not confirmed_delete:
        raise click.UsageError(
            "--mode delete destroys the inputs of composition_fractions.py, "
            "best_analysis_arms.py, best_analysis_from_smodels.py and "
            "mode_switch_diagnostic.py, and composition_fractions.py states "
            "that state.pt cannot substitute for them. Pass "
            "--i-know-this-deletes-analysis-inputs if that is genuinely "
            "intended, or use the default --mode tar, which keeps every byte.")
    t0 = time.time()
    deadline = t0 + deadline_min * 60 if deadline_min else None
    now = time.time()
    tot_b = tot_f = tot_d = 0
    n_runs = 0
    stopped = ""

    for run in sorted(glob.glob(runs)):
        run = Path(run)
        if not run.is_dir():
            continue
        iters = sorted(run.glob("iteration_[0-9][0-9][0-9]"))
        if len(iters) <= keep_last:
            continue
        n_runs += 1
        run_b = run_f = run_d = 0
        for it in iters[:-keep_last] if keep_last else iters:
            paths = _debris(it)
            if not paths:
                continue
            if (it / "debris.tar").exists():
                continue        # already packed on an earlier pass
            if min_age_hours:
                try:
                    if now - it.stat().st_mtime < min_age_hours * 3600:
                        continue
                except OSError:
                    continue
            b, f = _measure(paths)
            run_b += b
            run_f += f
            run_d += len(paths)
            if apply:
                if mode == "tar":
                    # Pack first, verify the archive opens, and only then remove
                    # the trees. A tar that cannot be read back is worse than no
                    # tar at all, so the removal is gated on the read.
                    tarball = it / "debris.tar"
                    try:
                        with tarfile.open(tarball, "w") as tf:
                            for p in paths:
                                tf.add(p, arcname=p.name)
                        with tarfile.open(tarball, "r") as tf:
                            if not tf.getnames():
                                raise OSError("empty archive")
                    except Exception as exc:                # noqa: BLE001
                        click.echo(f"  [skip] {it}: pack failed "
                                   f"({type(exc).__name__}: {exc})")
                        tarball.unlink(missing_ok=True)
                        continue
                    for p in paths:
                        shutil.rmtree(p, ignore_errors=True)
                else:
                    for p in paths:
                        shutil.rmtree(p, ignore_errors=True)
            tot_d += len(paths)
            if max_dirs and tot_d >= max_dirs:
                stopped = f"--max-dirs {max_dirs} reached"
                break
            if deadline and time.time() > deadline:
                stopped = f"--deadline-min {deadline_min} reached"
                break
        if run_f:
            click.echo(f"  {run.name[:78]:78s} "
                       f"{run_b / 2**30:7.2f} GiB  {run_f:8,d} files")
        tot_b += run_b
        tot_f += run_f
        if stopped:
            break

    verb = ("packed" if mode == "tar" else "deleted") if apply \
        else "reclaimable"
    click.echo(f"\n[prune] {n_runs} run(s) examined, {tot_d} debris "
               f"director{'y' if tot_d == 1 else 'ies'}")
    click.echo(f"[prune] {verb}: {tot_b / 2**40:.3f} TiB, {tot_f:,d} files "
               f"in {time.time() - t0:.0f}s")
    if stopped:
        click.echo(f"[prune] stopped early: {stopped}; run again to continue")
    if not apply and tot_f:
        click.echo("[prune] DRY RUN: nothing was written or removed. "
                   "Add --apply to pack into debris.tar.")


if __name__ == "__main__":
    main()
