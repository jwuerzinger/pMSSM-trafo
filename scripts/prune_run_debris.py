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

# The in-loop packer. Using it here rather than a second implementation means a
# retrofitted run ends up indistinguishable from a campaign run: the ntuples
# are moved out to iteration_NNN/ntuples/ before the tar is written, so
# composition_fractions.py still finds them, and the SModelS winners are
# summarised to smodels_best_analysis.json first, so best_analysis_arms.py,
# best_analysis_from_smodels.py and mode_switch_diagnostic.py can read the
# record through load_iteration_smodels() without expanding the tar. Nothing
# that reads a run needs an untar afterwards.
import logging
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pmssm.iteration_housekeeping import (            # noqa: E402
    pack_iteration_debris, summarise_smodels)


def _debris(iter_dir: Path):
    return sorted([p for p in iter_dir.glob("worker_*") if p.is_dir()] +
                  [p for p in iter_dir.glob("retry_*") if p.is_dir()])


def _recover_interrupted(it: Path, paths, apply: bool) -> bool:
    """Handle debris.tar + leftover trees. True means "pack this iteration now"."""
    tarball = it / "debris.tar"
    on_disk = {str(Path(r, f).relative_to(it))
               for p in paths for r, _d, fs in os.walk(p) for f in fs}
    try:
        with tarfile.open(tarball, "r") as tf:
            inside = {m.name for m in tf.getmembers() if m.isfile()}
        complete = bool(inside) and on_disk <= inside
    except Exception:                                          # noqa: BLE001
        complete = False
    if complete:
        click.echo(f"  [recover] {it}: tar complete, finishing the removal")
        if apply:
            for pth in paths:
                shutil.rmtree(pth, ignore_errors=False)
        return False
    aside = it / f"debris.tar.interrupted_{int(time.time())}"
    click.echo(f"  [recover] {it}: tar truncated, set aside as {aside.name}; "
               f"re-packing ({len(on_disk):,} files still on disk)")
    if apply:
        tarball.rename(aside)
    return apply


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
@click.option("--runs", multiple=True, default=["/ptmp/jwuerzin/output/*"],
              show_default=True,
              help="Glob over run directories; repeatable, so a batch can be "
                   "given as explicit directories.")
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

    log = logging.getLogger("prune")
    for run in sorted(set(r for g in runs for r in glob.glob(g))):
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
                # A tar beside leftover trees means an earlier pass was killed
                # between writing it and removing them. Do not trust it either
                # way without reading it: if it holds every file still on disk
                # the earlier pass had already verified it and only the removal
                # is missing; otherwise it was cut off mid-write, so it is set
                # aside (renamed, never deleted) and the iteration is packed
                # afresh. Skipping it, as this did before, left such an
                # iteration unpacked for good.
                if not _recover_interrupted(it, paths, apply):
                    continue
                recovered = True    # the rename just touched the dir's mtime
            else:
                recovered = False
            if min_age_hours and not recovered:
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
                    # SModelS summary first: it reads the *.slha.py files that
                    # are about to move into the tar. A DMRD run has none and
                    # the call returns None, which is fine.
                    summarise_smodels(it, log, true_value=1.0, tolerance=0.10)
                    # Then the verified pack: counts the files on disk, writes
                    # the tar, reads it back and compares the member count, and
                    # only then removes the trees. On any mismatch the tar is
                    # unlinked and the trees stay. Nothing is removed that has
                    # not been read back out of its archive.
                    info = pack_iteration_debris(it, log)
                    if info is None:
                        click.echo(f"  [keep] {it}: NOT removed (pack skipped "
                                   f"or failed; see the housekeeping log line)")
                        continue
                    # run_f was accumulated from the pre-rescue measure above,
                    # which still counted the ntuples that were then moved out
                    # loose. Report what actually went into the archive.
                    run_f += info["files"] - f
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
