"""Keep every campaign cell advancing towards 200 iterations, unattended.

Why this exists
---------------
No fresh active-learning run reaches 200 iterations inside the 24 h wall clock:
the measured cost is 2-5 GPU-node-days per run, and the cost per iteration grows
with the labelled set. Every cell therefore needs to be resumed several times.
``AL_RESUME_TO`` in ``slurm/submit_al_bundled.sh`` does the resuming and is
idempotent by construction (it derives each seed's ``n_additional`` from that
seed's own ``state.pt`` and skips seeds already at the target), so the only
missing piece is something that keeps calling it. That is this script, driven by
``slurm/submit_campaign_chase.sh`` on a six-hourly self-resubmitting cycle.

What it will not do
-------------------
Submit a second job for a cell that already has one queued or running. Two jobs
resuming one run directory would interleave writes to the same ``state.pt`` and
corrupt it; the ``squeue`` check keyed on the ``c200_`` job name is the only
thing preventing that, so it is checked before anything else.

Which cells
-----------
Two sources, unioned:

1. the campaign's own cells, from the CSV ``submit_campaign_200.sh`` wrote;
2. any cell in either sweep manifest whose longest seed is past 40 iterations
   but short of the target. That is exactly the set the ``b200``/``b200e``
   continuation bundles are advancing, without having to guess which cells those
   twelve jobs chose: once a bundle lifts a cell past the 40-iteration benchmark
   horizon, the cell becomes visible here and is topped up from then on.

Usage
-----
    python scripts/campaign_chase.py --cells <csv> --dry-run
    python scripts/campaign_chase.py --cells <csv> --queue-cap 24
"""
from __future__ import annotations

import csv
import glob
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import click

REPO = Path(__file__).resolve().parent.parent
EXPR_POOL = "/ptmp/jwuerzin/data/260804"
DMRD_POOL = "/ptmp/jwuerzin/data/18387358"
MANIFESTS = {"ExpR": "/ptmp/jwuerzin/analysis/expr_runs/sweep_manifest.csv",
             "DMRD": "/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv"}
# Variant families that are not part of this campaign and must never be resumed
# by it: the oracle runs consume the MCMC posterior as their candidate source
# and the laplace runs swap the acquisition uncertainty, so extending them would
# silently mix two different experiments into one cell.
SKIP_MODEL_SUFFIXES = ("_oracle", "_laplace")
NEURAL = {"transformer", "dnn", "dnn_match_trafo"}
GP = {"deep_gp", "exact_gp", "laplace_gpc"}


def _iters(run_dir: str) -> int:
    """Iterations completed, counted the way AL_RESUME_TO counts them.

    ``state.pt`` is authoritative, but loading torch here would cost seconds per
    seed across ~200 seeds; the ``iteration_NNN`` directories are written one per
    completed iteration and agree with it, so they are counted instead. The strict
    three-digit glob matters: a bare ``iteration_*`` also matches
    ``iteration_metrics.png`` and over-counts by one.
    """
    return len(glob.glob(os.path.join(run_dir, "iteration_[0-9][0-9][0-9]")))


def bare_model(manifest_model: str) -> str:
    """`transformer_expr` -> `transformer`; the bundle dispatch wants the driver key."""
    return re.sub(r"_expr$", "", manifest_model)


def family(model: str) -> str:
    if model in GP:
        return "gp"
    if model == "tabpfn":
        return "tabpfn"
    return "neural"


def derive_extra(target: str, model: str, strategy: str) -> str:
    """The flag set this cell must be resumed with.

    Mirrors ``common_for`` in submit_campaign_200.sh. It has to be reconstructed
    rather than remembered because the continuation cells were submitted by
    other scripts, and because resuming with the wrong flags is not a soft
    failure: only the transformer driver refuses a configuration change, so a
    DNN or GP would silently continue under a different strategy or head.
    """
    pool = EXPR_POOL if target == "ExpR" else DMRD_POOL
    tv = "1.0" if target == "ExpR" else "0.12"
    mcmc = " --no-mcmc-eval" if target == "ExpR" else ""
    fam = family(model)
    if fam == "gp":
        common = f"--target {target}{mcmc} --data-dir {pool}"
    elif fam == "tabpfn":
        common = f"--target {target} --target-value {tv}{mcmc} --data-dir {pool}"
    else:
        common = (f"--target {target} --target-value {tv}{mcmc} "
                  f"--y-transform log --data-dir {pool}")
    if strategy in ("bald", "cls_entropy"):
        if model == "exact_gp":
            common += (" --model-type laplace_gpc --head classification"
                       " --epochs 3000 --patience 200 --learning-rate 1e-2")
        else:
            common += " --head classification"
    return common


def queued_names() -> set[str]:
    out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "jwuerzin"),
                          "-o", "%j"], capture_output=True, text=True).stdout
    return {l.strip() for l in out.splitlines() if l.strip()}


def n_my_jobs() -> int:
    out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "jwuerzin"),
                          "-o", "%i"], capture_output=True, text=True).stdout
    return len([l for l in out.splitlines() if l.strip()])


def job_name(target: str, model: str, strategy: str) -> str:
    short = {"tol_only_random": "tol", "cls_entropy": "clsent", "bald": "bald",
             "entropy_batch": "ent", "top_k": "topk", "top_k_tol_only": "tktol"}
    t = "e" if target == "ExpR" else "d"
    return f"c200_{t}_{bare_model(model)}_{short.get(strategy, strategy)}"


def cells_from_manifests(target: str, seeds: list[str], resume_to: int,
                         min_iters: int):
    """Cells already past the benchmark horizon: the continuation bundles' cells."""
    path = MANIFESTS[target]
    if not os.path.exists(path):
        return []
    by_cell: dict[tuple, dict] = {}
    for r in csv.DictReader(open(path)):
        model = r.get("model") or ""
        if any(model.endswith(s) for s in SKIP_MODEL_SUFFIXES):
            continue
        d = r.get("expected_run_dir") or ""
        if not d:
            continue
        key = (model, r["strategy"], r["warm_start"], r["sweep_id"])
        base = re.sub(r"_seed\d+_.*$", "", d)
        e = by_cell.setdefault(key, {"base": base, "seeds": {}})
        e["seeds"][r["seed"]] = _iters(d)
    out = []
    for (model, strategy, warm, sweep), e in by_cell.items():
        got = e["seeds"]
        if not got:
            continue
        if max(got.values()) <= min_iters:
            continue                      # never extended: not a b200 cell
        if min(got.get(s, 0) for s in seeds) >= resume_to:
            continue                      # done
        out.append({"target": target, "model": model, "strategy": strategy,
                    "warm": warm, "sweep_id": sweep, "base": e["base"],
                    "iters": got, "origin": "continuation"})
    return out


@click.command()
@click.option("--cells", default="", help="Campaign cells CSV from submit_campaign_200.sh.")
@click.option("--resume-to", default=200, show_default=True)
@click.option("--seeds", default="1,2,3,4,5", show_default=True)
@click.option("--per-wake", default=12, show_default=True,
              help="Most jobs to ADD in one wake. This is the real throttle: it "
                   "stops a single pass from dumping the whole campaign into the "
                   "queue at once, while still letting every cell be resumed "
                   "within a wake or two.")
@click.option("--queue-cap", default=250, show_default=True,
              help="Hard ceiling on my total queued+running jobs, as a runaway "
                   "guard against the association's MaxSubmit of 300. It must "
                   "stay ABOVE the campaign's standing depth (one job per cell, "
                   "so ~50): set to 24 it silently blocked every resume, because "
                   "a cell whose job had just timed out was counted against a "
                   "cap the other cells' queued jobs had already exhausted.")
@click.option("--min-iters-continuation", default=40, show_default=True,
              help="A manifest cell counts as a continuation target only past this.")
@click.option("--partition", default="apu", show_default=True)
@click.option("--dry-run/--submit", default=False)
def main(cells, resume_to, seeds, per_wake, queue_cap,
         min_iters_continuation, partition, dry_run):
    seed_list = [s for s in seeds.split(",") if s]
    running = queued_names()
    n_jobs = n_my_jobs()
    click.echo(f"[chase] {n_jobs} of my jobs queued/running; will add at most "
               f"{per_wake} this wake (hard ceiling {queue_cap})")

    work: list[dict] = []
    if cells and os.path.exists(cells):
        for r in csv.DictReader(open(cells)):
            tag = "_expr" if r["target"] == "ExpR" else ""
            base = (f"/ptmp/jwuerzin/output/active_learning_{r['model']}{tag}"
                    f"_{r['strategy']}_{r['warm']}")
            got = {s: _iters(f"{base}_seed{s}_{r['campaign_id']}") for s in seed_list}
            if min(got.values()) >= resume_to:
                continue
            work.append({"target": r["target"], "model": r["model"] + tag,
                         "strategy": r["strategy"], "warm": r["warm"],
                         "sweep_id": r["campaign_id"], "base": base,
                         "iters": got, "origin": "campaign"})
    # The b200/b200e bundles queued on 2026-08-18 are already advancing exactly
    # the continuation cells found below, under their own job names. Their names
    # do not start with c200_, so the per-cell busy check cannot see them, and
    # two jobs resuming one run directory would interleave writes to the same
    # state.pt. Stand off entirely while any of them is alive; once they drain,
    # this takes over their cells.
    # Only the names that actually target continuation cells count here. The
    # generic al_bundled name does not: the campaign's own bundles are renamed to
    # c200_* at submit time so the per-cell check below sees them, and the older
    # al_bundled jobs are fresh 40-iteration sweep cells, which by definition
    # cannot be continuation cells (those need a seed past 40 already).
    legacy = sorted(n for n in running
                    if n.startswith("b200") or n.startswith("ext160"))
    if legacy:
        click.echo(f"[chase] {len(legacy)} legacy continuation job(s) alive "
                   f"({', '.join(legacy[:4])}{'...' if len(legacy) > 4 else ''}); "
                   "leaving every continuation cell to them this round")
    else:
        for t in MANIFESTS:
            work += cells_from_manifests(t, seed_list, resume_to,
                                         min_iters_continuation)

    # The CSV and the manifest scan can name the same cell: a continuation cell
    # listed explicitly is also found by the >min_iters scan. Two entries would
    # mean two sbatch calls in one wake, since the busy check reads squeue once
    # at the start and cannot see a job this pass just created. Deduplicate by
    # job name, keeping the CSV entry, whose sweep id is stated rather than
    # inferred.
    seen_names: dict[str, dict] = {}
    for c in work:
        n = job_name(c["target"], c["model"], c["strategy"])
        if n not in seen_names or c.get("origin") == "campaign":
            seen_names[n] = c
    if len(seen_names) != len(work):
        click.echo(f"[chase] {len(work) - len(seen_names)} duplicate cell(s) "
                   "collapsed (listed and discovered)")
    work = list(seen_names.values())

    submitted = skipped_busy = skipped_cap = 0
    for c in work:
        name = job_name(c["target"], c["model"], c["strategy"])
        prog = ",".join(str(c["iters"].get(s, 0)) for s in seed_list)
        if name in running:
            click.echo(f"  [busy] {name:34s} seeds({prog}) already queued")
            skipped_busy += 1
            continue
        if submitted >= per_wake or n_jobs >= queue_cap:
            skipped_cap += 1
            continue
        extra = derive_extra(c["target"], bare_model(c["model"]), c["strategy"])
        env = dict(os.environ)
        env.update({
            "AL_MODEL": bare_model(c["model"]),
            "AL_STRATEGY": c["strategy"],
            "AL_WARM": c["warm"],
            "AL_SEEDS": ",".join(seed_list),
            "AL_OUTPUT_BASE": c["base"],
            "AL_SWEEP_ID": c["sweep_id"],
            "AL_EXTRA_ARGS_BASE": extra,
            "AL_RESUME_TO": str(resume_to),
            "AL_START_MISSING": "1",
        })
        env.pop("AL_DATA_DIR", None)   # never let a pool leak between targets
        mem = "64G" if bare_model(c["model"]) == "tabpfn" else "128G"
        cmd = ["sbatch", "--parsable", f"--job-name={name}",
               f"--partition={partition}", f"--nodes={len(seed_list)}",
               "--gres=gpu:2", f"--mem={mem}", "--exclusive", "--export=ALL",
               "slurm/submit_al_bundled.sh"]
        if dry_run:
            click.echo(f"  [dry ] {name:34s} seeds({prog}) -> +{resume_to}\n"
                       f"         base={c['base']}_seed*_{c['sweep_id']}\n"
                       f"         extra={extra}")
            submitted += 1
            continue
        p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            click.echo(f"  [FAIL] {name}: {p.stderr.strip()[:200]}")
            continue
        jid = p.stdout.strip()
        click.echo(f"  [sub ] {jid} {name:34s} seeds({prog}) -> +{resume_to}")
        submitted += 1
        n_jobs += 1

    click.echo(f"[chase] {submitted} submitted, {skipped_busy} already running, "
               f"{skipped_cap} deferred to a later wake, "
               f"{len(work)} cells short of {resume_to}")
    if not work:
        click.echo("[chase] nothing left to do: every cell has reached the target")
    return 0


if __name__ == "__main__":
    main()
