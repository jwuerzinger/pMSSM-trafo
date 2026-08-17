"""Build a sweep-manifest CSV for runs that were not launched by the sweep script.

The extended-budget and large-batch probes were submitted directly through the
per-model scripts, so they have no rows in sweep_manifest.csv. Every downstream
analysis (composition, MCMC diagnostics, UQ) discovers work through a manifest,
so those probes are invisible to all of it. This reconstructs the rows from the
directories on disk.

For a resumed run, only the MOST ADVANCED directory is emitted. The continuation
inherits the entire history in its state.pt and the compute parser follows the
base->continuation chain to locate the early iteration directories, so that one
row already carries the full trajectory. Emitting the base as a second row of the
same cell makes it look like a second seed, and the trajectory is then truncated
to the shorter of the two, i.e. back to 40 iterations, silently hiding the very
extension these probes exist to measure.

Usage:
    python scripts/build_probe_manifest.py --pattern '*ext160*' \\
        --out /ptmp/jwuerzin/analysis/probe_extended/manifest.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import click

# Longest first: dnn_match_trafo must win over dnn, and the _20k / _laplace
# tagged variants over their untagged parents.
MODELS = ["dnn_match_trafo_laplace", "dnn_match_trafo", "transformer_laplace",
          "transformer_oracle", "deep_gp_oracle", "dnn_laplace", "transformer",
          "exact_gp", "deep_gp", "tabpfn", "dnn"]
STRATEGIES = ["top_k_tol_only", "entropy_batch", "top_k"]
WARMS = ["warm", "cold", "tabpfn"]


def _eat(rest: str, options: list[str], max_tags: int = 2):
    """Match one of ``options`` at the start of ``rest``, skipping short tags.

    Probe directories carry a free-form tag whose position is not fixed: the
    large-batch runs put it between the strategy and the warm mode
    (transformer_top_k_20k_cold_seed1) while other conventions put it after the
    model. Rather than hard-code a slot, skip up to ``max_tags`` underscore
    tokens looking for a known keyword. Returns (match, remainder) or None.
    """
    for _ in range(max_tags + 1):
        hit = next((o for o in options if rest.startswith(o + "_")), None)
        if hit is not None:
            return hit, rest[len(hit) + 1:]
        head, sep, rest = rest.partition("_")
        if not sep or head in ("",):
            return None
    return None


def _parse(name: str):
    """('model', 'strategy', 'warm', seed) from a run-directory name, or None.

    The tag itself is dropped: each probe set gets its own manifest, so the
    plain model name is what keeps these rows keyed the same way as the main
    sweep's and therefore comparable to it.
    """
    if not name.startswith("active_learning_"):
        return None
    rest = name[len("active_learning_"):]
    got = _eat(rest, MODELS, max_tags=0)
    if got is None:
        return None
    model, rest = got
    got = _eat(rest, STRATEGIES)
    if got is None:
        return None
    strat, rest = got
    got = _eat(rest, WARMS)
    if got is None:
        return None
    warm, rest = got
    if not rest.startswith("seed"):
        return None
    seed_str = rest[4:].split("_")[0]
    if not seed_str.isdigit():
        return None
    return model, strat, warm, int(seed_str)


def _from_manifest(src, out, sweep_id, seeds, min_iterations):
    """Emit the subset of an existing manifest that the extension advanced.

    The ExpR 160-iteration probes were submitted with ``AL_RESUME_TO`` against
    the seed-1 runs of the main sweep, so they occupy the SAME directories. No
    directory glob can tell them from their 40-iteration siblings; the only
    signal on disk is the iteration count in state.pt. Selecting by seed and
    then by length gives a probe manifest that the run-set figures can use
    without disturbing the main one.
    """
    want = {int(s) for s in seeds.split(",") if s.strip()} if seeds else None
    rows = []
    for r in csv.DictReader(open(src)):
        if want is not None and int(r.get("seed") or -1) not in want:
            continue
        d = Path(r["expected_run_dir"])
        if not (d / "state.pt").exists():
            continue
        n_iter = 0
        if min_iterations:
            try:
                import torch  # noqa: PLC0415
                n_iter = len(list(torch.load(d / "state.pt", weights_only=False,
                                             map_location="cpu")
                                  .get("al_n_train") or []))
            except Exception:                                   # noqa: BLE001
                n_iter = 0
            if n_iter < min_iterations:
                continue
        rows.append({**{k: r.get(k, "") for k in
                        ("submit_time", "model", "strategy", "warm_start",
                         "seed", "job_id", "expected_run_dir", "slurm_log")},
                     "sweep_id": sweep_id, "status": "completed"})
        click.echo(f"[probe-manifest] {r['model']}/{r['strategy']}/"
                   f"{r['warm_start']}/seed{r['seed']}: {n_iter or '?'} iterations")
    if not rows:
        raise click.ClickException(
            f"no rows of {src} matched seeds={seeds!r} with "
            f">= {min_iterations} iterations")
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    order = ["sweep_id", "submit_time", "model", "strategy", "warm_start",
             "seed", "job_id", "expected_run_dir", "status", "slurm_log"]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=order)
        w.writeheader()
        w.writerows({k: r[k] for k in order} for r in rows)
    click.echo(f"[probe-manifest] wrote {len(rows)} rows to {p}")


@click.command()
@click.option("--output-root", default="/ptmp/jwuerzin/output", show_default=True)
@click.option("--pattern", default="",
              help="Glob against directory names, e.g. '*ext160*' or '*_20k_*'. "
                   "Required unless --from-manifest is given.")
@click.option("--out", required=True, help="Manifest CSV to write.")
@click.option("--sweep-id", default="probe", show_default=True)
@click.option("--from-manifest", default="",
              help="Select rows from an existing manifest instead of globbing "
                   "directories. Needed when the extended runs RESUME IN PLACE "
                   "and so share their 40-iteration siblings' directory names, "
                   "which no glob can separate: the ExpR probes are the seed-1 "
                   "rows of the main sweep. Combine with --seeds.")
@click.option("--seeds", default="",
              help="Comma list of seeds to keep from --from-manifest.")
@click.option("--min-iterations", default=0, type=int,
              help="With --from-manifest, keep only runs whose state.pt has at "
                   "least this many iterations, so unextended cells drop out.")
def main(output_root, pattern, out, sweep_id, from_manifest, seeds,
         min_iterations):
    if from_manifest:
        return _from_manifest(from_manifest, out, sweep_id, seeds,
                              min_iterations)
    if not pattern:
        raise click.UsageError("give --pattern, or --from-manifest to select "
                               "rows of an existing manifest")
    root = Path(output_root)
    candidates: dict[tuple, tuple[int, Path]] = {}
    for d in sorted(root.glob(pattern)):
        if not d.is_dir() or not (d / "state.pt").exists():
            continue
        parsed = _parse(d.name)
        if parsed is None:
            click.echo(f"[probe-manifest] unparsed, skipping: {d.name}", err=True)
            continue
        # Keep only the MOST ADVANCED directory per cell. A resumed run's
        # continuation inherits the whole history in its state.pt, and
        # plot_compute_vs_dataset follows the base->continuation chain to find
        # the early iteration directories, so the continuation alone carries the
        # full trajectory. Emitting the base as well would make it a second
        # "seed" of the same cell, and the trajectory would then be truncated to
        # the shorter of the two, i.e. back to 40 iterations, hiding the very
        # extension the probe exists to measure.
        try:
            import torch  # noqa: PLC0415
            n_iter = len(list(torch.load(d / "state.pt", weights_only=False,
                                         map_location="cpu").get("al_n_train") or []))
        except Exception:                                   # noqa: BLE001
            n_iter = 0
        key = parsed
        if key not in candidates or n_iter > candidates[key][0]:
            candidates[key] = (n_iter, d)

    rows = []
    for (model, strat, warm, seed), (n_iter, d) in sorted(candidates.items()):
        click.echo(f"[probe-manifest] {model}/{strat}/{warm}/seed{seed}: "
                   f"{n_iter} iterations from {d.name}")
        rows.append({
            "sweep_id": sweep_id, "submit_time": "", "model": model,
            "strategy": strat, "warm_start": warm, "seed": seed,
            "job_id": "", "expected_run_dir": str(d),
            "status": "completed", "slurm_log": "",
        })
    if not rows:
        raise click.ClickException(f"no runs matched {pattern!r} under {root}")
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    click.echo(f"[probe-manifest] wrote {len(rows)} rows to {p}")
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["model"], r["strategy"], r["warm_start"]), []).append(r["seed"])
    for k, v in sorted(cells.items()):
        click.echo(f"    {k[0]:<26} {k[1]:<15} {k[2]:<6} seeds={sorted(v)}")


if __name__ == "__main__":
    main()
