"""Build a sweep-manifest CSV for runs that were not launched by the sweep script.

The extended-budget and large-batch probes were submitted directly through the
per-model scripts, so they have no rows in sweep_manifest.csv. Every downstream
analysis (composition, MCMC diagnostics, UQ) discovers work through a manifest,
so those probes are invisible to all of it. This reconstructs the rows from the
directories on disk.

Resumed runs are emitted as SEPARATE rows of the same (model, strategy, warm)
cell, one per directory. That is deliberate: a continuation directory holds only
the iterations it ran, so pooling the base and the continuation is what
reproduces the full labelled trajectory for cell-level statistics.

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


@click.command()
@click.option("--output-root", default="/ptmp/jwuerzin/output", show_default=True)
@click.option("--pattern", required=True,
              help="Glob against directory names, e.g. '*ext160*' or '*_20k_*'.")
@click.option("--out", required=True, help="Manifest CSV to write.")
@click.option("--sweep-id", default="probe", show_default=True)
def main(output_root, pattern, out, sweep_id):
    root = Path(output_root)
    rows = []
    for d in sorted(root.glob(pattern)):
        if not d.is_dir() or not (d / "state.pt").exists():
            continue
        parsed = _parse(d.name)
        if parsed is None:
            click.echo(f"[probe-manifest] unparsed, skipping: {d.name}", err=True)
            continue
        model, strat, warm, seed = parsed
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
