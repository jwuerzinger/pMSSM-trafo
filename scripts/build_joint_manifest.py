"""Manifest selecting the most ADVANCED run dir per (cell, seed).

The extended-budget probes are in-place resumes, so a resumed run's trajectory
arrays already run 1..N contiguously: one manifest pointing at the longest dir
per seed therefore yields a single joint curve per seed, with no stitching. Read
with --min-seeds 1 the plotting scripts then average over whatever seeds exist
at each iteration, so the multi-seed band narrows where the extension continues
alone instead of the curve stopping at the benchmark horizon.

Candidate dirs per (cell, seed) are every directory whose name starts with the
cell's base and carries that seed, across output roots. The archived
output_benchmark40 copies are excluded: they are the frozen 40-iteration
reference the benchmark analysis reads and must never be extended.
"""
from __future__ import annotations
import csv, glob, os, re, sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcmc_diagnostics import picks_with_tag  # noqa: E402


def _iters(d: str) -> int:
    p = os.path.join(d, "state.pt")
    if not os.path.exists(p):
        return -1
    try:
        s = torch.load(p, map_location="cpu", weights_only=False)
        return len(list(s.get("al_n_train") or []))
    except Exception:
        return -1


@click.command()
@click.option("--manifest", required=True, help="Sweep manifest to take cells from.")
@click.option("--out", required=True, help="Joint manifest to write.")
@click.option("--model-tag", default="", help="OUTPUT_TAG of a variant sweep, e.g. 'expr'.")
@click.option("--roots", default="/ptmp/jwuerzin/output",
              help="Comma-separated output roots to search.")
@click.option("--all-cells", is_flag=True, default=False,
              help="Keep every cell rather than the canonical pick per model.")
def main(manifest, out, model_tag, roots, all_cells):
    picks = picks_with_tag(model_tag)
    rows = list(csv.DictReader(open(manifest)))
    roots = [r.strip().rstrip("/") for r in roots.split(",") if r.strip()]

    want = {}
    for r in rows:
        key = (r["model"], r["strategy"], r["warm_start"])
        if not all_cells and picks.get(r["model"]) != (key[1], key[2]):
            continue
        want.setdefault(key, {})[int(r["seed"])] = r

    written = []
    for (model, strat, warm), seeds in sorted(want.items()):
        for seed in sorted(seeds):
            row = seeds[seed]
            # Strip the sweep suffix to get this cell's dir base, then look for
            # every dir of this cell and seed under each root, longest wins.
            stem = re.sub(r"_seed\d+_.*$", "", os.path.basename(row["expected_run_dir"]))
            cands = []
            for root in roots:
                cands += glob.glob(f"{root}/{stem}_seed{seed}_*")
                cands += glob.glob(f"{root}/{stem}_seed{seed}")
            cands = [c for c in cands if "output_benchmark40" not in c]
            scored = sorted(((_iters(c), c) for c in set(cands)), reverse=True)
            scored = [(n, c) for n, c in scored if n > 0]
            if not scored:
                click.echo(f"[joint] {model}/{strat}/{warm} seed{seed}: no live dir", err=True)
                continue
            n, best = scored[0]
            out_row = dict(row)
            out_row["expected_run_dir"] = best
            out_row["sweep_id"] = "joint"
            out_row["status"] = "completed"
            written.append(out_row)
            extra = f"   (of {len(scored)} candidates)" if len(scored) > 1 else ""
            click.echo(f"[joint] {model:22s} {strat:14s} {warm:6s} seed{seed}: "
                       f"{n:4d} iters  {os.path.basename(best)}{extra}")

    if not written:
        raise click.ClickException("no rows selected")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(written)
    click.echo(f"[joint] wrote {out}: {len(written)} runs, "
               f"{len({(r['model'],r['strategy'],r['warm_start']) for r in written})} cells")


if __name__ == "__main__":
    main()
