"""Islands of the target region in the baseline scan, and who finds them first.

Motivation: an MCMC samples on-likelihood and a flat scan samples blind, so a
small disconnected piece of the target region is a thing both can miss, while
active learning screens ``n_candidates`` points per iteration (1,000,000 in the
benchmark runs) and only labels ``n_select`` of them (500), i.e. it searches
2000x more of the space than it pays for. This script tests that empirically:

  1. Identify the components. Bin the baseline scan's in-band models on an
     equal-occupancy quantile grid over the three axes that carry the
     constraint's information (IN_M_1, IN_M_2, IN_mu), keep cells holding at
     least ``--min-cell`` in-band models, and label connected components of
     those cells under 26-connectivity. Cells are the same construction the
     support figures use, so a component is a connected group of their cells.
  2. Report each component's size, its share of the whole scan (``p``, the
     per-call probability a blind draw lands in it) and its bounding box.
  3. Benchmark discovery. For every AL run and for the scan itself read in its
     stored order, find the budget at which the first in-band point inside each
     component appears, and how many such points it holds by the end. The
     random scan's own curve is the baseline: 1/p calls to first contact.

Resolution matters and is not a detail: component structure depends on the grid,
so ``--n-bins`` and ``--min-cell`` are swept by ``--sweep`` and the component
count is reported for each setting before any benchmark is run. A component that
appears only at one resolution is a binning artefact, not an island.

The 3-axis projection is deliberate. In the full 9-D space 12k in-band models
have a mean nearest-neighbour distance of 0.27 in box units, so single-linkage
connectivity there is set by the choice of threshold rather than by the data;
the three axes chosen carry essentially all of the mutual information with band
membership, and structure in them is physical (sign of mu, wino/bino/higgsino
regions) rather than an artefact of projection.

Usage:

    P=/ptmp/jwuerzin/pixi-envs/pytorch-conda-forge-2863954108128992291/envs/rocm/bin/python
    $P scripts/find_islands.py --manifest /ptmp/jwuerzin/analysis/joint/manifest_dmrd.csv \\
        --output-dir /ptmp/jwuerzin/analysis/joint/dmrd_islands --min-seeds 1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcmc_diagnostics import PARAM_ORDER, picks_with_tag  # noqa: E402
from plot_support_efficiency import _al_sequence, _discover  # noqa: E402

AXES = ["IN_M_1", "IN_M_2", "IN_mu"]


def _cell_index(X, edges, nb):
    """Flat cell index per row, and the per-axis bin triple."""
    b = np.stack([np.clip(np.digitize(X[:, j], e[1:-1]), 0, nb - 1)
                  for j, e in enumerate(edges)], axis=1)
    flat = (b[:, 0] * nb + b[:, 1]) * nb + b[:, 2]
    return flat, b


def _components(occupied, nb):
    """Label connected components of occupied cells under 26-connectivity.

    Iterative flood fill over the set of occupied flat indices; 26-connectivity
    (any shared corner counts) is the permissive choice, so it MERGES cells that
    a stricter face-only rule would split. That biases against finding islands,
    which is the conservative direction for the claim being tested.
    """
    occ = set(int(c) for c in occupied)
    off = [(dx, dy, dz)
           for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
           if (dx, dy, dz) != (0, 0, 0)]
    lab = {}
    comp = 0
    for seed in sorted(occ):
        if seed in lab:
            continue
        stack, comp = [seed], comp + 1
        lab[seed] = comp
        while stack:
            c = stack.pop()
            z = c % nb
            y = (c // nb) % nb
            x = c // (nb * nb)
            for dx, dy, dz in off:
                nx, ny, nz = x + dx, y + dy, z + dz
                if not (0 <= nx < nb and 0 <= ny < nb and 0 <= nz < nb):
                    continue
                n = (nx * nb + ny) * nb + nz
                if n in occ and n not in lab:
                    lab[n] = comp
                    stack.append(n)
    return lab, comp


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/joint/manifest_dmrd.csv",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/joint/dmrd_islands",
              show_default=True)
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358", show_default=True)
@click.option("--target", default="DMRD", show_default=True)
@click.option("--model-tag", default="", show_default=True)
@click.option("--include-status", default="completed,running,timeout,submitted",
              show_default=True)
@click.option("--all-cells/--picks-only", default=False, show_default=True)
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--n-bins", default=12, show_default=True,
              help="Bins per axis for the reported components.")
@click.option("--min-cell", default=20, show_default=True,
              help="In-band models a cell needs to count as occupied.")
@click.option("--sweep", default="8:5,8:20,12:5,12:20,16:20", show_default=True,
              help="n_bins:min_cell settings to report component counts for, so "
                   "a resolution-dependent 'island' is visible as such.")
@click.option("--min-seeds", default=1, show_default=True)
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True)
def main(manifest, output_dir, baseline_data_dir, target, model_tag,
         include_status, all_cells, tolerance, n_bins, min_cell, sweep,
         min_seeds, require_neutralino_lsp):
    from pmssm.config import TARGET_CONFIG
    from pmssm.data import load_pmssm_data

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    true_val = float(TARGET_CONFIG[target]["true_value"])
    ax_idx = [PARAM_ORDER.index(a) for a in AXES]

    X, Y = load_pmssm_data(n_datasets=-1, data_dir=baseline_data_dir,
                           target=target, plot_dir=str(out_dir),
                           require_neutralino_lsp=require_neutralino_lsp)
    X = np.asarray(X.numpy() if hasattr(X, "numpy") else X)[:, ax_idx]
    Y = np.asarray(Y.numpy() if hasattr(Y, "numpy") else Y, dtype=np.float64).ravel()
    n_total = len(Y)
    inb = np.abs(Y - true_val) / true_val < tolerance
    click.echo(f"[isl] target={target} pool {n_total:,} valid models, "
               f"{int(inb.sum()):,} in band ({inb.mean():.3%})")

    # ── resolution sweep, before committing to one grid ──────────────────────
    rows = []
    for spec in sweep.split(","):
        nb_s, mc_s = (int(v) for v in spec.split(":"))
        edges_s = [np.quantile(X[inb][:, j], np.linspace(0, 1, nb_s + 1))
                   for j in range(3)]
        for e in edges_s:
            e[0], e[-1] = -np.inf, np.inf
        flat_s, _ = _cell_index(X[inb], edges_s, nb_s)
        cnt_s = np.bincount(flat_s, minlength=nb_s ** 3)
        occ_s = np.where(cnt_s >= mc_s)[0]
        lab_s, ncomp_s = _components(occ_s, nb_s)
        sizes = {}
        for c, l in lab_s.items():
            sizes[l] = sizes.get(l, 0) + int(cnt_s[c])
        big = max(sizes.values()) if sizes else 0
        frac = big / max(1, sum(sizes.values()))
        rows.append(dict(n_bins=nb_s, min_cell=mc_s, cells=len(occ_s),
                         components=ncomp_s, largest_share=frac,
                         sizes=sorted(sizes.values(), reverse=True)))
        click.echo(f"[isl] {nb_s:>2} bins, min_cell {mc_s:>2}: "
                   f"{len(occ_s):>4} cells, {ncomp_s:>3} components, "
                   f"largest holds {frac:.1%} of the in-band models, "
                   f"sizes {sorted(sizes.values(), reverse=True)[:8]}")

    # ── the grid the benchmark uses ──────────────────────────────────────────
    edges = [np.quantile(X[inb][:, j], np.linspace(0, 1, n_bins + 1))
             for j in range(3)]
    for e in edges:
        e[0], e[-1] = -np.inf, np.inf
    flat_all, _ = _cell_index(X, edges, n_bins)
    cnt = np.bincount(flat_all[inb], minlength=n_bins ** 3)
    occ = np.where(cnt >= min_cell)[0]
    lab, ncomp = _components(occ, n_bins)
    cell_comp = -np.ones(n_bins ** 3, dtype=np.int64)
    for c, l in lab.items():
        cell_comp[c] = l
    comp_of_row = np.where(inb, cell_comp[flat_all], -1)

    comps = []
    for l in range(1, ncomp + 1):
        m = comp_of_row == l
        n_in = int(m.sum())
        box = [[float(X[m][:, j].min()), float(X[m][:, j].max())] for j in range(3)]
        comps.append(dict(component=l, n_inband=n_in, n_cells=int((cell_comp == l).sum()),
                          p=n_in / n_total, box=box,
                          first_row_in_pool=int(np.argmax(m)) if n_in else -1))
    comps.sort(key=lambda c: c["n_inband"], reverse=True)
    click.echo(f"[isl] benchmark grid {n_bins} bins, min_cell {min_cell}: "
               f"{ncomp} components over {len(occ)} cells")
    for c in comps:
        b = c["box"]
        click.echo(f"[isl]  comp {c['component']:>3}: {c['n_inband']:>6,} in-band "
                   f"({c['n_cells']:>3} cells) p={c['p']:.2e}  "
                   f"1/p={1 / c['p']:>10,.0f} calls  "
                   f"M1[{b[0][0]:.0f},{b[0][1]:.0f}] "
                   f"M2[{b[1][0]:.0f},{b[1][1]:.0f}] mu[{b[2][0]:.0f},{b[2][1]:.0f}]")

    # ── discovery benchmark ─────────────────────────────────────────────────
    # Random: the scan in its stored order, which is i.i.d., so the row index of
    # the first in-band hit in a component IS the number of calls it took.
    bench = {"random_scan": {}}
    for c in comps:
        l = c["component"]
        hit = np.where(comp_of_row == l)[0]
        bench["random_scan"][l] = dict(
            first_call=int(hit[0]) + 1 if len(hit) else None,
            expected_first_call=1 / c["p"],
            n_by_20k=int((hit < 20_000).sum()), total=len(hit))

    statuses = {s.strip() for s in include_status.split(",")}
    cells_found = _discover(manifest, statuses, picks_with_tag(model_tag),
                            all_cells, None)
    bench["al"] = {}
    for key, dirs in sorted(cells_found.items()):
        name = "/".join(key)
        per_seed = []
        for d in dirs:
            try:
                got = _al_sequence(d, ax_idx, require_neutralino_lsp)
            except Exception as exc:                          # noqa: BLE001
                click.echo(f"[isl]   skip {Path(d).name}: {exc}", err=True)
                continue
            if got is None:
                continue
            Xs, Ys = got
            f_s, _ = _cell_index(Xs, edges, n_bins)
            inb_s = np.abs(Ys - true_val) / true_val < tolerance
            comp_s = np.where(inb_s, cell_comp[f_s], -1)
            rec = {}
            for c in comps:
                l = c["component"]
                hit = np.where(comp_s == l)[0]
                rec[l] = dict(first_call=int(hit[0]) + 1 if len(hit) else None,
                              n_total=int(len(hit)), budget=int(len(Ys)))
            per_seed.append(rec)
        if not per_seed:
            continue
        bench["al"][name] = {"n_replicas": len(per_seed), "per_component": {}}
        for c in comps:
            l = c["component"]
            firsts = [r[l]["first_call"] for r in per_seed]
            found = [f for f in firsts if f is not None]
            counts = [r[l]["n_total"] for r in per_seed]
            bench["al"][name]["per_component"][l] = dict(
                found_in=len(found), of=len(firsts),
                median_first_call=float(np.median(found)) if found else None,
                min_first_call=min(found) if found else None,
                mean_points=float(np.mean(counts)))
        click.echo(f"[isl] {name}: " + "  ".join(
            f"c{c['component']}:"
            + (f"{bench['al'][name]['per_component'][c['component']]['median_first_call']:.0f}"
               if bench["al"][name]["per_component"][c["component"]]["found_in"] else "miss")
            + f"({bench['al'][name]['per_component'][c['component']]['found_in']}"
              f"/{bench['al'][name]['per_component'][c['component']]['of']})"
            for c in comps))

    payload = {"config": dict(target=target, tolerance=tolerance, axes=AXES,
                              n_bins=n_bins, min_cell=min_cell,
                              pool_rows=n_total, pool_inband=int(inb.sum()),
                              manifest=manifest),
               "resolution_sweep": rows, "components": comps, "benchmark": bench}
    (out_dir / f"islands_{target}.json").write_text(json.dumps(payload, indent=1))
    click.echo(f"[isl] wrote {out_dir / f'islands_{target}.json'}")


if __name__ == "__main__":
    main()
