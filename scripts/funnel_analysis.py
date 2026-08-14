"""Low-|M1| (Z/h-funnel) region analysis: who samples it, who profits.

Reproduces the numbers quoted in the paper's corner-plot appendix about the
central |M1| enhancement of the input marginals (fig:input-overlay):

  * random pool  — volume share of |M1| < m1-cut, share among IN-BAND points
                   (the enrichment establishes the lobe as genuine), and the
                   |m_chi1| quantiles of the in-band low-|M1| population
                   (Z funnel ~45.6 GeV, h funnel ~62.5 GeV);
  * emcee MCMC   — posterior mass at |M1| < cut (fair-share comparison);
  * AL cells     — budget share at |M1| < cut, in-band rate there vs overall,
                   share of all hits, and the PER-ITERATION budget-share
                   trajectory (the discriminator: epistemic learners start
                   near fair share and migrate in as the smoother boundary
                   saturates; prior-followers camp there from iteration 1).

Usage:
    python scripts/funnel_analysis.py --require-neutralino-lsp
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import click
import numpy as np

import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyse_runs import PARAM_ORDER  # noqa: E402
from mcmc_diagnostics import DEFAULT_AL_PICKS, picks_with_tag  # noqa: E402

M1_COL = PARAM_ORDER.index("IN_M_1")


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358",
              show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True)
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True)
@click.option("--m1-cut", default=100.0, show_default=True,
              help="|M1| threshold defining the funnel region (GeV).")
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--n-mchi-files", default=40, show_default=True,
              help="Number of pool ntuples scanned for the m_chi quantiles "
                   "(branch not in the .npy cache).")
@click.option("--model-tag", default="", show_default=True,
              help="OUTPUT_TAG of a variant sweep (e.g. 'expr'), so its tagged manifest rows resolve against the default per-model picks.")
@click.option("--target", default="DMRD", show_default=True,
              help="TARGET_CONFIG key. Selects which branch the pool is read from and the band centre; a literal here silently loads relic-density data.")
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True)
def main(manifest, baseline_data_dir, mcmc_data_dir, cache_dir, output_dir, target, model_tag,
         m1_cut, tolerance, n_mchi_files, require_neutralino_lsp):
    import torch
    import uproot
    import plot_hit_rate_trajectories_multiseed as phr
    from analyse_runs import filter_run_neutralino_lsp, load_run

    true_value = 0.12
    out: dict = {"m1_cut": m1_cut, "tolerance": tolerance}

    # ── random pool ──────────────────────────────────────────────────────────
    # target, NOT "DMRD" (see compute_yield_comparison).
    X, Y = phr._load_xy_full(baseline_data_dir, target, Path(cache_dir))
    M1 = np.asarray(X[:, M1_COL]); om = np.asarray(Y).ravel()
    inband = np.abs(om - true_value) / true_value < tolerance
    low = np.abs(M1) < m1_cut
    out["pool"] = {
        "volume_share": float(low.mean()),
        "inband_share": float((inband & low).sum() / inband.sum()),
        "enrichment": float(((inband & low).sum() / inband.sum()) / low.mean()),
    }
    click.echo(f"[funnel] pool: volume share {out['pool']['volume_share']:.3f}, "
               f"in-band share {out['pool']['inband_share']:.3f} "
               f"(enrichment x{out['pool']['enrichment']:.1f})")

    # m_chi quantiles of in-band low-|M1| pool points (needs the ntuples)
    mchi = []
    for fn in sorted(glob.glob(f"{baseline_data_dir}/ntuple.*.root"))[:n_mchi_files]:
        t = uproot.open(fn)["susy"]
        o = t["MO_Omega"].array(library="np")
        m1 = t["IN_M_1"].array(library="np")
        mh = t["SP_m_h"].array(library="np")
        mc = t["SP_m_chi_10"].array(library="np")
        m = ((o > 0) & (o < 1) & (mh != -1)
             & (np.abs(o - true_value) / true_value < tolerance)
             & (np.abs(m1) < m1_cut))
        mchi.append(np.abs(mc[m]))
    mchi = np.concatenate(mchi)
    out["pool"]["mchi_quantiles_5_25_50_75_95"] = \
        [float(q) for q in np.percentile(mchi, [5, 25, 50, 75, 95])]
    click.echo(f"[funnel] pool in-band low-|M1| |m_chi| quantiles "
               f"(n={len(mchi)}): {np.round(out['pool']['mchi_quantiles_5_25_50_75_95'], 1)}")

    # ── emcee reference ──────────────────────────────────────────────────────
    tot = lowm = 0
    for fn in sorted(glob.glob(f"{mcmc_data_dir}/*.root")):
        t = uproot.open(fn)["susy"]
        m1 = t["IN_M_1"].array(library="np")
        tot += len(m1); lowm += int((np.abs(m1) < m1_cut).sum())
    out["mcmc"] = {"share": lowm / tot}
    click.echo(f"[funnel] emcee posterior share at |M1|<{m1_cut:.0f}: "
               f"{out['mcmc']['share']:.4f}")

    # ── AL cells ─────────────────────────────────────────────────────────────
    rows = [r for r in csv.DictReader(open(manifest)) if r["status"] == "completed"]
    out["al"] = {}
    for model, (strat, warm) in picks_with_tag(model_tag).items():
        m1s, oms, trajs = [], [], []
        for r in rows:
            if (r["model"], r["strategy"], r["warm_start"]) != (model, strat, warm):
                continue
            try:
                run = load_run(r["expected_run_dir"])
            except Exception:
                continue
            if require_neutralino_lsp:
                run = filter_run_neutralino_lsp(run)
            M1r = run.X[:, M1_COL]
            m1s.append(M1r); oms.append(run.Y)
            nt = [0] + list(run.n_train_per_iter)
            fr = [float((np.abs(M1r[a:min(b, len(M1r))]) < m1_cut).mean())
                  if min(b, len(M1r)) - a >= 20 else np.nan
                  for a, b in zip(nt[:-1], nt[1:])]
            trajs.append(fr)
        if not m1s:
            continue
        m1 = np.concatenate(m1s); om = np.concatenate(oms)
        low = np.abs(m1) < m1_cut
        ib = np.abs(om - true_value) / true_value < tolerance
        L = max(len(t) for t in trajs)
        A = np.full((len(trajs), L), np.nan)
        for i, t in enumerate(trajs):
            A[i, :len(t)] = t
        mtraj = np.nanmean(A, axis=0)
        blocks = [float(np.nanmean(mtraj[i:i + 8]))
                  for i in range(0, min(len(mtraj), 40), 8)]
        out["al"][model] = {
            "strategy": strat, "warm": warm, "n_seeds": len(m1s),
            "budget_share": float(low.mean()),
            "inband_rate_in_funnel": float((low & ib).sum() / max(low.sum(), 1)),
            "inband_rate_overall": float(ib.mean()),
            "hit_share_from_funnel": float((low & ib).sum() / max(ib.sum(), 1)),
            "budget_share_per_8_iters": blocks,
        }
        click.echo(f"[funnel] {model}: budget {low.mean():.2f}, "
                   f"hits-from-funnel {out['al'][model]['hit_share_from_funnel']:.2f}, "
                   f"trajectory {['%.2f' % b for b in blocks]}")

    # ── Omega ladder: composition of the pool by relic-density band ──────────
    # Explains the low-Omega structure of the overlay figure: light winos
    # (SU(2) coannihilation) fill the deepest-underabundant bands, and the
    # median lightest electroweakino-mass parameter climbs decade by decade
    # (thermal Omega ~ m^2), pushing the in-band population to the heavy
    # higgsino/bino corner.
    M1a, M2a, MUa = (np.asarray(X[:, PARAM_ORDER.index(b)])
                     for b in ("IN_M_1", "IN_M_2", "IN_mu"))
    Ya = np.asarray(Y).ravel()
    bands = [("lt_1e-4", Ya < 1e-4),
             ("1e-4_to_1e-3", (Ya >= 1e-4) & (Ya < 1e-3)),
             ("1e-3_to_1e-2", (Ya >= 1e-3) & (Ya < 1e-2)),
             ("1e-2_to_5e-2", (Ya >= 1e-2) & (Ya < 5e-2)),
             ("inband", np.abs(Ya - true_value) / true_value < tolerance)]
    out["omega_ladder"] = {}
    click.echo(f"\n[ladder] {'band':<14s} {'share':>7s} {'bino/wino/higgsino':>20s} {'med min-mass':>13s}")
    for name, m in bands:
        stacked = np.stack([np.abs(M1a[m]), np.abs(M2a[m]), np.abs(MUa[m])])
        key = np.argmin(stacked, axis=0)
        comp = [float((key == k).mean()) for k in range(3)]
        med = float(np.median(np.min(stacked, axis=0)))
        out["omega_ladder"][name] = {"share": float(m.mean()),
                                     "bino_wino_higgsino": comp,
                                     "median_min_mass_GeV": med}
        click.echo(f"[ladder] {name:<14s} {m.mean():7.3f} "
                   f"{comp[0]:.2f}/{comp[1]:.2f}/{comp[2]:.2f}{'':>8s} {med:10.0f}")

    p = Path(output_dir) / "funnel_analysis.json"
    p.write_text(json.dumps(out, indent=2))
    click.echo(f"[funnel] wrote {p}")


if __name__ == "__main__":
    main()
