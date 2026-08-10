"""Electroweakino-composition fractions for every scan in the comparison.

Answers "which annihilation channels does each scan actually populate?" using
the *same* definition as the LSP-type figure (`plot_omega_vs_m2_lsp.py`):
the LSP's bino/wino/higgsino mixing fractions with a purity floor, via
`pmssm.visualization.classify_lsp_type` (bino / wino / higgsino / mixed, the
last when no component reaches `LSP_PURITY_MIN`).

SPheno's stored `SP_LSP_*_frac` branches are used wherever they exist: the
random pool's ntuples, and the AL runs' per-iteration candidate ntuples. The
emcee reference ntuples carry no mixing-fraction branches, so that one row
falls back to a reconstruction, obtained by diagonalising the tree-level
neutralino mass matrix in the (B, W3, Hd, Hu) basis from
(M_1, M_2, mu, tan beta) and taking the eigenvector of the smallest-|mass|
eigenstate:

    bino = N_11^2,  wino = N_12^2,  higgsino = N_13^2 + N_14^2

`--validate-against-spheno` checks that reconstruction against the pool's
stored SPheno fractions and reports the label agreement rate, so the size of
the tree-level-versus-loop-corrected difference is measured rather than
assumed: 98.5% over all pool rows, 86.9% restricted to in-band rows, where
near-degeneracies make the labelling more delicate.

Additionally reports the fraction inside the pure-higgsino thermal window
(|mu| in --higgsino-window), the value the reference concentrates on and the
AL loops miss.

Usage:
    python scripts/composition_fractions.py \\
        --manifest /ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv \\
        --baseline-data-dir /ptmp/jwuerzin/data/full_scan \\
        --mcmc-data-dir /ptmp/jwuerzin/data/neutralino_v4 \\
        --output-dir /ptmp/jwuerzin/analysis/all_runs
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

from analyse_runs import FREE_PARAM_INDICES, FREE_PARAM_NAMES  # noqa: E402
from mcmc_diagnostics import DEFAULT_AL_PICKS, _picks_from_manifest  # noqa: E402
from plot_al_input_target_diagnostics import _pooled_cell_data  # noqa: E402
from pmssm.visualization import (  # noqa: E402
    LSP_PURITY_MIN,
    LSP_TYPE_NAMES,
    classify_lsp_type,
)

DISPLAY = {
    "transformer": "Transformer",
    "deep_gp": "Deep GP",
    "exact_gp": "Exact GP",
    "dnn": "DNN",
    "dnn_match_trafo": "DNN-matched",
    "tabpfn": "TabPFN",
}

M_Z = 91.1876
SIN2_THETA_W = 0.23122


def neutralino_lsp_fracs(M1, M2, mu, tanb):
    """Tree-level (bino, wino, higgsino) fractions of the lightest neutralino.

    Diagonalises the symmetric neutralino mass matrix in the
    (B, W3, Hd, Hu) basis for each row and returns the squared mixing-matrix
    entries of the smallest-|mass| eigenstate, matching the convention of
    SPheno's `SP_LSP_*_frac` branches:
    bino = N_11^2, wino = N_12^2, higgsino = N_13^2 + N_14^2.
    """
    M1 = np.asarray(M1, dtype=float)
    beta = np.arctan(np.asarray(tanb, dtype=float))
    sb, cb = np.sin(beta), np.cos(beta)
    sw = np.sqrt(SIN2_THETA_W)
    cw = np.sqrt(1.0 - SIN2_THETA_W)
    n = M1.shape[0]
    A = np.zeros((n, 4, 4))
    A[:, 0, 0] = M1
    A[:, 1, 1] = np.asarray(M2, dtype=float)
    A[:, 0, 2] = A[:, 2, 0] = -M_Z * sw * cb
    A[:, 0, 3] = A[:, 3, 0] = M_Z * sw * sb
    A[:, 1, 2] = A[:, 2, 1] = M_Z * cw * cb
    A[:, 1, 3] = A[:, 3, 1] = -M_Z * cw * sb
    A[:, 2, 3] = A[:, 3, 2] = -np.asarray(mu, dtype=float)
    w, V = np.linalg.eigh(A)                     # w ascending, V columns
    lsp = np.argmin(np.abs(w), axis=1)
    Nrow = np.take_along_axis(V, lsp[:, None, None], axis=2).squeeze(2)  # (n, 4)
    N2 = Nrow ** 2
    return np.stack([N2[:, 0], N2[:, 1], N2[:, 2] + N2[:, 3]], axis=1)


def sanitize_spheno_fracs(fr):
    """NaN out the ntupler's -1 sentinel for non-neutralino LSPs.

    Those rows have no bino/wino/higgsino composition, but a raw -1 survives
    the isfinite check in `classify_lsp_type` and is labelled "mixed", which
    would otherwise make the mixed class a proxy for the sneutrino/stau-LSP
    population (86% of in-band random-pool rows).
    """
    fr = np.asarray(fr, dtype=np.float64).copy()
    fr[(fr < 0).any(axis=1)] = np.nan
    return fr


def _summarize(labels):
    out = {"n": int((labels >= 0).sum()), "n_all": int(len(labels))}
    ok = labels >= 0
    for k, name in LSP_TYPE_NAMES.items():
        out[name] = float((labels[ok] == k).mean()) if ok.any() else float("nan")
    # share of the dataset with no neutralino LSP at all
    out["no_neutralino_lsp"] = float((~ok).mean())
    return out


def _classify_from_params(X_free):
    """Composition via the tree-level reconstruction (available everywhere)."""
    fr = neutralino_lsp_fracs(X_free[:, FREE_PARAM_NAMES.index("M_1")],
                              X_free[:, FREE_PARAM_NAMES.index("M_2")],
                              X_free[:, FREE_PARAM_NAMES.index("mu")],
                              X_free[:, FREE_PARAM_NAMES.index("tanb")])
    return _summarize(classify_lsp_type(fr))


def _classify_from_spheno(fr):
    """Composition from SPheno's stored fractions (sentinel-aware)."""
    return _summarize(classify_lsp_type(sanitize_spheno_fracs(fr)))


def _inband(Y, tol, true_val):
    Y = np.asarray(Y).ravel()
    return np.abs(Y - true_val) / true_val < tol


AL_BRANCHES = ("IN_M_1", "IN_M_2", "IN_mu", "IN_tanb", "MO_Omega",
               "SP_LSP_Bino_frac", "SP_LSP_Wino_frac", "SP_LSP_Higgsino_frac")


def _al_cell_from_ntuples(run_dirs, max_iter=None):
    """Read the per-worker SPheno ntuples of an AL cell's runs.

    The AL driver keeps an ntuple for every evaluated candidate under
    `iteration_NNN/`, both directly in `worker_*/scan/` and, for candidates the
    loop re-dispatched, in `retry_NNN/worker_*/scan/`. Both must be read: the
    direct workers alone hold only ~45% of the evaluated points, and the
    retries are not a random subset of them. These carry SPheno's own LSP
    fraction branches. The
    pooled state.pt holds only inputs and Omega, so these files are the only
    route to the same composition definition the LSP-type figure uses.

    Returns (Omega, fracs (N,3), n_seeds, n_files).
    """
    import uproot                                          # noqa: PLC0415
    oms, frs, n_files = [], [], 0
    for d in run_dirs:
        it_dirs = sorted(Path(d).glob("iteration_*"))
        if max_iter is not None:
            it_dirs = it_dirs[:max_iter]
        for it in it_dirs:
            for p in it.rglob("ntuple.*.root"):
                try:
                    t = uproot.open(p)["susy"]
                    cols = {b: t[b].array(library="np") for b in AL_BRANCHES}
                except Exception:                          # noqa: BLE001
                    continue
                oms.append(cols["MO_Omega"])
                frs.append(np.stack([cols["SP_LSP_Bino_frac"],
                                     cols["SP_LSP_Wino_frac"],
                                     cols["SP_LSP_Higgsino_frac"]], axis=1))
                n_files += 1
    if not oms:
        return None, None, 0, 0
    return (np.concatenate(oms), np.concatenate(frs), len(run_dirs), n_files)


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True)
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/full_scan",
              show_default=True, help="Random-scan pool (set empty to skip).")
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True, help="emcee reference (set empty to skip).")
@click.option("--mcmc-max-samples", default=500_000, show_default=True, type=int)
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True)
@click.option("--tolerance", default=0.1, show_default=True)
@click.option("--target", default=0.12, show_default=True)
@click.option("--higgsino-window", default="1000,1250", show_default=True,
              help="|mu| window (GeV) of the pure-higgsino thermal target.")
@click.option("--validate-against-spheno/--no-validate-against-spheno",
              default=True, show_default=True,
              help="Check the recomputed fractions against the pool's stored "
                   "SPheno branches and report label agreement.")
@click.option("--require-neutralino-lsp", is_flag=True, default=False)
@click.option("--max-iter", default=0, type=int, show_default=True,
              help="Read only the first N AL iterations' ntuples (0 = all). "
                   "Useful for a quick smoke test; the full read is ~95k files.")
@click.option("--models", default="", help="Comma-separated subset of models.")
def main(manifest, output_dir, baseline_data_dir, mcmc_data_dir,
         mcmc_max_samples, cache_dir, tolerance, target,
         higgsino_window, validate_against_spheno, require_neutralino_lsp,
         max_iter, models):
    lo, hi = (float(v) for v in higgsino_window.split(","))
    imu = FREE_PARAM_NAMES.index("mu")
    out: dict = {"config": {"tolerance": tolerance, "target": target,
                            "lsp_purity_min": LSP_PURITY_MIN,
                            "higgsino_window": [lo, hi],
                            "require_neutralino_lsp": require_neutralino_lsp},
                 "rows": {}}

    def record(name, *, Y, fracs=None, X_free=None, extra=None):
        """Store one row. `fracs` = SPheno's own fractions (preferred);
        `X_free` enables the tree-level reconstruction and the |mu| window."""
        keep = _inband(Y, tolerance, target)
        if keep.sum() == 0:
            click.echo(f"[comp] {name}: no in-band points, skipping", err=True)
            return
        rec: dict = {"n_inband_raw": int(keep.sum())}
        if fracs is not None:
            rec.update(_classify_from_spheno(np.asarray(fracs)[keep]))
            rec["source"] = "spheno"
        if X_free is not None:
            Xi = np.asarray(X_free)[keep]
            recon = _classify_from_params(Xi)
            if fracs is None:
                rec.update(recon)
                rec["source"] = "reconstructed"
            rec["reconstructed"] = recon
            amu = np.abs(Xi[:, imu])
            rec["higgsino_target_window"] = float(((amu >= lo) & (amu <= hi)).mean())
        rec.update(extra or {})
        out["rows"][name] = rec
        click.echo(f"[comp] {name:<17} [{rec['source']:<13}] n={rec['n']:>7}  "
                   f"bino={rec['bino']:.3f} wino={rec['wino']:.3f} "
                   f"higgsino={rec['higgsino']:.3f} mixed={rec['mixed']:.3f}  "
                   f"noNeutralinoLSP={rec['no_neutralino_lsp']:.3f}  "
                   f"|mu|in[{lo:.0f},{hi:.0f}]="
                   f"{rec.get('higgsino_target_window', float('nan')):.3f}")

    # ── AL cells: SPheno fractions from the per-iteration worker ntuples ─────
    picks = dict(DEFAULT_AL_PICKS)
    if models:
        keep_m = {m.strip() for m in models.split(",") if m.strip()}
        picks = {m: sw for m, sw in picks.items() if m in keep_m}
    for model, run_dirs in _picks_from_manifest(manifest, picks).items():
        name = DISPLAY.get(model, model)
        if not run_dirs:
            click.echo(f"[comp] {model}: no runs for {picks[model]}", err=True)
            continue
        # X for the |mu| window and the reconstruction comes from state.pt
        X_free, omega_state, n_seeds = _pooled_cell_data(run_dirs,
                                                        require_neutralino_lsp)
        om, fr, n_seeds_nt, n_files = _al_cell_from_ntuples(
            run_dirs, max_iter=max_iter or None)
        if fr is None:
            click.echo(f"[comp] {name}: no worker ntuples found; "
                       f"falling back to the reconstruction", err=True)
            if X_free is not None:
                record(name, Y=omega_state, X_free=X_free,
                       extra={"n_seeds": n_seeds})
            continue
        # composition from SPheno; |mu| window from the pooled state
        record(name, Y=om, fracs=fr,
               extra={"n_seeds": n_seeds_nt, "n_ntuple_files": n_files})
        if X_free is not None:
            keep = _inband(omega_state, tolerance, target)
            amu = np.abs(np.asarray(X_free)[keep][:, imu])
            out["rows"][name]["higgsino_target_window"] = \
                float(((amu >= lo) & (amu <= hi)).mean())
            out["rows"][name]["reconstructed"] = \
                _classify_from_params(np.asarray(X_free)[keep])
            click.echo(f"[comp] {'':<17}  |mu| window from state.pt: "
                       f"{out['rows'][name]['higgsino_target_window']:.3f}")

    # ── random pool: SPheno fractions straight from the ntuples ──────────────
    if baseline_data_dir:
        import uproot                                       # noqa: PLC0415
        import plot_hit_rate_trajectories_multiseed as phr   # noqa: PLC0415
        Xb, Yb = phr._load_xy_full(baseline_data_dir, "DMRD", Path(cache_dir))
        Xb_free = np.asarray(Xb)[:, FREE_PARAM_INDICES]
        oms, frs = [], []
        for p in sorted(Path(baseline_data_dir).glob("ntuple.*.root")):
            try:
                t = uproot.open(p)["susy"]
                oms.append(t["MO_Omega"].array(library="np"))
                frs.append(np.stack(
                    [t[b].array(library="np") for b in
                     ("SP_LSP_Bino_frac", "SP_LSP_Wino_frac",
                      "SP_LSP_Higgsino_frac")], axis=1))
            except Exception:                               # noqa: BLE001
                continue
        if oms:
            record("random pool", Y=np.concatenate(oms),
                   fracs=np.concatenate(frs))
            keep = _inband(Yb, tolerance, target)
            amu = np.abs(Xb_free[keep][:, imu])
            out["rows"]["random pool"]["higgsino_target_window"] = \
                float(((amu >= lo) & (amu <= hi)).mean())
            out["rows"]["random pool"]["reconstructed"] = \
                _classify_from_params(Xb_free[keep])

            # ── validate the reconstruction against SPheno, in-band ─────────
            if validate_against_spheno:
                om_all = np.concatenate(oms)
                fr_all = sanitize_spheno_fracs(np.concatenate(frs))
                ib = _inband(om_all, tolerance, target)
                # align by row order: the cached X and the ntuple scan agree
                if len(om_all) == len(Xb_free):
                    ref = classify_lsp_type(fr_all[ib])
                    ok = ref >= 0
                    Xi = Xb_free[ib][ok]
                    mine = classify_lsp_type(neutralino_lsp_fracs(
                        Xi[:, FREE_PARAM_NAMES.index("M_1")],
                        Xi[:, FREE_PARAM_NAMES.index("M_2")],
                        Xi[:, FREE_PARAM_NAMES.index("mu")],
                        Xi[:, FREE_PARAM_NAMES.index("tanb")]))
                    out["validation"] = {
                        "n_inband_neutralino": int(ok.sum()),
                        "label_agreement_inband": float((mine == ref[ok]).mean()),
                    }
                    click.echo(f"[comp] reconstruction vs SPheno on "
                               f"{int(ok.sum())} in-band neutralino-LSP pool "
                               f"rows: labels agree "
                               f"{out['validation']['label_agreement_inband']*100:.2f}%")
                else:
                    click.echo("[comp] validation skipped: cached X and ntuple "
                               "row counts differ", err=True)

    # ── emcee reference: no fraction branches, reconstruction only ───────────
    if mcmc_data_dir:
        from pmssm.data import load_mcmc_data  # noqa: PLC0415
        Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir,
                                require_neutralino_lsp=require_neutralino_lsp,
                                max_samples=mcmc_max_samples or None)
        Xm = Xm.numpy() if hasattr(Xm, "numpy") else np.asarray(Xm)
        Ym = (Ym.numpy() if hasattr(Ym, "numpy") else np.asarray(Ym)).ravel()
        record("emcee reference", Y=Ym, X_free=Xm[:, FREE_PARAM_INDICES])

    p = Path(output_dir) / "composition_fractions.json"
    p.write_text(json.dumps(out, indent=1))
    click.echo(f"[comp] wrote {p}")

    # ── LaTeX table body, ready to paste ─────────────────────────────────────
    order = ["random pool", "emcee reference", "Deep GP", "Exact GP",
             "Transformer", "DNN", "DNN-matched", "TabPFN"]
    click.echo("\n% --- tab:composition body ---")
    for name in order:
        r = out["rows"].get(name)
        if r is None:
            continue
        mark = "$^{\\ast}$" if r.get("source") == "reconstructed" else ""
        click.echo(f"{name+mark:<25}& {r['bino']*100:5.1f} & {r['wino']*100:5.1f} "
                   f"& {r['higgsino']*100:5.1f} & {r['mixed']*100:5.1f} "
                   f"& {r['no_neutralino_lsp']*100:5.1f} "
                   f"& {r.get('higgsino_target_window', float('nan'))*100:5.1f} \\\\")


if __name__ == "__main__":
    main()
