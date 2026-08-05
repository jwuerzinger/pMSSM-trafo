"""Prior-predictive / importance-sampling cross-check of the MCMC reference.

The random pool is a prior predictive sample for the emcee reference: both
draw from the flat prior over the identical 9-D box with the same validity
filters (the emcee run additionally enforces a neutralino LSP at sampling,
so the pool is filtered accordingly by default). Reweighting each pool point
by the emcee likelihood

    w_i = exp(-((Omega_i - 0.120)^2) / (2 * 0.012^2))

turns the pool into a self-normalised importance-sampling (SNIS) estimate of
the posterior with *uniform mode coverage* — an independent check on the
(formally unconverged) ensemble output that catches exactly what R-hat
cannot: modes or islands the walkers never visited.

Reported:
  * prior-predictive stats — mass near the datum (+-1/2 sigma of the
    likelihood, +-10% band), per-parameter prior->posterior contraction
    (std ratio, 1-D KL in bits), SNIS effective sample size
  * per-free-parameter weighted KS and range-normalised W1 distances between
    the SNIS posterior and the emcee marginals, + a 3x3 overlay figure
    (prior / SNIS posterior / emcee)
  * Omega moment check: SNIS vs emcee vs the targeted Gaussian
  * funnel cross-check: SNIS share at |M1| < m1-cut vs the emcee share

Outputs ``prior_predictive_check.json`` and
``prior_predictive_marginals.png`` in --output-dir.

Usage:
    python scripts/prior_predictive_check.py
(first run reads the full pool from ROOT, ~minutes; cached afterwards)
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

M1_IDX = FREE_PARAM_INDICES[FREE_PARAM_NAMES.index("M_1")]


def _load_pool(data_dir: str, cache_dir: Path, veto: bool):
    """Full valid pool (X, Y) with its own veto-aware .npy cache."""
    tag = data_dir.replace("/", "_").strip("_") + f"_veto{int(veto)}"
    xc, yc = cache_dir / f"ppc_x_{tag}.npy", cache_dir / f"ppc_y_{tag}.npy"
    if xc.exists() and yc.exists():
        return np.load(xc, mmap_mode="r"), np.load(yc)
    from pmssm.data import load_pmssm_data
    X, Y = load_pmssm_data(n_datasets=-1, target="DMRD", data_dir=data_dir,
                           plot_dir=str(cache_dir), require_neutralino_lsp=veto)
    X = X.numpy().astype(np.float32) if hasattr(X, "numpy") else np.asarray(X, np.float32)
    Y = Y.numpy().astype(np.float64).ravel() if hasattr(Y, "numpy") else np.asarray(Y, np.float64).ravel()
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(xc, X)
    np.save(yc, Y)
    return np.load(xc, mmap_mode="r"), Y


def _ecdf_distances(v1, w1, v2, w2, nbins=2000):
    """Weighted KS and range-normalised W1 between two weighted 1-D samples."""
    lo = min(v1.min(), v2.min())
    hi = max(v1.max(), v2.max())
    edges = np.linspace(lo, hi, nbins + 1)
    h1, _ = np.histogram(v1, bins=edges, weights=w1)
    h2, _ = np.histogram(v2, bins=edges, weights=w2)
    F1 = np.cumsum(h1) / h1.sum()
    F2 = np.cumsum(h2) / h2.sum()
    d = np.abs(F1 - F2)
    ks = float(d.max())
    w1d = float(d.sum() * (edges[1] - edges[0]) / (hi - lo))
    return ks, w1d


def _kl_bits(v_prior, v_post_vals, v_post_w, nbins=50):
    """1-D KL(posterior || prior) in bits from weighted histograms."""
    lo, hi = v_prior.min(), v_prior.max()
    edges = np.linspace(lo, hi, nbins + 1)
    p, _ = np.histogram(v_post_vals, bins=edges, weights=v_post_w)
    q, _ = np.histogram(v_prior, bins=edges)
    p = p / p.sum()
    q = q / q.sum()
    m = (p > 0) & (q > 0)
    return float(np.sum(p[m] * np.log2(p[m] / q[m])))


@click.command()
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358",
              show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True)
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--true-value", default=0.120, show_default=True,
              help="Center of the emcee Gaussian likelihood.")
@click.option("--likelihood-sigma", default=0.012, show_default=True,
              help="Width of the emcee Gaussian likelihood.")
@click.option("--tolerance", default=0.10, show_default=True,
              help="Relative band used for the in-band fraction.")
@click.option("--mcmc-max-samples", default=500_000, show_default=True)
@click.option("--m1-cut", default=100.0, show_default=True)
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp", default=True,
              show_default=True,
              help="Filter the pool to neutralino-LSP rows (the emcee prior "
                   "enforces this at sampling; keep ON for a like-for-like prior).")
def main(baseline_data_dir, mcmc_data_dir, cache_dir, output_dir, true_value,
         likelihood_sigma, tolerance, mcmc_max_samples, m1_cut,
         require_neutralino_lsp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pmssm.data import load_mcmc_data

    cache_dir = Path(cache_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load prior predictive sample (pool) and posterior sample (emcee) ─────
    Xp, Yp = _load_pool(baseline_data_dir, cache_dir, require_neutralino_lsp)
    Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir, target="DMRD",
                            require_neutralino_lsp=require_neutralino_lsp,
                            max_samples=mcmc_max_samples)
    Xm = Xm.numpy() if hasattr(Xm, "numpy") else np.asarray(Xm)
    Ym = np.asarray(Ym.view(-1) if hasattr(Ym, "view") else Ym, np.float64).ravel()
    click.echo(f"[ppc] pool n={len(Yp):,}, emcee n={len(Ym):,} "
               f"(veto={'on' if require_neutralino_lsp else 'off'})")

    # ── SNIS weights + ESS ────────────────────────────────────────────────────
    w = np.exp(-0.5 * ((Yp - true_value) / likelihood_sigma) ** 2)
    W = w.sum()
    ess = float(W ** 2 / np.sum(w ** 2))
    wn = w / W
    click.echo(f"[ppc] SNIS ESS = {ess:,.0f} "
               f"({ess / len(Yp):.2%} of pool; sum w = {W:,.1f})")

    # ── prior predictive stats ────────────────────────────────────────────────
    inband = np.abs(Yp - true_value) / true_value < tolerance
    f1 = float(np.mean(np.abs(Yp - true_value) < likelihood_sigma))
    f2 = float(np.mean(np.abs(Yp - true_value) < 2 * likelihood_sigma))
    out = {
        "config": {"true_value": true_value, "likelihood_sigma": likelihood_sigma,
                   "tolerance": tolerance, "mcmc_max_samples": mcmc_max_samples,
                   "require_neutralino_lsp": require_neutralino_lsp,
                   "pool_n": int(len(Yp)), "emcee_n": int(len(Ym))},
        "snis_ess": ess,
        "prior_predictive": {
            "frac_within_1sigma": f1,
            "frac_within_2sigma": f2,
            "frac_inband_10pct": float(inband.mean()),
        },
    }
    click.echo(f"[ppc] prior predictive mass: {f1:.4f} within +-1sig, "
               f"{f2:.4f} within +-2sig, {inband.mean():.4f} in +-10% band")

    # ── Omega moment check ────────────────────────────────────────────────────
    om_mean = float(np.sum(wn * Yp))
    om_std = float(np.sqrt(np.sum(wn * (Yp - om_mean) ** 2)))
    out["omega"] = {
        "snis_mean": om_mean, "snis_std": om_std,
        "emcee_mean": float(Ym.mean()), "emcee_std": float(Ym.std()),
        "target_mean": true_value, "target_std": likelihood_sigma,
    }
    click.echo(f"[ppc] Omega: SNIS {om_mean:.4f}+-{om_std:.4f} | "
               f"emcee {Ym.mean():.4f}+-{Ym.std():.4f} | "
               f"target {true_value:.4f}+-{likelihood_sigma:.4f}")

    # ── per-parameter marginals: contraction + SNIS-vs-emcee distances ───────
    ones_m = np.ones(len(Ym))
    out["params"] = {}
    click.echo(f"\n[ppc] {'param':<10} {'prior std':>10} {'post std':>9} "
               f"{'contract':>9} {'KL(bits)':>9} {'KS':>7} {'W1/range':>9} "
               f"{'support(pool | emcee)':>34}")
    for idx, name in zip(FREE_PARAM_INDICES, FREE_PARAM_NAMES):
        vp = np.asarray(Xp[:, idx], np.float64)
        vm = np.asarray(Xm[:, idx], np.float64)
        mu_post = float(np.sum(wn * vp))
        sd_post = float(np.sqrt(np.sum(wn * (vp - mu_post) ** 2)))
        sd_prior = float(vp.std())
        ks, w1d = _ecdf_distances(vp, w, vm, ones_m)
        kl = _kl_bits(vp, vp, w)
        short = name.replace("IN_", "")
        out["params"][short] = {
            "prior_std": sd_prior, "posterior_std_snis": sd_post,
            "contraction": sd_prior / sd_post if sd_post > 0 else float("inf"),
            "kl_post_vs_prior_bits": kl,
            "ks_snis_vs_emcee": ks, "w1_over_range_snis_vs_emcee": w1d,
            "pool_range": [float(vp.min()), float(vp.max())],
            "emcee_range": [float(vm.min()), float(vm.max())],
        }
        click.echo(f"[ppc] {short:<10} {sd_prior:10.1f} {sd_post:9.1f} "
                   f"{sd_prior / sd_post:9.2f} {kl:9.3f} {ks:7.4f} {w1d:9.5f} "
                   f"  [{vp.min():8.1f},{vp.max():8.1f}] | "
                   f"[{vm.min():8.1f},{vm.max():8.1f}]")

    # ── funnel cross-check ────────────────────────────────────────────────────
    m1p = np.abs(np.asarray(Xp[:, M1_IDX], np.float64))
    m1m = np.abs(np.asarray(Xm[:, M1_IDX], np.float64))
    share_snis = float(np.sum(wn * (m1p < m1_cut)))
    share_snis_sem = float(np.sqrt(max(share_snis * (1 - share_snis), 0.0) / ess))
    share_emcee = float((m1m < m1_cut).mean())
    out["funnel"] = {"m1_cut": m1_cut, "snis_share": share_snis,
                     "snis_share_sem": share_snis_sem, "emcee_share": share_emcee}
    click.echo(f"\n[ppc] funnel |M1|<{m1_cut:.0f}: SNIS {share_snis:.4f}"
               f"+-{share_snis_sem:.4f} | emcee {share_emcee:.4f}")

    # ── overlay figure: prior / SNIS posterior / emcee, 3x3 ─────────────────
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    ones_p = np.ones(len(Yp))
    for ax, idx, name in zip(axes.ravel(), FREE_PARAM_INDICES, FREE_PARAM_NAMES):
        vp = np.asarray(Xp[:, idx], np.float64)
        vm = np.asarray(Xm[:, idx], np.float64)
        lo = min(vp.min(), vm.min())
        hi = max(vp.max(), vm.max())
        edges = np.linspace(lo, hi, 60)
        for vals, wts, label, kw in (
                (vp, ones_p, "prior (pool)", dict(color="0.6", lw=1.0)),
                (vp, w, "SNIS posterior", dict(color="tab:green", lw=1.6)),
                (vm, ones_m, "emcee", dict(color="tab:purple", lw=1.6, ls="--"))):
            h, _ = np.histogram(vals, bins=edges, weights=wts, density=True)
            ax.stairs(h, edges, label=label, **kw)
        ax.set_xlabel(name.replace("IN_", ""))
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Prior predictive check: flat-prior pool reweighted by the "
                 "relic likelihood vs the emcee posterior", fontsize=11)
    fig.tight_layout()
    fp = out_dir / "prior_predictive_marginals.png"
    fig.savefig(fp, dpi=200)
    plt.close(fig)
    click.echo(f"[ppc] wrote {fp}")

    pj = out_dir / "prior_predictive_check.json"
    pj.write_text(json.dumps(out, indent=2))
    click.echo(f"[ppc] wrote {pj}")


if __name__ == "__main__":
    main()
