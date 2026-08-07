"""Rank-statistic uniformity across AL seed replicas (and the MCMC reference).

Ports the rank-plot / computational-faithfulness check of Run3ModelGen's
``emcee_diagnostics.py`` (Vehtari et al. 2021 rank plots, the computable
stand-in for SBC rank statistics) to active-learning acquisition streams.

Every draw of a parameter is ranked POOLED over all replicas, then the ranks
are histogrammed per replica. If the replicas realise the same acquisition
distribution each histogram is uniform; departures show HOW they differ:

  * ``inverted-U``  — this replica is concentrated in a sub-region of the
    pooled range (redundant coverage; the within-sequence deficit)
  * ``U-shaped``    — over-dispersed relative to the pooled sample
  * ``skewed-low/high`` — systematically displaced toward one end (the
    between-replica offset seen on the weakly constrained slepton axes)

Interpretation for AL (NOT the MCMC reading): the seeds are replicas of a
stochastic acquisition process, not Markov chains targeting a common
stationary law, so uniformity is a *reproducibility* statement (independently
seeded runs acquire statistically indistinguishable point sets), never a
statement about posterior calibration. This is the same caveat that applies
to the R-hat column of the paper's diagnostics table.

Thinning: the histogram uses every acquired point, the TEST uses a thinned,
approximately independent subsample, because a chi-square that assumes
independence rejects far too often when consecutive draws are correlated.
Two conventions are offered:

  * ``--thin-mode tau``   (default) thin by the measured integrated
    autocorrelation length of the acquisition sequence (max over parameters,
    from ESS_bulk) — a no-op for the neural surrogates (tau ~ 1) and the
    real correction for the exact GP (tau ~ 25-40)
  * ``--thin-mode batch`` keep one point per acquisition batch: points inside
    one iteration are chosen jointly by one frozen surrogate, so the
    iteration, not the point, is the natural independent unit

Usage:
    python scripts/rank_uniformity_al.py --require-neutralino-lsp
    python scripts/rank_uniformity_al.py --thin-mode batch --models deep_gp
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

from mcmc_diagnostics import (  # noqa: E402
    DEFAULT_AL_PICKS,
    MODEL_DISPLAY,
    PARAM_ORDER,
    _load_al_chains,
    _picks_from_manifest,
    ess_bulk,
)

RANK_BINS = 10        # rank-histogram bins (upstream convention)
RANK_Z_FLAT = 3.0     # |z| below this: no shape called
FREE_PARAMS = ["IN_M_1", "IN_M_2", "IN_mu", "IN_tanb", "IN_At", "IN_Ab",
               "IN_Atau", "IN_meL", "IN_meR"]


def _rank_shape(counts: np.ndarray) -> tuple[str, float, float]:
    """Classify one replica's rank histogram via two orthogonal contrasts.

    location   L = sum_b p_b x_b          (which end of the pooled range)
    dispersion Q = sum_b p_b (x_b^2 - m2) (extremes vs middle)

    Both standardised by their exact discrete null moments over the B bin
    midpoints, so no continuous approximation enters.
    """
    counts = np.asarray(counts, dtype=float)
    n, B = counts.sum(), counts.size
    if n <= 0 or B < 3:
        return "n/a", float("nan"), float("nan")
    p = counts / n
    x = (np.arange(B) + 0.5) / B - 0.5
    m2 = float((x ** 2).mean())
    q = x ** 2 - m2
    var_x, var_q = float((x ** 2).mean()), float((q ** 2).mean())
    if var_x <= 0 or var_q <= 0:
        return "n/a", float("nan"), float("nan")
    z_loc = float((p * x).sum() / np.sqrt(var_x / n))
    z_dis = float((p * q).sum() / np.sqrt(var_q / n))
    if max(abs(z_loc), abs(z_dis)) < RANK_Z_FLAT:
        return "uniform", z_loc, z_dis
    if abs(z_loc) >= RANK_Z_FLAT and abs(z_dis) >= RANK_Z_FLAT:
        return "mixed", z_loc, z_dis
    if abs(z_loc) >= abs(z_dis):
        return ("skewed-high" if z_loc > 0 else "skewed-low"), z_loc, z_dis
    return ("U-shaped" if z_dis > 0 else "inverted-U"), z_loc, z_dis


def _holm(pvals: list[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, input order preserved."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.size, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if finite.size == 0:
        return out
    order = finite[np.argsort(p[finite])]
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (order.size - i) * float(p[idx])))
        out[idx] = running
    return out


def _tau_multichain(chains: list[np.ndarray]) -> float:
    """Max over parameters of M*N/ESS_bulk (the multichain autocorrelation time).

    Reported for contrast only, NOT used for thinning: the multichain estimator
    folds between-replica disagreement into an effective correlation at every
    lag, so a permanent offset on one axis inflates it even when each sequence
    is internally independent. Thinning by it would discard almost everything.
    """
    n = min(len(c) for c in chains)
    arr = np.stack([c[:n] for c in chains])           # (M, N, P)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = arr.shape[0] * n / np.asarray(ess_bulk(arr), dtype=float)
    tau = tau[np.isfinite(tau)]
    return float(tau.max()) if tau.size else 1.0


def _tau_within(chains: list[np.ndarray]) -> float:
    """Max over parameters of the WITHIN-sequence integrated autocorrelation.

    Per replica and parameter: FFT autocorrelation, Geyer initial-positive-
    sequence truncation, tau = 1 + 2 sum rho_t; averaged over replicas, then
    the maximum over parameters. This is the quantity that governs how far
    apart two acquired points must be to count as independent, and it is what
    the chi-square test needs to be calibrated.
    """
    n = min(len(c) for c in chains)
    taus = []
    for pi in range(chains[0].shape[1]):
        per_chain = []
        for c in chains:
            x = np.asarray(c[:n, pi], dtype=float)
            if not np.all(np.isfinite(x)) or x.std() == 0:
                continue
            x = x - x.mean()
            nfft = 1 << (2 * len(x) - 1).bit_length()
            f = np.fft.rfft(x, nfft)
            acf = np.fft.irfft(f * np.conjugate(f), nfft)[:len(x)].real
            acf /= acf[0]
            # Geyer: truncate at the first non-positive consecutive pair sum
            s, k = 0.0, 1
            while k + 1 < len(acf):
                pair = acf[k] + acf[k + 1]
                if pair <= 0:
                    break
                s += pair
                k += 2
            per_chain.append(max(1.0, 1.0 + 2.0 * s))
        if per_chain:
            taus.append(float(np.mean(per_chain)))
    return max(taus) if taus else 1.0


def rank_uniformity(chains: list[np.ndarray], params: list[str],
                    step: int) -> dict:
    """Pooled-rank histograms per replica + a thinned homogeneity chi-square."""
    from scipy.stats import chi2, rankdata

    C = len(chains)
    n_all = min(len(c) for c in chains)
    flat = [c[:n_all] for c in chains]
    thin = [c[::step] for c in flat]
    n_thin = min(len(t) for t in thin)
    thin = [t[:n_thin] for t in thin]

    B = RANK_BINS
    estimable = n_thin >= 5 * B
    hist_all, hist_thin, raw_p, shapes, shapes_thin, stats = {}, {}, [], {}, {}, {}
    for pi, name in enumerate(params):
        pooled = np.concatenate([f[:, pi] for f in flat])
        if not np.all(np.isfinite(pooled)):
            hist_all[name] = np.zeros((C, B))
            raw_p.append(float("nan"))
            shapes[name] = ["n/a"] * C
            shapes_thin[name] = ["n/a"] * C
            continue
        # histogram over ALL draws (the picture)
        r = rankdata(pooled, method="average").reshape(C, n_all)
        edges = np.linspace(0.5, C * n_all + 0.5, B + 1)
        h = np.stack([np.histogram(r[c], bins=edges)[0] for c in range(C)]).astype(float)
        hist_all[name] = h
        shapes[name] = [_rank_shape(h[c])[0] for c in range(C)]

        # chi-square on the thinned, approximately independent subsample
        pooled_t = np.concatenate([t[:, pi] for t in thin])
        rt = rankdata(pooled_t, method="average").reshape(C, n_thin)
        edges_t = np.linspace(0.5, C * n_thin + 0.5, B + 1)
        ht = np.stack([np.histogram(rt[c], bins=edges_t)[0] for c in range(C)]).astype(float)
        hist_thin[name] = ht
        shapes_thin[name] = [_rank_shape(ht[c])[0] for c in range(C)]
        exp = ht.sum(axis=0, keepdims=True) * ht.sum(axis=1, keepdims=True) / ht.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = float(np.nansum((ht - exp) ** 2 / np.where(exp > 0, exp, np.nan)))
        dof = (C - 1) * (B - 1)
        p = float(chi2.sf(stat, dof)) if estimable and dof > 0 else float("nan")
        raw_p.append(p)
        stats[name] = {"chi2": stat, "dof": dof, "p_raw": p}

    adj = _holm(raw_p)
    for name, a in zip(params, adj):
        if name in stats:
            stats[name]["p_holm"] = float(a) if np.isfinite(a) else None
    worst = int(np.nanargmax([-(a if np.isfinite(a) else np.inf) for a in adj])) \
        if np.any(np.isfinite(adj)) else None
    return {
        "n_chains": C, "n_all": n_all, "n_thinned": n_thin, "thin_step": int(step),
        "estimable": bool(estimable), "bins": B,
        "hist_all": {k: v.tolist() for k, v in hist_all.items()},
        "shapes": shapes, "shapes_thinned": shapes_thin,
        "per_param": stats,
        "worst_param": params[int(np.nanargmin(adj))] if np.any(np.isfinite(adj)) else None,
        "worst_p_holm": float(np.nanmin(adj)) if np.any(np.isfinite(adj)) else None,
    }


def _plot(res: dict, params: list[str], out_path: Path, model_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C, B = res["n_chains"], res["bins"]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    cmap = plt.get_cmap("tab10")
    for ax, name in zip(axes.ravel(), params):
        h = np.asarray(res["hist_all"][name], dtype=float)
        edges = np.linspace(0, 1, B + 1)
        for c in range(C):
            frac = h[c] / max(h[c].sum(), 1) * B      # 1.0 = uniform
            ax.stairs(frac, edges, color=cmap(c % 10), lw=1.3,
                      label=f"seed {c + 1}" if name == params[0] else None)
        ax.axhline(1.0, color="black", ls="--", lw=0.9)
        ax.set_ylabel(name.replace("IN_", ""), fontsize=9)
        ax.grid(alpha=0.25)
        sh = res["shapes"][name]
        tag = "uniform" if all(s == "uniform" for s in sh) else \
            "/".join(sorted(set(s for s in sh if s != "uniform")))
        ax.text(0.98, 0.94, tag, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="0.35")
    for ax in axes[-1]:
        ax.set_xlabel("pooled rank (normalised)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--models", default=None,
              help="Comma list of picks (default: all DEFAULT_AL_PICKS).")
@click.option("--thin-mode", type=click.Choice(["tau", "batch", "none"]),
              default="tau", show_default=True,
              help="Independence convention for the chi-square test.")
@click.option("--batch-size", default=500, show_default=True,
              help="Points per acquisition batch (--thin-mode batch).")
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True)
def main(manifest, output_dir, models, thin_mode, batch_size, require_neutralino_lsp):
    picks = dict(DEFAULT_AL_PICKS)
    if models:
        wanted = {m.strip() for m in models.split(",")}
        picks = {m: sw for m, sw in picks.items() if m in wanted}
    run_dirs = _picks_from_manifest(manifest, picks)
    param_idx = [PARAM_ORDER.index(p) for p in FREE_PARAMS]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {"config": {"thin_mode": thin_mode, "batch_size": batch_size,
                          "bins": RANK_BINS,
                          "require_neutralino_lsp": require_neutralino_lsp},
               "results": {}}
    for model, dirs in run_dirs.items():
        chains = _load_al_chains(dirs, param_idx,
                                 require_neutralino_lsp=require_neutralino_lsp)
        if len(chains) < 2:
            click.echo(f"[rank] {model}: {len(chains)} replicas — skipped")
            continue
        tau_w = _tau_within(chains)
        tau_m = _tau_multichain(chains)
        if thin_mode == "tau":
            step = max(1, int(np.ceil(tau_w)))
        elif thin_mode == "batch":
            step = max(1, int(batch_size))
        else:
            step = 1
        res = rank_uniformity(chains, FREE_PARAMS, step)
        res["tau_within"] = tau_w
        res["tau_multichain"] = tau_m
        payload["results"][model] = res
        _plot(res, FREE_PARAMS, out_dir / f"rank_uniformity_{model}.png",
              MODEL_DISPLAY.get(model, model))
        nonuni = {p: s for p, s in res["shapes_thinned"].items()
                  if any(x != "uniform" for x in s)}
        click.echo(f"[rank] {model:<16} C={res['n_chains']} N={res['n_all']:>6} "
                   f"tau_within={tau_w:6.1f} tau_multi={tau_m:7.1f} "
                   f"thin={step:>3} n_test={res['n_thinned']:>5} "
                   f"p_Holm={res['worst_p_holm']:.1e} ({res['worst_param']}); "
                   f"non-uniform (tested): "
                   + (", ".join(f"{p.replace('IN_','')}:{'/'.join(sorted(set(x for x in s if x != 'uniform')))}"
                                for p, s in nonuni.items()) or "none"))

    p = out_dir / "rank_uniformity_al.json"
    p.write_text(json.dumps(payload, indent=1))
    click.echo(f"[rank] wrote {p} and rank_uniformity_<model>.png")


if __name__ == "__main__":
    main()
