"""Post-hoc uncertainty-quantification (UQ) evaluation from AL checkpoints.

For every best-per-model pick (``mcmc_diagnostics.DEFAULT_AL_PICKS``) this
walks the sweep-manifest seed runs, reloads the AL model at a subset of
iterations, evaluates the *full predictive distribution* on two seeded eval
sets, and scores it. Everything runs in transformed (log) target space, the
space the models are trained and selected in.

Predictive distributions (matched to what selection consumed):
  * transformer / dnn / dnn_match_trafo — T MC-dropout forward passes
    (default T=30, the selection-time value; ``--mc-samples 30,100`` ablates T)
  * exact_gp  — analytic Gaussian posterior (likelihood-marginal mean/var)
  * deep_gp   — mixture of S Gaussians from S likelihood samples (default
    S=8). NOTE: selection uses only the mixture's *within* component
    (mean of variances); we score the full mixture and report the
    within/between decomposition separately.
  * tabpfn    — no checkpoints (frozen, in-context): the regressor is re-fit
    on the iteration's stored training set; predictive quantiles from the
    bar-distribution ``icdf`` when available, Gaussian(mean, var) fallback.

Metrics per (cell, seed, iteration, eval set):
  * calibration — mean/variance of z=(y-mu)/sigma; PIT-based central-interval
    coverage at nominal 10..90,95% + miscalibration area
  * proper scores — NLPD (exact for Gaussian/mixture, Gaussian-approx for
    draw/quantile families) and CRPS (closed form for Gaussian and mixture,
    sample estimator for dropout draws, pinball for quantiles)
  * ranking — Spearman rho(sigma, |error|) and AUSE (area between the
    sigma-ordered and oracle-ordered sparsification curves)
  * sharpness — mean sigma; its per-iteration trajectory is the epistemic
    "shrinkage" test (the simulator is deterministic, so honest uncertainty
    must contract as the training set grows)

Eval sets: a seeded slice of the static random pool (same seed-123 carving as
the accuracy plots) and a seeded subsample of the emcee reference. Results
are cached per run dir (``uq_eval_cache.json``) so re-runs are cheap.

Usage (GPU node; see slurm/submit_uq_eval.sh):
    python scripts/evaluate_uq.py --require-neutralino-lsp
Smoke test (CPU):
    python scripts/evaluate_uq.py --models deep_gp --seeds 1 \
        --iter-step 40 --eval-size 2000 --skip-tabpfn
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcmc_diagnostics import DEFAULT_AL_PICKS, MODEL_DISPLAY  # noqa: E402

COVERAGE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SPARS_FRACS = np.linspace(0.0, 0.95, 20)
EVAL_DATASETS = ("static_random", "mcmc")
CACHE_NAME = "uq_eval_cache.json"
_DRAW_SEED = 20260805  # torch RNG seed for dropout / deep-GP likelihood draws

_SQRT2 = np.sqrt(2.0)
_SQRTPI = np.sqrt(np.pi)


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian helpers
# ──────────────────────────────────────────────────────────────────────────────

def _phi_cdf(z: np.ndarray) -> np.ndarray:
    from scipy.special import erf
    return 0.5 * (1.0 + erf(np.asarray(z) / _SQRT2))


def _phi_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z) ** 2) / np.sqrt(2.0 * np.pi)


def _crps_gaussian(err: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma^2) at y, with err = y - mu."""
    z = err / sigma
    return sigma * (z * (2.0 * _phi_cdf(z) - 1.0) + 2.0 * _phi_pdf(z) - 1.0 / _SQRTPI)


def _crps_mixture(y: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of an equal-weight Gaussian mixture (Grimit et al. 2006).

    mus, sigmas: (S, N). Returns (N,).
    """
    S = mus.shape[0]

    def A(mu, sig):  # E|X| for X ~ N(mu, sig^2)
        znz = mu / sig
        return mu * (2.0 * _phi_cdf(znz) - 1.0) + 2.0 * sig * _phi_pdf(znz)

    term1 = A(y[None, :] - mus, sigmas).mean(axis=0)
    term2 = np.zeros_like(y)
    for s in range(S):
        d = mus[s][None, :] - mus
        sig = np.sqrt(sigmas[s][None, :] ** 2 + sigmas ** 2)
        term2 += A(d, sig).sum(axis=0)
    term2 /= S * S
    return term1 - 0.5 * term2


def _crps_draws(y: np.ndarray, draws: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Sample-based CRPS estimator from (T, N) draws: E|X-y| - 0.5 E|X-X'|."""
    T, N = draws.shape
    out = np.empty(N)
    for i in range(0, N, chunk):
        d = draws[:, i:i + chunk]
        t1 = np.abs(d - y[None, i:i + chunk]).mean(axis=0)
        t2 = np.abs(d[:, None, :] - d[None, :, :]).mean(axis=(0, 1))
        out[i:i + chunk] = t1 - 0.5 * t2
    return out


def _crps_quantiles(y: np.ndarray, quants: np.ndarray, qgrid: np.ndarray) -> np.ndarray:
    """Pinball-loss CRPS approximation from a (Q, N) quantile table."""
    u = y[None, :] - quants
    pinball = u * (qgrid[:, None] - (u < 0).astype(np.float64))
    return 2.0 * pinball.mean(axis=0)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.empty(len(a)); ra[np.argsort(a)] = np.arange(len(a))
    rb = np.empty(len(b)); rb[np.argsort(b)] = np.arange(len(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _sparsification(abs_err: np.ndarray, sigma: np.ndarray):
    """RMSE-of-retained curves after removing the top-f fraction by sigma
    (model) and by |error| (oracle), both normalised to the full-set RMSE."""
    n = len(abs_err)
    sq = abs_err ** 2
    rmse_full = float(np.sqrt(sq.mean()))
    curves = {}
    for name, order in (("model", np.argsort(-sigma)), ("oracle", np.argsort(-abs_err))):
        # cumulative sum over the kept tail (least-uncertain points last)
        sq_ord = sq[order]
        tail_cumsum = np.cumsum(sq_ord[::-1])  # tail_cumsum[k-1] = sum of k smallest-ranked
        vals = []
        for f in SPARS_FRACS:
            keep = max(1, int(round(n * (1.0 - f))))
            vals.append(float(np.sqrt(tail_cumsum[keep - 1] / keep)) / rmse_full)
        curves[name] = vals
    ause = float(np.mean(np.asarray(curves["model"]) - np.asarray(curves["oracle"])))
    return curves, ause


# ──────────────────────────────────────────────────────────────────────────────
# Metric engine
# ──────────────────────────────────────────────────────────────────────────────

def _uq_metrics(y: np.ndarray, pred: dict) -> dict:
    """Score one predictive distribution against truths y (both transformed).

    ``pred``: {family, mean, var} plus family payload:
      empirical -> draws (T, N); mixture -> comp_means/comp_vars (S, N);
      quantiles -> quants (Q, N), qgrid (Q,).
    """
    family = pred["family"]
    mean = np.asarray(pred["mean"], dtype=np.float64).ravel()
    var = np.clip(np.asarray(pred["var"], dtype=np.float64).ravel(), 1e-12, None)
    sigma = np.sqrt(var)
    err = y - mean
    z = err / sigma

    # PIT of the truth under the predictive CDF
    if family == "gaussian":
        pit = _phi_cdf(z)
    elif family == "mixture":
        mus, svs = pred["comp_means"], np.sqrt(np.clip(pred["comp_vars"], 1e-12, None))
        pit = _phi_cdf((y[None, :] - mus) / svs).mean(axis=0)
    elif family == "empirical":
        d = pred["draws"]
        pit = ((d < y[None, :]).sum(axis=0) + 0.5 * (d == y[None, :]).sum(axis=0)) / len(d)
    elif family == "quantiles":
        quants, qgrid = pred["quants"], pred["qgrid"]
        pit = np.array([np.interp(y[j], quants[:, j], qgrid) for j in range(len(y))])
    else:
        raise ValueError(f"unknown family: {family}")

    coverage = {f"{L:g}": float(np.mean(np.abs(pit - 0.5) <= L / 2)) for L in COVERAGE_LEVELS}
    miscal = float(np.mean([abs(coverage[f"{L:g}"] - L) for L in COVERAGE_LEVELS]))

    # NLPD
    if family == "mixture":
        mus, cvs = pred["comp_means"], np.clip(pred["comp_vars"], 1e-12, None)
        logp = -0.5 * (np.log(2 * np.pi * cvs) + (y[None, :] - mus) ** 2 / cvs)
        m = logp.max(axis=0)
        nlpd = float(-(m + np.log(np.exp(logp - m).mean(axis=0))).mean())
    else:
        # exact for gaussian; moment-matched Gaussian approx for draws/quantiles
        nlpd = float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * z ** 2))

    # CRPS
    if family == "gaussian":
        crps = float(_crps_gaussian(err, sigma).mean())
    elif family == "mixture":
        crps = float(_crps_mixture(y, pred["comp_means"],
                                   np.sqrt(np.clip(pred["comp_vars"], 1e-12, None))).mean())
    elif family == "empirical":
        crps = float(_crps_draws(y, pred["draws"]).mean())
    else:
        crps = float(_crps_quantiles(y, pred["quants"], pred["qgrid"]).mean())

    abs_err = np.abs(err)
    spars_curves, ause = _sparsification(abs_err, sigma)

    out = {
        "n": int(len(y)),
        "family": family,
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean_sigma": float(sigma.mean()),
        "mean_z": float(z.mean()),
        "var_z": float(z.var()),
        "coverage": coverage,
        "miscalibration_area": miscal,
        "nlpd": nlpd,
        "crps": crps,
        "spearman_sigma_abserr": _spearman(sigma, abs_err),
        "ause": ause,
        "sparsification": {"fracs": [float(f) for f in SPARS_FRACS], **spars_curves},
    }
    if family == "mixture":
        out["var_within_mean"] = float(pred["comp_vars"].mean())
        out["var_between_mean"] = float(pred["comp_means"].var(axis=0).mean())
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Predictive adapters (all return transformed-space predictions)
# ──────────────────────────────────────────────────────────────────────────────

def _predict_dropout(model, X_eval, X_tr, Y_tr, T: int, device: str) -> dict:
    import torch
    from pmssm.data import compute_stats
    from pmssm.uncertainty import compute_uncertainty_mc_dropout
    stats = compute_stats(X_tr, Y_tr.unsqueeze(-1), torch.arange(len(X_tr)))
    torch.manual_seed(_DRAW_SEED + T)
    _, _, preds = compute_uncertainty_mc_dropout(
        model, X_eval.float(), stats, n_samples=T, device=device,
        logger=None, return_predictions=True)
    draws = preds.squeeze(-1).numpy().astype(np.float64)  # (T, N)
    return {"family": "empirical", "mean": draws.mean(axis=0),
            "var": draws.var(axis=0), "draws": draws}


def _predict_exact_gp(model, X_eval, jitter: float, device: str) -> dict:
    from pmssm.data import build_norm_tensors
    from pmssm.uncertainty import compute_uncertainty_gp
    dmin, dmax = build_norm_tensors()
    mean, var = compute_uncertainty_gp(model, X_eval.float(), dmin, dmax,
                                       "exact_gp", jitter=jitter)
    return {"family": "gaussian", "mean": mean.numpy().astype(np.float64),
            "var": var.numpy().astype(np.float64)}


def _predict_deep_gp(model, X_eval, jitter: float, num_samples: int, device: str) -> dict:
    """Mixture of S Gaussians from S likelihood samples (full predictive:
    total var = within + between; selection used only the within part)."""
    import torch
    import gpytorch
    from pmssm.data import build_norm_tensors, normalize_x
    dmin, dmax = build_norm_tensors()
    xn = normalize_x(X_eval.float(), dmin, dmax)
    model.eval()
    model.likelihood.eval()
    torch.manual_seed(_DRAW_SEED)
    means, varis = [], []
    bs = 10_000
    for i in range(0, len(xn), bs):
        xb = xn[i:i + bs].to(device)
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(False), \
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
             gpytorch.settings.num_likelihood_samples(num_samples):
            preds = model.likelihood(model(xb))
            means.append(preds.mean.detach().cpu().reshape(num_samples, -1))
            varis.append(preds.variance.detach().cpu().reshape(num_samples, -1))
    mus = np.concatenate([m.numpy() for m in means], axis=1).astype(np.float64)   # (S, N)
    cvs = np.concatenate([v.numpy() for v in varis], axis=1).astype(np.float64)   # (S, N)
    total_var = cvs.mean(axis=0) + mus.var(axis=0)
    return {"family": "mixture", "mean": mus.mean(axis=0), "var": total_var,
            "comp_means": mus, "comp_vars": cvs}


def _predict_tabpfn(X_tr, Y_tr, X_eval, device: str, target: str) -> dict:
    """Re-fit the frozen TabPFN in-context on the iteration's training set."""
    from tabpfn import TabPFNRegressor
    from pmssm.data import transform_y
    reg = TabPFNRegressor(device=device)
    reg.fit(X_tr.numpy(), transform_y(Y_tr, target=target).view(-1).numpy())
    qgrid = np.linspace(0.05, 0.95, 19)
    means, varis, quants = [], [], []
    bs = 10_000
    have_quants = True
    for i in range(0, len(X_eval), bs):
        res = reg.predict(X_eval[i:i + bs].numpy(), output_type="full")
        means.append(np.asarray(res["mean"], dtype=np.float64))
        v = res["criterion"].variance(res["logits"]).detach().cpu().numpy()
        if v.ndim > 1:
            v = v.mean(axis=0)
        varis.append(v.astype(np.float64))
        if have_quants:
            try:
                qs = [res["criterion"].icdf(res["logits"], float(q))
                      .detach().cpu().numpy().ravel() for q in qgrid]
                quants.append(np.stack(qs).astype(np.float64))  # (Q, B)
            except Exception:
                have_quants = False
                quants = []
    out = {"mean": np.concatenate(means), "var": np.concatenate(varis)}
    if have_quants and quants:
        out.update(family="quantiles", quants=np.concatenate(quants, axis=1),
                   qgrid=qgrid)
    else:
        out["family"] = "gaussian"
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-run evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _al_train_val(state: dict, iter_idx: int):
    import torch

    def _t(a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().to(torch.float32)
        return torch.from_numpy(np.asarray(a, dtype=np.float32))

    n_tr = int(state["al_n_train"][iter_idx])
    n_va = int(state["al_n_val"][iter_idx])
    return (_t(state["X"])[:n_tr], _t(state["Y"]).view(-1)[:n_tr],
            _t(state["X_val"])[:n_va], _t(state["Y_val"]).view(-1)[:n_va])


def _iter_list(n_iters: int, step: int) -> list[int]:
    """1-based iteration numbers: 1, step, 2*step, ... plus the final one."""
    its = {1, n_iters}
    its.update(range(step, n_iters + 1, step))
    return sorted(its)


def _eval_run(run_dir: str, model_type: str, eval_sets: dict, iter_step: int,
              mc_samples: list[int], dropout: float, device: str, target: str,
              refresh: bool, cache_tag: str) -> dict:
    """Return {iter_no: {ds: [metrics, ...]}} for one seed run (AL role only)."""
    import torch
    import plot_hit_rate_trajectories_multiseed as phr

    run_dir_p = Path(run_dir)
    state_path = run_dir_p / "state.pt"
    if not state_path.exists():
        click.echo(f"[uq]   skip (no state.pt): {run_dir}", err=True)
        return {}
    state = torch.load(state_path, weights_only=False, map_location="cpu")
    run_kwargs = phr._parse_run_kwargs_from_log(run_dir_p)
    yt = run_kwargs.get("y_transform", "log")
    if model_type not in ("tabpfn",) and yt not in (None, "log"):
        click.echo(f"[uq]   skip ({run_dir_p.name}): y_transform={yt} unsupported", err=True)
        return {}
    jitter = float(run_kwargs.get("jitter", 1e-3) or 1e-3)
    gp_ns = int(run_kwargs.get("gp_num_samples", 8) or 8)

    cache_path = run_dir_p / CACHE_NAME
    cache = {}
    if cache_path.exists() and not refresh:
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    n_iters = len(list(state.get("al_n_train") or []))
    out: dict = {}
    for iter_no in _iter_list(n_iters, iter_step):
        iter_dir = run_dir_p / f"iteration_{iter_no:03d}"
        if not iter_dir.exists():
            continue
        X_tr, Y_tr, X_va, Y_va = _al_train_val(state, iter_no - 1)
        if len(X_tr) == 0:
            continue

        model = None
        wrote = False
        variants = (mc_samples if model_type in ("transformer", "dnn", "dnn_match_trafo")
                    else [None])
        for ds, (X_ev, Y_ev) in eval_sets.items():
            for T in variants:
                key = f"v1|{cache_tag}|iter{iter_no}|{ds}|" + (f"T{T}" if T else
                      ("tabpfn" if model_type == "tabpfn" else f"gp{gp_ns}"))
                if key in cache and not refresh:
                    out.setdefault(iter_no, {}).setdefault(ds, []).append(cache[key])
                    continue

                t0 = time.time()
                if model is None and model_type != "tabpfn":
                    model = phr._load_iter_model(model_type, "al", iter_dir,
                                                 X_tr, Y_tr, X_va, Y_va,
                                                 run_kwargs, device, dropout=dropout)
                    if model is None:
                        click.echo(f"[uq]   iter {iter_no}: checkpoint missing", err=True)
                        break
                try:
                    if model_type in ("transformer", "dnn", "dnn_match_trafo"):
                        pred = _predict_dropout(model, X_ev, X_tr, Y_tr, T, device)
                    elif model_type == "exact_gp":
                        pred = _predict_exact_gp(model, X_ev, jitter, device)
                    elif model_type == "deep_gp":
                        pred = _predict_deep_gp(model, X_ev, jitter, gp_ns, device)
                    elif model_type == "tabpfn":
                        pred = _predict_tabpfn(X_tr, Y_tr, X_ev, device, target)
                    else:
                        raise ValueError(f"unsupported model_type: {model_type}")
                except Exception as exc:
                    click.echo(f"[uq]   iter {iter_no}/{ds}: predict failed: {exc}",
                               err=True)
                    continue

                from pmssm.data import transform_y
                y_true = transform_y(Y_ev.float(), target=target) \
                    .numpy().ravel().astype(np.float64)
                m = _uq_metrics(y_true, pred)
                if T is not None:
                    m["mc_samples"] = int(T)
                cache[key] = m
                wrote = True
                out.setdefault(iter_no, {}).setdefault(ds, []).append(m)
                click.echo(f"[uq]   iter {iter_no:>3}/{ds:<13} "
                           + (f"T={T:<4}" if T else "      ")
                           + f" var_z={m['var_z']:8.2f} "
                           f"miscal={m['miscalibration_area']:.3f} "
                           f"crps={m['crps']:.4f} ause={m['ause']:.3f} "
                           f"sigma={m['mean_sigma']:.4f} ({time.time()-t0:5.1f}s)")
                sys.stdout.flush()

        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if wrote:
            tmp = cache_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cache, indent=1, sort_keys=True))
            tmp.replace(cache_path)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation + plots
# ──────────────────────────────────────────────────────────────────────────────

_SCALARS = ("rmse", "mean_sigma", "mean_z", "var_z", "miscalibration_area",
            "nlpd", "crps", "spearman_sigma_abserr", "ause",
            "var_within_mean", "var_between_mean")


def _aggregate_final(per_seed: dict, T_default: int) -> dict:
    """Across-seed mean/SEM of scalar metrics at each seed's final iteration."""
    agg: dict = {}
    for ds in EVAL_DATASETS:
        rows = []
        for _seed, iters in per_seed.items():
            if not iters:
                continue
            last = max(iters)
            entries = iters[last].get(ds) or []
            entry = next((e for e in entries
                          if e.get("mc_samples", T_default) == T_default), None)
            if entry:
                rows.append(entry)
        if not rows:
            continue
        agg[ds] = {"n_seeds": len(rows)}
        for k in _SCALARS:
            vals = np.asarray([r[k] for r in rows if k in r], dtype=np.float64)
            if len(vals):
                agg[ds][k] = {
                    "mean": float(vals.mean()),
                    "sem": float(vals.std(ddof=1) / np.sqrt(len(vals)))
                           if len(vals) > 1 else 0.0,
                }
        # seed-mean coverage curve + sparsification for the plots
        agg[ds]["coverage"] = {
            f"{L:g}": float(np.mean([r["coverage"][f"{L:g}"] for r in rows]))
            for L in COVERAGE_LEVELS}
        fr = rows[0]["sparsification"]["fracs"]
        agg[ds]["sparsification"] = {
            "fracs": fr,
            "model": [float(np.mean([r["sparsification"]["model"][i] for r in rows]))
                      for i in range(len(fr))],
            "oracle": [float(np.mean([r["sparsification"]["oracle"][i] for r in rows]))
                       for i in range(len(fr))],
        }
    return agg


def _make_plots(results: dict, out_dir: Path, T_default: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_hit_rate_trajectories_multiseed as phr

    def _color(model):
        return phr.MODEL_COLORS.get(model, "gray")

    for ds in EVAL_DATASETS:
        # reliability diagram (final iteration, seed-mean)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for _cell, rec in results.items():
            agg = rec["final_aggregate"].get(ds)
            if not agg:
                continue
            lv = [float(k) for k in agg["coverage"]]
            ax.plot(lv, list(agg["coverage"].values()), "o-", ms=3,
                    color=_color(rec["model"]),
                    label=MODEL_DISPLAY.get(rec["model"], rec["model"]))
        ax.set_xlabel("nominal central coverage")
        ax.set_ylabel("empirical coverage")
        ax.set_title(f"Reliability, final iteration ({ds})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"uq_reliability_{ds}.png", dpi=200)
        plt.close(fig)

        # sharpness trajectory (epistemic-shrinkage test)
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        for _cell, rec in results.items():
            iters_all: dict[int, list[float]] = {}
            for _seed, iters in rec["per_seed"].items():
                for it, dss in iters.items():
                    for e in dss.get(ds) or []:
                        if e.get("mc_samples", T_default) == T_default:
                            iters_all.setdefault(it, []).append(e["mean_sigma"])
            if not iters_all:
                continue
            xs = sorted(iters_all)
            mu = np.array([np.mean(iters_all[i]) for i in xs])
            sem = np.array([np.std(iters_all[i], ddof=1) / np.sqrt(len(iters_all[i]))
                            if len(iters_all[i]) > 1 else 0.0 for i in xs])
            c = _color(rec["model"])
            ax.plot(xs, mu, "o-", ms=3, color=c,
                    label=MODEL_DISPLAY.get(rec["model"], rec["model"]))
            ax.fill_between(xs, mu - sem, mu + sem, color=c, alpha=0.2, lw=0)
        ax.set_yscale("log")
        ax.set_xlabel("AL iteration")
        ax.set_ylabel(r"mean predictive $\sigma$ (transformed space)")
        ax.set_title(f"Sharpness vs iteration ({ds})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"uq_sigma_trajectory_{ds}.png", dpi=200)
        plt.close(fig)

        # sparsification curves (final iteration, seed-mean)
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        for _cell, rec in results.items():
            agg = rec["final_aggregate"].get(ds)
            if not agg:
                continue
            sp = agg["sparsification"]
            c = _color(rec["model"])
            ax.plot(sp["fracs"], sp["model"], "-", color=c,
                    label=MODEL_DISPLAY.get(rec["model"], rec["model"]))
            ax.plot(sp["fracs"], sp["oracle"], "--", color=c, alpha=0.5, lw=1)
        ax.set_xlabel("fraction removed (most-uncertain first)")
        ax.set_ylabel("RMSE of retained / full RMSE")
        ax.set_title(f"Sparsification, final iteration ({ds}; dashed = oracle)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"uq_sparsification_{ds}.png", dpi=200)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358",
              show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True)
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True,
              help="Directory holding the pool .npy caches (same as the plot scripts).")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--models", default=None,
              help="Comma list of picks to evaluate (default: all DEFAULT_AL_PICKS).")
@click.option("--seeds", default=None, help="Comma list of seeds (default: all).")
@click.option("--include-status", default="completed", show_default=True,
              help="Comma list of manifest statuses to accept.")
@click.option("--eval-size", default=20_000, show_default=True,
              help="Points per eval set (seeded subsample).")
@click.option("--tabpfn-eval-size", default=5_000, show_default=True,
              help="Eval points for TabPFN (in-context inference is expensive).")
@click.option("--iter-step", default=5, show_default=True,
              help="Evaluate iterations 1, step, 2*step, ... plus the final one.")
@click.option("--mc-samples", default="30", show_default=True,
              help="Comma list of dropout pass counts (T ablation), e.g. '30,100'.")
@click.option("--dropout", default=0.1, show_default=True)
@click.option("--target", default="DMRD", show_default=True)
@click.option("--device", default=None, help="cuda/cpu (default: auto).")
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True)
@click.option("--skip-tabpfn", is_flag=True, default=False)
@click.option("--refresh", is_flag=True, default=False,
              help="Ignore per-run uq_eval_cache.json entries.")
def main(manifest, baseline_data_dir, mcmc_data_dir, cache_dir, output_dir,
         models, seeds, include_status, eval_size, tabpfn_eval_size, iter_step,
         mc_samples, dropout, target, device, require_neutralino_lsp,
         skip_tabpfn, refresh):
    import torch
    import plot_hit_rate_trajectories_multiseed as phr
    from pmssm.data import load_mcmc_data

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    T_list = [int(t) for t in str(mc_samples).split(",") if t.strip()]
    T_default = T_list[0]
    picks = dict(DEFAULT_AL_PICKS)
    if models:
        wanted = {m.strip() for m in models.split(",")}
        picks = {m: sw for m, sw in picks.items() if m in wanted}
    if skip_tabpfn:
        picks.pop("tabpfn", None)
    seed_filter = {int(s) for s in seeds.split(",")} if seeds else None
    statuses = {s.strip() for s in include_status.split(",")}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── eval sets ────────────────────────────────────────────────────────────
    X_full, Y_full = phr._load_xy_full(baseline_data_dir, target, Path(cache_dir))
    static_idx = phr._static_random_indices(len(Y_full), 2000, static_eval_size=eval_size)
    X_static = torch.from_numpy(np.asarray(X_full)[static_idx].astype(np.float32))
    Y_static = torch.from_numpy(np.asarray(Y_full)[static_idx].astype(np.float32))
    Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir, target=target,
                            require_neutralino_lsp=require_neutralino_lsp,
                            max_samples=eval_size)
    X_mcmc, Y_mcmc = Xm.float(), Ym.float().view(-1)
    click.echo(f"[uq] eval sets: static n={len(X_static)}, mcmc n={len(X_mcmc)}, "
               f"device={device}, T={T_list}, iter_step={iter_step}")

    veto_tag = f"veto{int(require_neutralino_lsp)}"

    rows = [r for r in csv.DictReader(open(manifest)) if r["status"] in statuses]
    results: dict = {}
    for model, (strat, warm) in picks.items():
        base_type = model[:-len("_oracle")] if model.endswith("_oracle") else model
        sel = [r for r in rows
               if (r["model"], r["strategy"], r["warm_start"]) == (model, strat, warm)
               and (seed_filter is None or int(r["seed"]) in seed_filter)]
        if not sel:
            click.echo(f"[uq] {model}-{strat}-{warm}: no manifest rows — skipped")
            continue
        n_ev = tabpfn_eval_size if base_type == "tabpfn" else eval_size
        eval_sets = {"static_random": (X_static[:n_ev], Y_static[:n_ev]),
                     "mcmc": (X_mcmc[:n_ev], Y_mcmc[:n_ev])}
        click.echo(f"[uq] {model}/{strat}/{warm}: {len(sel)} seed run(s)")
        per_seed: dict = {}
        for r in sel:
            click.echo(f"[uq]  seed {r['seed']}: {r['expected_run_dir']}")
            sys.stdout.flush()
            per_seed[int(r["seed"])] = _eval_run(
                r["expected_run_dir"], base_type, eval_sets, iter_step,
                T_list, dropout, device, target, refresh,
                cache_tag=f"n{n_ev}|{veto_tag}")
        rec = {"model": model, "strategy": strat, "warm": warm,
               "per_seed": per_seed,
               "final_aggregate": _aggregate_final(per_seed, T_default)}
        if any(per_seed.values()):
            results[f"{model}|{strat}|{warm}"] = rec

    # ── outputs ──────────────────────────────────────────────────────────────
    payload = {
        "config": {"manifest": manifest, "mcmc_data_dir": mcmc_data_dir,
                   "eval_size": eval_size, "tabpfn_eval_size": tabpfn_eval_size,
                   "iter_step": iter_step, "mc_samples": T_list, "dropout": dropout,
                   "require_neutralino_lsp": require_neutralino_lsp,
                   "include_status": sorted(statuses),
                   "coverage_levels": list(COVERAGE_LEVELS)},
        "results": {k: {"model": v["model"], "strategy": v["strategy"],
                        "warm": v["warm"],
                        "per_seed": {str(s): {str(i): d for i, d in its.items()}
                                     for s, its in v["per_seed"].items()},
                        "final_aggregate": v["final_aggregate"]}
                    for k, v in results.items()},
    }
    p = out_dir / "uq_evaluation.json"
    p.write_text(json.dumps(payload, indent=1))
    click.echo(f"[uq] wrote {p}")

    if results:
        _make_plots(results, out_dir, T_default)
        click.echo(f"[uq] wrote uq_reliability/uq_sigma_trajectory/"
                   f"uq_sparsification PNGs to {out_dir}")

        # console summary (final iteration, across-seed means)
        for ds in EVAL_DATASETS:
            click.echo(f"\n[uq] ── final-iteration summary ({ds}) — seed means ──")
            click.echo(f"{'model':<18} {'Var(z)':>8} {'miscal':>7} {'NLPD':>8} "
                       f"{'CRPS':>8} {'AUSE':>7} {'rho':>6} {'sigma':>8}")
            for _cell, rec in results.items():
                a = rec["final_aggregate"].get(ds)
                if not a:
                    continue
                click.echo(f"{rec['model']:<18} "
                           f"{a['var_z']['mean']:8.2f} "
                           f"{a['miscalibration_area']['mean']:7.3f} "
                           f"{a['nlpd']['mean']:8.3f} "
                           f"{a['crps']['mean']:8.4f} "
                           f"{a['ause']['mean']:7.3f} "
                           f"{a['spearman_sigma_abserr']['mean']:6.2f} "
                           f"{a['mean_sigma']['mean']:8.4f}")


if __name__ == "__main__":
    main()
