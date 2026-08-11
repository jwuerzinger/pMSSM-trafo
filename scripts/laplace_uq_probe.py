"""Is the Transformer's acquisition uncertainty fixable post-hoc?

Table 4 shows MC dropout's predictive spread barely orders the Transformer's own
errors on target-weighted samples (Spearman rho = 0.17) while the GPs reach
0.44-0.47, and a K=5 ensemble bought nothing at 2.9x the compute. Both dropout
and ensembling measure the same thing: how unstable the fitted function is under
perturbations of its own weights. Neither is a function of distance from the
training data, which is precisely what the GP posterior variance is.

This probe asks whether the trained network already carries that missing
information, without retraining anything. Three uncertainty candidates are
scored against the SAME errors, on the SAME eval sets as Table 4:

  dropout       T stochastic forward passes (the paper's method, recomputed here
                so it shares the eval subsample with everything else).
  laplace       Last-layer linearised Laplace: a Gaussian posterior on the final
                Linear's weights with covariance Sigma = (Phi^T Phi/sn2 + lam I)^-1,
                giving sigma^2(x) = phi(x)^T Sigma phi(x) + sn2. Because the output
                layer is linear and the loss is MSE, this GGN is EXACT, so the
                only approximation is holding the features phi fixed
                (Daxberger et al. 2021; Immer et al. 2021; Kristiadi et al. 2020).
                Scored over a grid of prior precisions lam, whose two limits are
                informative in themselves: lam -> inf ranks by ||phi||, while
                lam -> 0 ranks by the Mahalanobis norm of phi under the training
                feature covariance, i.e. the distance-aware quantity.
  knn           Mean distance to the k nearest training points, in normalised
                input space and in feature space. No posterior at all, so it
                isolates whether bare geometry already beats dropout.

The mean prediction is the network's own deterministic (eval-mode) output for
every candidate, so RMSE is held fixed and any change in rho is attributable to
sigma alone. Dropout's own-mean rho is reported alongside for continuity with
Table 4.

Reading the result: if Laplace's rho climbs toward the GPs' 0.44 then the
representation does carry distance information and a full AL run is worth
submitting. If rho stays near 0.17 for every lam AND k-NN distance also scores
low, the features have collapsed and no head on top can recover it, which points
at spectral normalisation / SNGP instead (Liu et al. 2020).

Usage:
    python scripts/laplace_uq_probe.py --models transformer,dnn \\
        --output-dir /ptmp/jwuerzin/analysis/all_runs
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

# Prior precisions to scan. Must reach far enough down to expose the
# data-dominated limit: at lam -> 0 sigma becomes the Mahalanobis norm of phi
# under the training feature covariance, which is the distance-aware quantity
# we are actually testing for. The reported "principled" value is whichever lam
# minimises validation NLPD; a lam* pinned at either end means the grid is too
# narrow and should be widened rather than read as a result.
LAMBDA_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
               1.0, 10.0, 100.0, 1e3, 1e4)
BATCH = 8192          # PyTorch efficient attention fails above 65535
_DRAW_SEED = 20260810


def _resolve_picks(manifest: Path, models: list[str]) -> dict:
    """{model: [run_dir, ...]} for each model's best (strategy, warm) cell."""
    from mcmc_diagnostics import DEFAULT_AL_PICKS
    rows = list(csv.DictReader(manifest.open()))
    out = {}
    for model in models:
        if model not in DEFAULT_AL_PICKS:
            raise click.ClickException(f"no canonical pick for {model!r}")
        strategy, warm = DEFAULT_AL_PICKS[model]
        dirs = [r["expected_run_dir"] for r in rows
                if r["model"] == model and r["strategy"] == strategy
                and r["warm_start"] == warm
                and r["status"] in ("completed", "timeout")
                and Path(r["expected_run_dir"], "state.pt").exists()]
        if not dirs:
            click.echo(f"[lap] WARNING no runs for {model}/{strategy}/{warm}",
                       err=True)
        out[model] = sorted(dirs)
    return out


def _last_linear(model):
    """The final nn.Linear, whose weights carry the Laplace posterior."""
    import torch.nn as nn
    last = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    if last is None:
        raise RuntimeError("model has no nn.Linear to linearise")
    return last


def _forward_features(model, X_norm, device, layer):
    """Deterministic mean prediction and the features feeding ``layer``.

    Returns (mu (N,), Phi (N, d+1)) with a trailing 1 for the bias, both float64
    on the CPU. Runs in eval mode so dropout is off: this is the MAP function.
    """
    import torch
    grab = {}

    def hook(_mod, inputs, _out):
        grab["phi"] = inputs[0].detach()

    h = layer.register_forward_hook(hook)
    model.to(device).eval()
    mus, phis = [], []
    try:
        with torch.no_grad():
            for i in range(0, len(X_norm), BATCH):
                xb = X_norm[i:i + BATCH].to(device)
                y = model(xb)
                mus.append(y.detach().cpu().reshape(-1))
                phis.append(grab["phi"].cpu())
    finally:
        h.remove()
    Phi = torch.cat(phis).to(torch.float64)
    ones = torch.ones(len(Phi), 1, dtype=torch.float64)
    return (torch.cat(mus).to(torch.float64).numpy(),
            torch.cat([Phi, ones], dim=1))


def _laplace_var(Phi_tr, Phi_ev, sn2: float, lam: float) -> np.ndarray | None:
    """sigma^2(x) = phi^T (Phi^T Phi / sn2 + lam I)^-1 phi + sn2.

    Cholesky-solved rather than inverted; O(d^3) once, then O(d^2) per point, so
    the cost is set by the feature dimension and not by the labelled-set size.
    Verified against laplace-torch's last-layer full-GGN GLM predictive to 1e-6
    relative (that library returns the functional variance, i.e. without the
    trailing sn2). Returns None if A is not positive definite, which happens at
    small lam when the ReLU features are rank-deficient (dead units).
    """
    import torch
    d = Phi_tr.shape[1]
    A = Phi_tr.T @ Phi_tr / sn2 + lam * torch.eye(d, dtype=torch.float64)
    try:
        L = torch.linalg.cholesky(A)
    except Exception:
        return None
    # q = ||L^-1 phi||^2 for every row of Phi_ev
    V = torch.linalg.solve_triangular(L, Phi_ev.T, upper=False)
    q = (V ** 2).sum(dim=0)
    return (q + sn2).numpy().astype(np.float64)


def _knn_dist(A, B, k: int, chunk: int = 2048) -> np.ndarray:
    """Mean distance from each row of A to its k nearest rows of B."""
    import torch
    out = np.empty(len(A))
    kk = min(k, len(B))
    for i in range(0, len(A), chunk):
        d = torch.cdist(A[i:i + chunk].float(), B.float())
        out[i:i + chunk] = d.topk(kk, dim=1, largest=False).values \
            .mean(dim=1).numpy()
    return out


def _rank_scores(abs_err: np.ndarray, sigma: np.ndarray, euq) -> dict:
    """The two ordering-only metrics: nothing here depends on sigma's scale."""
    _curves, ause = euq._sparsification(abs_err, sigma)
    return {"spearman_sigma_abserr": euq._spearman(sigma, abs_err),
            "ause": float(ause)}


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--models", default="transformer,dnn", show_default=True,
              help="Comma-separated; each uses its canonical best cell.")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True)
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358",
              show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True)
@click.option("--eval-size", default=20_000, show_default=True)
@click.option("--mc-samples", default=30, show_default=True)
@click.option("--knn-k", default=10, show_default=True)
@click.option("--dropout", default=0.1, show_default=True)
@click.option("--max-seeds", default=0, show_default=True,
              help="0 = every seed in the cell.")
@click.option("--device", default="cuda", show_default=True)
@click.option("--target", default="DMRD", show_default=True)
def main(manifest, models, output_dir, baseline_data_dir, mcmc_data_dir,
         eval_size, mc_samples, knn_k, dropout, max_seeds, device, target):
    import torch
    import importlib.util
    import plot_hit_rate_trajectories_multiseed as phr
    from pmssm.data import compute_stats, load_mcmc_data, transform_y

    spec = importlib.util.spec_from_file_location(
        "euq", str(_REPO_ROOT / "scripts" / "evaluate_uq.py"))
    euq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(euq)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    picks = _resolve_picks(Path(manifest), model_list)

    # ---- eval sets, built exactly as evaluate_uq.py does -------------------
    X_full, Y_full = phr._load_xy_full(baseline_data_dir, target, out_dir)
    static_idx = phr._static_random_indices(len(Y_full), 2000,
                                            static_eval_size=eval_size)
    ev_idx = static_idx[1] if isinstance(static_idx, tuple) else static_idx
    Xs = torch.as_tensor(np.asarray(X_full))[torch.as_tensor(np.asarray(ev_idx))]
    Ys = torch.as_tensor(np.asarray(Y_full)).ravel()[
        torch.as_tensor(np.asarray(ev_idx))]

    Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir, target=target,
                            max_samples=500_000)
    Xm = torch.as_tensor(np.asarray(Xm)).float()
    Ym = torch.as_tensor(np.asarray(Ym)).float().ravel()
    sel = torch.randperm(len(Ym),
                         generator=torch.Generator().manual_seed(123))[:eval_size]
    eval_sets = {"static_random": (Xs.float()[:eval_size], Ys.float()[:eval_size]),
                 "mcmc": (Xm[sel], Ym[sel])}
    for name, (xe, _ye) in eval_sets.items():
        click.echo(f"[lap] eval set {name:<14} n={len(xe)}")

    results: dict = {}
    for model_type, run_dirs in picks.items():
        if max_seeds:
            run_dirs = run_dirs[:max_seeds]
        for run_dir in run_dirs:
            rd = Path(run_dir)
            t0 = time.time()
            state = torch.load(rd / "state.pt", weights_only=False,
                               map_location="cpu")
            run_kwargs = phr._parse_run_kwargs_from_log(rd)
            n_iters = len(list(state.get("al_n_train") or []))
            iter_dir = rd / f"iteration_{n_iters:03d}"
            if not iter_dir.exists():
                click.echo(f"[lap] skip {rd.name}: no {iter_dir.name}", err=True)
                continue
            X_tr, Y_tr, X_va, Y_va = euq._al_train_val(state, n_iters - 1)
            model = phr._load_iter_model(model_type, "al", iter_dir, X_tr, Y_tr,
                                         X_va, Y_va, run_kwargs, device,
                                         dropout=dropout)
            if model is None:
                click.echo(f"[lap] skip {rd.name}: checkpoint missing", err=True)
                continue
            layer = _last_linear(model)
            mean_X, std_X, _mY, _sY = compute_stats(
                X_tr, Y_tr.unsqueeze(-1), torch.arange(len(X_tr)))

            def _norm(x):
                return ((x.float() - mean_X) / std_X)

            # Features on the labelled set, and the noise scale from held-out
            # residuals (training residuals would understate it).
            _mu_tr, Phi_tr = _forward_features(model, _norm(X_tr), device, layer)
            mu_va, Phi_va = _forward_features(model, _norm(X_va), device, layer)
            y_va = transform_y(Y_va.float(), target=target).numpy().ravel()
            sn2 = float(np.mean((y_va - mu_va) ** 2))
            d_feat = Phi_tr.shape[1]
            click.echo(f"\n[lap] {rd.name}")
            click.echo(f"[lap]   N_train={len(X_tr)} d_phi={d_feat} "
                       f"sn2={sn2:.4f} (val RMSE {np.sqrt(sn2):.3f}) "
                       f"iters={n_iters}")

            # lam chosen once, on the validation set, by NLPD.
            best = (np.inf, None)
            usable = []
            for lam in LAMBDA_GRID:
                v = _laplace_var(Phi_tr, Phi_va, sn2, lam)
                if v is None:
                    continue
                usable.append(lam)
                nlpd = float(np.mean(0.5 * np.log(2 * np.pi * v)
                                     + 0.5 * (y_va - mu_va) ** 2 / v))
                if nlpd < best[0]:
                    best = (nlpd, lam)
            if best[1] is None:
                click.echo(f"[lap] skip {rd.name}: no usable lam", err=True)
                continue
            lam_star = best[1]
            edge = ("  <- AT GRID EDGE, widen LAMBDA_GRID"
                    if lam_star in (min(usable), max(usable)) else "")
            click.echo(f"[lap]   lam* = {lam_star:g} (val NLPD {best[0]:.4f})"
                       f"{edge}")

            rec = {"model": model_type, "run_dir": str(rd), "n_train": len(X_tr),
                   "d_phi": int(d_feat), "sn2": sn2, "lam_star": float(lam_star),
                   "val_nlpd": float(best[0]), "iterations": n_iters,
                   "eval": {}}

            for ds, (X_ev, Y_ev) in eval_sets.items():
                Xn = _norm(X_ev)
                mu, Phi_ev = _forward_features(model, Xn, device, layer)
                y_true = transform_y(Y_ev.float(), target=target).numpy().ravel()
                abs_err = np.abs(y_true - mu)          # MAP errors, shared by all
                cands: dict = {}

                # the paper's method, on this subsample
                pred = euq._predict_dropout(model, X_ev, X_tr, Y_tr,
                                            mc_samples, device)
                cands["dropout"] = np.sqrt(np.clip(pred["var"], 1e-12, None))
                own = euq._uq_metrics(y_true, pred)

                for lam in LAMBDA_GRID:
                    v = _laplace_var(Phi_tr, Phi_ev, sn2, lam)
                    if v is not None:
                        cands[f"laplace_lam{lam:g}"] = np.sqrt(v)
                cands["feature_norm"] = np.linalg.norm(
                    Phi_ev.numpy()[:, :-1], axis=1)
                cands["knn_input"] = _knn_dist(Xn, _norm(X_tr), knn_k)
                cands["knn_feature"] = _knn_dist(Phi_ev[:, :-1].float(),
                                                 Phi_tr[:, :-1].float(), knn_k)

                scored = {k: _rank_scores(abs_err, s, euq)
                          for k, s in cands.items()}
                lap = scored[f"laplace_lam{lam_star:g}"]
                rec["eval"][ds] = {
                    "map_rmse": float(np.sqrt(np.mean(abs_err ** 2))),
                    "dropout_own_mean": {k: own[k] for k in
                                         ("rmse", "spearman_sigma_abserr",
                                          "ause", "var_z",
                                          "miscalibration_area", "nlpd")},
                    "candidates": scored,
                    "laplace_at_lam_star": lap,
                }
                click.echo(f"[lap]   {ds:<14} MAP rmse={rec['eval'][ds]['map_rmse']:.3f} "
                           f"| rho: dropout={scored['dropout']['spearman_sigma_abserr']:+.3f} "
                           f"(tab4-style {own['spearman_sigma_abserr']:+.3f}) "
                           f"laplace*={lap['spearman_sigma_abserr']:+.3f} "
                           f"knn_in={scored['knn_input']['spearman_sigma_abserr']:+.3f} "
                           f"knn_phi={scored['knn_feature']['spearman_sigma_abserr']:+.3f}")
                sys.stdout.flush()

            results[rd.name] = rec
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            click.echo(f"[lap]   ({time.time() - t0:.0f}s)")

    p = out_dir / "laplace_uq_probe.json"
    p.write_text(json.dumps({"config": {"models": model_list,
                                        "eval_size": eval_size,
                                        "mc_samples": mc_samples,
                                        "knn_k": knn_k,
                                        "lambda_grid": list(LAMBDA_GRID)},
                             "results": results}, indent=1))
    click.echo(f"\n[lap] wrote {p}")

    # ---- across-seed summary: rho on the target-weighted set ----------------
    click.echo("\n  rho(sigma, |err|), mean over seeds. 'tab4' is dropout scored "
               "against its own\n  mean, i.e. the paper's convention; every other "
               "column shares the MAP errors.")
    for ds in ("mcmc", "static_random"):
        click.echo(f"\n  --- {ds} ---")
        click.echo(f"  {'model':<16}{'tab4':>8}{'dropout':>9}{'laplace*':>10}"
                   f"{'lam best':>10}{'knn_in':>9}{'knn_phi':>9}{'||phi||':>9}{'n':>4}")
        for model_type in model_list:
            rows = [r for r in results.values() if r["model"] == model_type
                    and ds in r["eval"]]
            if not rows:
                continue

            def _m(key, _rows=rows, _ds=ds):
                vals = [r["eval"][_ds]["candidates"][key]["spearman_sigma_abserr"]
                        for r in _rows if key in r["eval"][_ds]["candidates"]]
                return np.mean(vals) if vals else float("nan")

            grid_best = np.mean([
                max(c["spearman_sigma_abserr"] for k, c
                    in r["eval"][ds]["candidates"].items() if k.startswith("laplace_"))
                for r in rows])
            lap_star = np.mean([r["eval"][ds]["laplace_at_lam_star"]
                                ["spearman_sigma_abserr"] for r in rows])
            tab4 = np.mean([r["eval"][ds]["dropout_own_mean"]
                            ["spearman_sigma_abserr"] for r in rows])
            click.echo(f"  {model_type:<16}{tab4:>+8.3f}{_m('dropout'):>+9.3f}"
                       f"{lap_star:>+10.3f}{grid_best:>+10.3f}"
                       f"{_m('knn_input'):>+9.3f}{_m('knn_feature'):>+9.3f}"
                       f"{_m('feature_norm'):>+9.3f}{len(rows):>4}")


if __name__ == "__main__":
    main()
