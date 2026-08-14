"""Does the acquisition's uncertainty ranking add anything beyond the hard cut?

Selection keeps candidates with |predicted log r| < tolerance, then ranks the
survivors by uncertainty (times a proximity weight) and takes the top n. The
question is whether that ranking lands nearer the true boundary than a random
pick from the same survivors.

Candidates were never simulated, so the pool stands in for the candidate stream:
its points carry true labels, so the model's own cut and ranking can be scored.

  base    true in-band rate among all points passing |pred| < tol
  var     same rate among the n highest-uncertainty survivors
  px*var  same, with the proximity weight selection actually applies
  lift    ratio to base; 1.00 means the ranking is inert

Run for each model's canonical cell (transformer entropy_batch/cold,
deep_gp entropy_batch/warm) so the two are compared at their best.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/viper/u2/jwuerzin/pMSSM-trafo")
sys.path.insert(0, "/viper/u2/jwuerzin/pMSSM-trafo/scripts")

import plot_hit_rate_trajectories_multiseed as phr            # noqa: E402
from pmssm.data import build_norm_tensors, compute_stats      # noqa: E402
from pmssm.uncertainty import (                               # noqa: E402
    compute_uncertainty_gp, compute_uncertainty_mc_dropout)

SWEEP = "20260813_183646"
ARMS = [
    ("transformer", "entropy_batch", "cold"),
    ("deep_gp",     "entropy_batch", "warm"),
]
ITERS = [1, 10, 20, 30, 40]
TOL_SEL, PROX, BAND, N_SELECT = 1.0, 0.1, 0.10, 500
N_POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

Xp_all = np.load("/ptmp/jwuerzin/analysis/expr_runs/"
                 "x_full_ptmp_jwuerzin_data_260804_ExpR.npy")
Yp_all = np.load("/ptmp/jwuerzin/analysis/expr_runs/"
                 "y_full_ptmp_jwuerzin_data_260804_ExpR.npy").ravel()
rng = np.random.default_rng(20260814)
sub = rng.choice(len(Yp_all), N_POOL, replace=False)
Xp = torch.from_numpy(np.asarray(Xp_all[sub], dtype=np.float32))
Yp = Yp_all[sub]
inband_all = np.abs(Yp - 1.0) < BAND
print(f"device {DEV}   pool slice {N_POOL:,}   true in-band {inband_all.mean():.4f}\n")

for model, strat, warm in ARMS:
    base_dir = f"/ptmp/jwuerzin/output/active_learning_{model}_expr_{strat}_{warm}_seed1_{SWEEP}"
    if not Path(f"{base_dir}/state.pt").exists():
        print(f"=== {model}/{strat}/{warm}: no run\n")
        continue
    state = torch.load(f"{base_dir}/state.pt", weights_only=False, map_location="cpu")
    run_kwargs = phr._parse_run_kwargs_from_log(Path(base_dir) / "active_learning.log")
    n_tr = list(state.get("al_n_train") or [])
    print(f"{'='*82}\n{model}/{strat}/{warm}   {len(n_tr)} iterations\n{'='*82}")
    print(f"{'iter':>5s} {'|L|':>7s} {'survive':>8s} {'base':>7s} "
          f"{'var':>7s} {'lift':>6s} {'px*var':>8s} {'lift':>6s} {'mean sigma':>11s} {'sd sigma':>9s}")
    for it in ITERS:
        if it > len(n_tr):
            continue
        idir = Path(base_dir) / f"iteration_{it:03d}"
        if not (idir / "al_model_checkpoint.pt").exists():
            continue
        ntr, nva = int(state["al_n_train"][it - 1]), int(state["al_n_val"][it - 1])
        Xtr, Ytr = state["X"][:ntr], state["Y"][:ntr]
        Xva, Yva = state["X_val"][:nva], state["Y_val"][:nva]
        try:
            m = phr._load_iter_model(model, "al", idir, Xtr, Ytr, Xva, Yva,
                                     run_kwargs, DEV)
        except Exception as e:
            print(f"{it:5d}   load failed: {type(e).__name__}: {e}")
            continue
        try:
            if model in ("transformer", "dnn"):
                stats = compute_stats(Xtr, Ytr, torch.arange(len(Xtr)))
                mean, var = compute_uncertainty_mc_dropout(m, Xp, stats, 30, DEV, None)
            else:
                dmin, dmax = build_norm_tensors()
                mean, var = compute_uncertainty_gp(m, Xp, dmin, dmax, model,
                                                   jitter=1e-3, num_samples=8)
        except Exception as e:
            print(f"{it:5d}   predict failed: {type(e).__name__}: {e}")
            continue
        mu = np.asarray(mean).ravel().astype(float)
        vr = np.asarray(var).ravel().astype(float)
        surv = np.abs(mu) < TOL_SEL
        if surv.sum() < N_SELECT:
            print(f"{it:5d} {ntr+nva:7d} {int(surv.sum()):8d}   too few survivors")
            continue
        idx = np.flatnonzero(surv)
        base = inband_all[idx].mean()
        top_v = idx[np.argsort(-vr[idx])[:N_SELECT]]
        w = np.exp(-(mu[idx] ** 2) / PROX) * vr[idx]
        top_w = idx[np.argsort(-w)[:N_SELECT]]
        rv, rw = inband_all[top_v].mean(), inband_all[top_w].mean()
        sig = np.sqrt(np.maximum(vr[idx], 0))
        print(f"{it:5d} {ntr+nva:7d} {int(surv.sum()):8d} {base:7.4f} "
              f"{rv:7.4f} {rv/base if base else np.nan:6.2f} "
              f"{rw:8.4f} {rw/base if base else np.nan:6.2f} "
              f"{sig.mean():11.4f} {sig.std():9.4f}")
    print()
