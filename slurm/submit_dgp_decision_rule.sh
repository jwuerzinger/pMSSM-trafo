#!/bin/bash
# Is the Bernoulli deep GP's verdict being scored with the WRONG decision rule?
#
# The accuracy diagnostics threshold gp_predict's output at 0. For a Bernoulli
# deep GP that output is the mean LATENT averaged over sample paths, so the rule
# is sign(mean_s mu_s). The model's own Bayes rule is p_bar > 0.5 with
# p_bar = mean_s Phi(mu_s / sqrt(1 + var_s)), and by Jensen those two are NOT
# the same decision for a mixture. The regression arm, by contrast, is scored
# with exactly the right rule (sign of the predicted t), so any gap here is an
# unfair asymmetry between the arms rather than a property of the head.
#SBATCH --job-name=dgp_rule
#SBATCH --partition=apudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/al_pmssmwithgp/model:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

.pixi/envs/rocm/bin/python - <<'PY'
import sys, glob
from pathlib import Path
import numpy as np, torch, gpytorch
sys.path.insert(0, "scripts"); sys.path.insert(0, "al_pmssmwithgp/model"); sys.path.insert(0, ".")
from plot_hit_rate_trajectories_multiseed import (
    _load_iter_model, _load_xy_full, _parse_run_kwargs_from_log,
    _static_random_indices)
from pmssm.data import build_norm_tensors, normalize_x, transform_y

X_full, Y_full = _load_xy_full("/ptmp/jwuerzin/data/260804", "ExpR",
                               Path("/ptmp/jwuerzin/analysis/pool_cache"))
idx = _static_random_indices(len(Y_full), 2000, static_eval_size=100_000)
Xs = torch.from_numpy(np.asarray(X_full[idx], dtype=np.float32))
Ys = torch.from_numpy(np.asarray(Y_full[idx], dtype=np.float32))
dmin, dmax = build_norm_tensors()
Xn = normalize_x(Xs, dmin, dmax)
yt = transform_y(Ys, target="ExpR").view(-1).numpy()
truth = yt >= 0.0
print(f"eval rows {len(idx)}, positive fraction {truth.mean():.4f}, "
      f"majority {max(truth.mean(), 1-truth.mean()):.4f}\n")

d = Path(sorted(glob.glob("/ptmp/jwuerzin/output/headtest_deepgp_clsent_seed1_*"))[-1])
st = torch.load(d / "state.pt", weights_only=False, map_location="cpu")
rk = _parse_run_kwargs_from_log(d)
print(f"run {d.name}  head={rk.get('head')}\n")
print(f"{'iter':>5} {'sign(mean latent)':>19} {'p_bar>0.5 (Bayes)':>19} {'diff':>8}")
for k in [4, 9, 14, len(list(st['al_n_train'])) - 1]:
    it = k + 1
    idir = d / f"iteration_{it:03d}"
    if not (idir / "al_model_checkpoint.pt").exists():
        continue
    t, v = int(st["al_n_train"][k]), int(st["al_n_val"][k])
    m = _load_iter_model("deep_gp", "al", idir, st["X"][:t], st["Y"][:t].view(-1),
                         st["X_val"][:v], st["Y_val"][:v].view(-1), rk, "cuda")
    lat_mu, pbar = [], []
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(8), \
         gpytorch.settings.fast_pred_var(False), \
         gpytorch.settings.cholesky_jitter(float_value=1e-3, double_value=1e-3):
        for i in range(0, len(Xn), 1024):
            xb = Xn[i:i+1024].to("cuda")
            lat = m(xb)                       # (S, N) latent
            mu, var = lat.mean.detach(), lat.variance.detach()
            lat_mu.append(mu.mean(dim=0).cpu())
            # p_bar = mean_s Phi(mu_s / sqrt(1+var_s)) -- the mixture's own rule
            p = torch.special.ndtr(mu / torch.sqrt(1.0 + var))
            pbar.append(p.mean(dim=0).cpu())
    a = float(((torch.cat(lat_mu).numpy() > 0) == truth).mean())
    b = float(((torch.cat(pbar).numpy() > 0.5) == truth).mean())
    print(f"{it:>5} {a:>19.4f} {b:>19.4f} {b-a:>+8.4f}")
    del m; torch.cuda.empty_cache()
PY
