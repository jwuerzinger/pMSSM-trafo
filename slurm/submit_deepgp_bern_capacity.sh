#!/bin/bash
# Is the Bernoulli deep GP's poor verdict accuracy an optimisation-budget
# problem or the model's genuine answer?
#
# Evidence so far: TRAIN accuracy 0.768 falling to 0.720 against a majority rate
# of 0.733, i.e. it does not fit its own training data; and it early-stops at
# 824 then 204 epochs where the regression arm runs 3172. That is underfitting,
# so the question is what the ELBO does with a longer/faster schedule.
#
# A Bernoulli expected-log-likelihood is bounded (~0.7 nat per point), while a
# Gaussian one at noise=1e-2 can improve by far more, so the KL term carries much
# more relative weight in the Bernoulli ELBO and can pin the variational
# posterior near the prior. lr and patience were both tuned for the Gaussian arm.
#SBATCH --job-name=dgp_lr
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
import numpy as np, torch
sys.path.insert(0, "al_pmssmwithgp/model"); sys.path.insert(0, ".")
from pmssm.data import build_norm_tensors, normalize_x, transform_y
from pmssm.heads import get_head
from pmssm.training import create_gp_model, train_gp_model
from pmssm.visualization import gp_predict

d = Path(sorted(glob.glob("/ptmp/jwuerzin/output/headtest_deepgp_bald_seed1_*"))[-1])
st = torch.load(d / "state.pt", weights_only=False, map_location="cpu")
k = len(st["al_n_train"]) - 1
t, v = int(st["al_n_train"][k]), int(st["al_n_val"][k])
print(f"using iteration {k+1} of the run")
Xtr, Ytr = st["X"][:t], st["Y"][:t].view(-1)
Xva, Yva = st["X_val"][:v], st["Y_val"][:v].view(-1)
dmin, dmax = build_norm_tensors()
xtr = normalize_x(Xtr, dmin, dmax); xva = normalize_x(Xva, dmin, dmax)
head = get_head("classification", threshold=0.0)
ytr_t = transform_y(Ytr, target="ExpR").view(-1)
yva_t = transform_y(Yva, target="ExpR").view(-1)
ytr = head.make_targets(ytr_t); yva = head.make_targets(yva_t)
maj = max(float((ytr > 0).float().mean()), 1 - float((ytr > 0).float().mean()))
print(f"n_train={t} n_val={v}  positive fraction {float((ytr>0).float().mean()):.4f}"
      f"  majority-class accuracy {maj:.4f}\n")

def run(tag, lr, iters, patience, m_ind=256, hid=10):
    torch.manual_seed(0)
    m = create_gp_model("deep_gp", xtr, ytr, xva, yva, n_dim=19, num_samples=8,
                        device="cuda", target="ExpR", head="classification",
                        kernel="RBF", lengthscale=1.0, noise=1e-2, use_ard=True,
                        num_inducing_max=m_ind, num_hidden_dims=hid,
                        num_middle_dims=0)
    m, tl, vl = train_gp_model(m, "deep_gp", lr=lr, iters=iters,
                               batch_size=256, jitter=1e-3, patience=patience)
    best = int(np.argmin(vl)) if len(vl) else -1
    with torch.no_grad():
        ptr = gp_predict(m, xtr, "deep_gp", num_samples=8)
        pva = gp_predict(m, xva, "deep_gp", num_samples=8)
    atr = float(((ptr > 0) == (ytr.cpu() > 0)).float().mean())
    ava = float(((pva > 0) == (yva.cpu() > 0)).float().mean())
    print(f"{tag:<34} epochs_run={len(tl):>5} best_val_epoch={best:>5}  "
          f"ELBO {vl[0]:.4f}->{min(vl):.4f}   train_acc={atr:.4f} val_acc={ava:.4f}")
    del m; torch.cuda.empty_cache()

run("A lr 1e-3, pat 100  (as running)", 1e-3, 10000, 100, 256)
run("B lr 1e-2, pat 300", 1e-2, 10000, 300, 256)
run("B2 lr 1e-2, pat 300 (repeat)", 1e-2, 10000, 300, 256)
PY
