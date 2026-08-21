#!/bin/bash
# Where does the Laplace hyperparameter-step cost go: Cholesky flops, or the
# autograd graph through the unrolled Newton loop? The cubic extrapolation from
# n=1600 predicted 185 s/step at n=14000, ~160x slower than a Cholesky of that
# size should be here, and the fix differs completely depending on the answer.
#
# The script is inlined rather than read from a path: /tmp is node-local, so a
# scratchpad file written on the login node does not exist on the compute node.
#SBATCH --job-name=laplace_timing
#SBATCH --partition=apudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/al_pmssmwithgp/model:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

.pixi/envs/rocm/bin/python - <<'PY'
import sys, time
sys.path.insert(0, "al_pmssmwithgp/model")
import torch
from gp_pipeline.models.laplace_gpc import LaplaceGPC

dev = "cuda"
torch.manual_seed(0)

def timeit(fn, reps=2):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps

print(f"{'n':>7} {'chol fp64':>10} {'chol fp32':>10} {'B build':>9} "
      f"{'newton nograd':>14} {'newton+grad':>12} {'peak GB':>8}")
for n in (2000, 5000, 10000, 14000):
    A = torch.randn(n, n, device=dev, dtype=torch.float64)
    K = (A @ A.T) / n + torch.eye(n, device=dev, dtype=torch.float64)
    del A; torch.cuda.empty_cache()
    t_c64 = timeit(lambda: torch.linalg.cholesky(K))
    K32 = K.float()
    t_c32 = timeit(lambda: torch.linalg.cholesky(K32))
    del K32; torch.cuda.empty_cache()
    sW = torch.rand(n, device=dev, dtype=torch.float64) + 0.5
    eye = torch.eye(n, device=dev, dtype=torch.float64)
    t_B = timeit(lambda: eye + sW.unsqueeze(-1) * K * sW.unsqueeze(-2))
    del K, eye; torch.cuda.empty_cache()

    X = torch.rand(n, 9, device=dev)
    y = (torch.rand(n, device=dev) > 0.6).float()
    m = LaplaceGPC(X, y, X[:100], y[:100], n_dim=9, lengthscale=1.0,
                   use_ard=True, kernel="RBF", newton_steps=8, device=dev)
    with torch.no_grad():
        Kk = m._K(m.x_train)
        t_ng = timeit(lambda: m._newton(Kk, m.y_train), reps=1)
    del Kk; torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    def grad_step():
        Kk2 = m._K(m.x_train)
        out = m._newton(Kk2, m.y_train)
        m.zero_grad(set_to_none=True)
        (-out[2]).backward()
    try:
        t_g = timeit(grad_step, reps=1)
        peak = torch.cuda.max_memory_allocated() / 1e9
    except RuntimeError as e:
        t_g, peak = float('nan'), float('nan')
        print(f"   n={n}: grad step failed: {str(e)[:80]}")
    print(f"{n:>7} {t_c64:>10.3f} {t_c32:>10.3f} {t_B:>9.3f} {t_ng:>14.3f} "
          f"{t_g:>12.3f} {peak:>8.1f}")
    del m; torch.cuda.empty_cache()
PY
