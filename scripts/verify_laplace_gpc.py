"""Verify the Laplace GP classifier against properties stated in Rasmussen &
Williams, not against itself.

Every check here is something the book asserts independently of the code, so a
passing run is evidence the implementation is the published algorithm rather
than evidence that it is self-consistent:

1. **eq. (3.16)**, the probit likelihood derivatives, against finite differences
   of log Phi(y f). The second derivative is checked against a second-order
   stencil, so agreement at the 1e-5 level is the stencil's error, not ours.
2. **eq. (3.17)**, the self-consistency of the mode: f_hat = K grad log p(y|f_hat).
   This is the fixed point Algorithm 3.1 is solving for, and it is stated
   separately from the algorithm, so it is a genuine external check.
3. **Algorithm 3.2 line 4** must return the mode when the test inputs ARE the
   training inputs, since f_bar = k(x*)^T grad log p and f_hat = K grad log p.
4. **eq. (3.25)** in the probit case has the closed form Phi(f_bar/sqrt(1+V)),
   which must agree with the Gauss-Hermite quadrature the acquisition head uses
   for the entropy terms. R&W sanction either.
5. **eq. (3.32)** is a log marginal likelihood, so optimising it must increase
   it, and the resulting classifier must fit a learnable boundary.

It also times a hyperparameter step and extrapolates cubically, because the
cost per step is one Cholesky per Newton iteration and the active-learning loop
grows n by an order of magnitude over 40 iterations.

Usage
-----
    python scripts/verify_laplace_gpc.py                 # CPU, n=300
    python scripts/verify_laplace_gpc.py --n 1600 --device cuda --time-scaling
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "al_pmssmwithgp" / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gp_pipeline.models.laplace_gpc import LaplaceGPC, _probit_derivatives  # noqa: E402
from pmssm.heads import get_head                                            # noqa: E402

FAIL = []


def check(name, value, tol, fmt="{:.2e}"):
    ok = value < tol
    if not ok:
        FAIL.append(name)
    click.echo(f"  [{'ok ' if ok else 'FAIL'}] {name}: {fmt.format(value)} "
               f"(tol {fmt.format(tol)})")


@click.command()
@click.option("--n", default=300, show_default=True, help="Training points.")
@click.option("--dim", default=9, show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--iters", default=40, show_default=True, help="Adam steps.")
@click.option("--time-scaling/--no-time-scaling", default=False,
              help="Time one hyperparameter step and extrapolate cubically.")
def main(n, dim, device, iters, time_scaling):
    torch.manual_seed(0)
    X = torch.rand(n, dim, dtype=torch.float32)
    f_true = (X[:, 0] - 0.5) * 3 + torch.sin(6 * X[:, 1]) - 0.5
    y01 = (f_true + 0.3 * torch.randn(n) > 0).float()
    n_val = max(50, n // 4)
    Xv, yv = X[:n_val], y01[:n_val]
    click.echo(f"n={n} dim={dim} device={device} "
               f"positive fraction {float(y01.mean()):.3f}")

    m = LaplaceGPC(X, y01, Xv, yv, n_dim=dim, lengthscale=0.5, use_ard=True,
                   kernel="RBF", newton_steps=12, device=device)

    click.echo("\n1. eq. (3.16) probit derivatives vs finite differences")
    f = torch.linspace(-6, 6, 13, dtype=torch.float64, device=m.device)
    yy = torch.where(torch.arange(13, device=m.device) % 2 == 0, 1.0, -1.0).double()
    g, W, _ = _probit_derivatives(f, yy)
    h = 1e-5
    ll = lambda fv: torch.special.log_ndtr(yy * fv)
    g_fd = (ll(f + h) - ll(f - h)) / (2 * h)
    W_fd = -(ll(f + h) - 2 * ll(f) + ll(f - h)) / h ** 2
    check("d/df log Phi(y f)", float(((g - g_fd).abs() / g_fd.abs().clamp_min(1e-6)).max()), 1e-6)
    check("W = -d2/df2", float(((W - W_fd).abs() / W_fd.abs().clamp_min(1e-6)).max()), 1e-3)

    click.echo("\n2. eq. (3.17) self-consistency of the mode")
    with torch.no_grad():
        K = m._K(m.x_train)
        fh, a, log_q, L, sW, grad = m._newton(K, m.y_train)
    # eq. (3.17) is stated for a zero-mean prior; with a constant mean the
    # self-consistency condition centres on it: f_hat - m = K grad log p.
    m_const = m.mean_const.detach().to(fh.dtype)
    check("|(f_hat - m) - K grad log p| / |f_hat - m|",
          float(((fh - m_const) - K @ grad).abs().max()
                / (fh - m_const).abs().max()), 1e-4)
    click.echo(f"        log q(y|X,theta) = {float(log_q):.4f}")

    click.echo("\n3. Algorithm 3.2 line 4 returns the mode at training inputs")
    m.refresh_mode()
    with torch.no_grad():
        f_bar, var = m._predict_latent(m.x_train)
    check("max |f_bar - f_hat|", float((f_bar - fh).abs().max()), 1e-3)

    click.echo("\n4. eq. (3.25): analytic probit form vs the head's quadrature")
    head = get_head("classification", link="probit")
    with torch.no_grad():
        p_analytic = m.predictive_probability(m.x_valid)
        fb, vb = m._predict_latent(m.x_valid)
        s = head.summarise_gaussian(fb.float(), vb.float())
    check("max |Phi(f/sqrt(1+V)) - quadrature|",
          float((p_analytic.float() - s["p_mean"]).abs().max()), 1e-5)
    mi = s["mutual_information"]
    click.echo(f"        BALD range {float(mi.min()):.5f} .. {float(mi.max()):.5f}; "
               f"entropy range {float(s['entropy'].min()):.4f} .. "
               f"{float(s['entropy'].max()):.4f}")
    if not (mi >= 0).all():
        FAIL.append("BALD non-negative")
        click.echo("  [FAIL] BALD must be non-negative (it is a mutual information)")

    click.echo("\n5. eq. (3.32) increases under optimisation, and the fit learns")
    with torch.no_grad():
        log_q0 = float(m.refresh_mode())
    t0 = time.time()
    m, tr, va = m.do_train_loop(lr=0.05, iters=iters, patience=None)
    dt = time.time() - t0
    with torch.no_grad():
        log_q1 = float(m.refresh_mode())
        p = m.predictive_probability(m.x_train)
        acc = float((((p > 0.5).double() * 2 - 1) == m.y_train).double().mean())
    check("log q must increase (reported as the decrease)",
          max(0.0, log_q0 - log_q1), 1e-9, fmt="{:.3e}")
    click.echo(f"        log q {log_q0:.3f} -> {log_q1:.3f}; train accuracy {acc:.4f}")
    if acc < 0.85:
        FAIL.append("train accuracy")
        click.echo(f"  [FAIL] accuracy {acc:.3f} too low for a learnable boundary")

    if time_scaling:
        per_step = dt / max(len(tr), 1)
        click.echo(f"\n6. cost: {per_step * 1000:.1f} ms per hyperparameter step at "
                   f"n={n}")
        for n_target in (5000, 10000, 14000, 20000):
            scaled = per_step * (n_target / n) ** 3
            click.echo(f"        extrapolated n={n_target:>6}: {scaled:7.3f} s/step "
                       f"-> {scaled * 400 / 60:7.1f} min for 400 steps")

    click.echo("")
    if FAIL:
        click.echo(f"FAILED: {', '.join(FAIL)}")
        raise SystemExit(1)
    click.echo("all checks passed")


if __name__ == "__main__":
    main()
