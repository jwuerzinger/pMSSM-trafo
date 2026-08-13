"""Why is the AL-trained Deep GP's prediction spread in the posterior region
larger than its random baseline's?

The MCMC eval set is near-constant in the target (its spread is the reference
sampler's likelihood width, sigma = 0.012 on a central 0.120), so the MSE
evaluated there reduces to  bias^2 + Var(prediction).  Measured at iteration 40
(seed 2) the AL model's prediction spread is 1.36 against the baseline's 1.00,
and that difference is the whole MSE gap.  Four candidate causes have already
been refuted by post-hoc measurement on the saved checkpoints:

  * AL's training data looks steeper            (|dt|/dd 4.86 vs 6.08: smoother)
  * the baseline reverts to prior for lack of data (predictive var flat in x)
  * AL data exceeds the Deep GP's fitting capacity (train-fit ratio uncorrelated
    with the MCMC ratio across six architectures, Spearman +0.14)
  * AL's on-shell points are more numerous / more diverse (numerous yes, 11x;
    diverse no, 0.438 vs 0.446 internal spread at matched count; and the
    Transformer has the same 9x with the spread going the other way)

This script settles it by intervention instead of observation: retrain the same
architecture, cold and at matched size, on training sets that differ in one
controlled way at a time, then measure the prediction spread in the posterior
region.

Arms
----
al                the actual AL training set (reference; expect spread ~1.36)
random            the actual random baseline set (reference; expect ~1.00)
stratified        pool points resampled so their Omega histogram matches the AL
                  set's, but selected WITHOUT AL's input-space preference.
                  Isolates the target distribution from the input placement:
                  if this behaves like `al` the cause is the target
                  distribution, if like `random` it is where AL puts its points.
random_plus_inband  the random set with a random 1195 of its points replaced by
                  the AL set's 1195 in-band points.  The complementary
                  ablation: what does adding on-shell points alone do?
al_ind1024        `al` with 1024 inducing points instead of 256 (capacity test
                  done properly, by retraining rather than via a train-loss proxy)
random_ind1024    `random` with 1024 inducing points (capacity control)

Usage
-----
    python scripts/dispersion_experiment.py --arm stratified --seed 1
    python scripts/dispersion_experiment.py --arm al --seed 1 --smoke
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "al_pmssmwithgp" / "model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pmssm.data import (  # noqa: E402
    build_norm_tensors,
    normalize_x,
    transform_y,
)
from pmssm.training import create_gp_model, train_gp_model  # noqa: E402
from pmssm.visualization import gp_predict  # noqa: E402

CACHE = Path("/ptmp/jwuerzin/analysis/all_runs")
POOL_X = CACHE / "x_full_ptmp_jwuerzin_data_18387358_DMRD.npy"
POOL_Y = CACHE / "y_full_ptmp_jwuerzin_data_18387358_DMRD.npy"
MCMC_X = CACHE / "mcmc_eval_x_ptmp_jwuerzin_data_neutralino_v4_DMRD_veto0_n500000_src35261440.npy"
MCMC_Y = CACHE / "mcmc_eval_y_ptmp_jwuerzin_data_neutralino_v4_DMRD_veto0_n500000_src35261440.npy"
# The reference run every arm is matched against. Its own AL and baseline
# training sets are arms `al` and `random`, so the two references are
# reproduced inside the same experiment rather than quoted from the sweep.
REF_RUN = Path("/ptmp/jwuerzin/output/active_learning_deep_gp_entropy_batch_warm_seed2_20260803_180047")
REF_SHUFFLE_SEED = 2      # the run's --seed, which fixes the pool shuffle
N_RESERVED = 2000         # rows [0..1999] are the shared initial split
N_EVAL = 30000            # eval subsample size, fixed across arms
ARMS = ("al", "random", "stratified", "random_plus_inband",
        "random_plus_al_outofband", "al_ind1024", "random_ind1024")


def _load_pool():
    """The valid random pool in the reference run's shuffled order."""
    X = torch.tensor(np.load(POOL_X), dtype=torch.float32)
    Y = torch.tensor(np.load(POOL_Y), dtype=torch.float32).view(-1)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(REF_SHUFFLE_SEED))
    return X[perm], Y[perm]


def _reference_sets(state, Xp, Yp):
    """(AL train, AL val, baseline train, baseline val) from the reference run.

    The baseline side is reconstructed from the recorded draw indices; the row
    counts are asserted against the run's own bookkeeping so a silent
    misalignment cannot pass as a result.
    """
    X_al, Y_al = state["X"], state["Y"].view(-1)
    X_al_val, Y_al_val = state["X_val"], state["Y_val"].view(-1)
    add = state["baseline_add_indices"]
    n_tr = int(state["prev_n_add_train"])
    X_b = torch.cat([Xp[:1600], Xp[add[:n_tr]]], dim=0)
    Y_b = torch.cat([Yp[:1600], Yp[add[:n_tr]]], dim=0)
    X_b_val = torch.cat([Xp[1600:N_RESERVED], Xp[add[n_tr:]]], dim=0)
    Y_b_val = torch.cat([Yp[1600:N_RESERVED], Yp[add[n_tr:]]], dim=0)
    expected = int(state["baseline_n_train"][-1])
    if len(X_b) != expected:
        raise RuntimeError(f"baseline reconstruction gives {len(X_b)} rows, "
                           f"the run recorded {expected}")
    return (X_al, Y_al), (X_al_val, Y_al_val), (X_b, Y_b), (X_b_val, Y_b_val)


def _stratified_like(Y_target, Xp, Yp, forbidden, rng, n_bins=40):
    """Pool rows whose log-Omega histogram matches `Y_target`'s.

    Bins are equal-width in t = log(Omega/0.12) over the target's range; each
    bin draws (without replacement) the same number of pool rows the target has
    there.  Rows in `forbidden` (the reserved initial split) are excluded.  A
    bin the pool cannot fill is topped up from the nearest bins that can, and
    the shortfall is reported so an unmatchable histogram cannot pass silently.
    """
    t_tgt = np.log(Y_target.numpy() / 0.12)
    t_pool = np.log(Yp.numpy() / 0.12)
    edges = np.linspace(t_tgt.min() - 1e-6, t_tgt.max() + 1e-6, n_bins + 1)
    avail_mask = np.ones(len(t_pool), dtype=bool)
    avail_mask[forbidden] = False

    picks, shortfall = [], 0
    want = np.histogram(t_tgt, bins=edges)[0]
    for b in range(n_bins):
        idx = np.flatnonzero((t_pool >= edges[b]) & (t_pool < edges[b + 1]) & avail_mask)
        take = min(want[b], len(idx))
        if take:
            chosen = rng.choice(idx, size=take, replace=False)
            picks.append(chosen)
            avail_mask[chosen] = False
        shortfall += want[b] - take
    picks = np.concatenate(picks) if picks else np.array([], dtype=int)

    if shortfall:
        # Top up from anywhere still available, nearest-in-t first, so the total
        # size still matches even where the pool cannot reproduce the shape.
        remaining = np.flatnonzero(avail_mask)
        order = np.argsort(np.abs(t_pool[remaining] - np.median(t_tgt)))
        picks = np.concatenate([picks, remaining[order[:shortfall]]])
    return torch.tensor(picks, dtype=torch.long), int(shortfall)


def _build_arm(arm, state, Xp, Yp, rng):
    """Return (X_train, Y_train, X_val, Y_val, num_inducing, provenance dict)."""
    (X_al, Y_al), (X_al_v, Y_al_v), (X_b, Y_b), (X_b_v, Y_b_v) = _reference_sets(state, Xp, Yp)
    n_ind = 1024 if arm.endswith("ind1024") else 256
    prov = {}

    if arm in ("al", "al_ind1024"):
        return X_al, Y_al, X_al_v, Y_al_v, n_ind, prov
    if arm in ("random", "random_ind1024"):
        return X_b, Y_b, X_b_v, Y_b_v, n_ind, prov

    forbidden = np.arange(N_RESERVED)
    if arm == "stratified":
        # Match the AL train and val Omega histograms separately so the val
        # split used for early stopping has the same distribution as training,
        # exactly as it does on both reference arms.
        i_tr, sf_tr = _stratified_like(Y_al, Xp, Yp, forbidden, rng)
        i_va, sf_va = _stratified_like(Y_al_v, Xp, Yp,
                                       np.concatenate([forbidden, i_tr.numpy()]), rng)
        prov = {"shortfall_train": sf_tr, "shortfall_val": sf_va}
        return Xp[i_tr], Yp[i_tr], Xp[i_va], Yp[i_va], n_ind, prov

    if arm in ("random_plus_inband", "random_plus_al_outofband"):
        band = (torch.abs(Y_al - 0.12) / 0.12) < 0.10
        n_band = int(band.sum())
        if arm == "random_plus_inband":
            Xi, Yi = X_al[band], Y_al[band]
        else:
            # The same substitution count, drawn from AL's acquisition MISSES
            # instead of its hits. `random_plus_inband` showed the hits are
            # harmless; this asks whether the misses carry the degradation.
            oob = (~band).nonzero(as_tuple=True)[0]
            pick = torch.tensor(rng.choice(len(oob), size=n_band, replace=False),
                                dtype=torch.long)
            Xi, Yi = X_al[oob[pick]], Y_al[oob[pick]]
        k = len(Xi)
        keep = torch.tensor(rng.choice(len(X_b), size=len(X_b) - k, replace=False),
                            dtype=torch.long)
        prov = {"n_inband_substituted": k}
        return (torch.cat([X_b[keep], Xi]), torch.cat([Y_b[keep], Yi]),
                X_b_v, Y_b_v, n_ind, prov)

    raise click.ClickException(f"unknown arm {arm!r}")


def _evaluate(model, X, Y, dmin, dmax, num_samples, jitter):
    """MSE and its bias/variance split in transformed target space."""
    t = transform_y(Y, target="DMRD").view(-1)
    th = gp_predict(model, normalize_x(X, dmin, dmax), "deep_gp",
                    jitter=jitter, num_samples=num_samples).view(-1).cpu()
    err = t - th
    return {
        "mse": float((err ** 2).mean()),
        "bias": float(err.mean()),
        "bias_sq": float(err.mean() ** 2),
        "pred_mean": float(th.mean()),
        "pred_std": float(th.std()),
        "pred_var": float(th.var()),
        "true_std": float(t.std()),
        "corr": float(np.corrcoef(t.numpy(), th.numpy())[0, 1]),
        "n": int(len(t)),
    }


def _geometry(Xtr, Xm, dmin, dmax):
    """Input-space relation between an arm's training set and the eval set.

    Recorded so a dispersion result can be read against how much data the arm
    actually has near the posterior region: the two candidate explanations
    (target distribution vs input placement) are distinguished by whether an
    arm reproduces AL's dispersion with AL-like or baseline-like density here.
    """
    free = [0, 1, 10, 11, 12, 14, 15, 16, 18]   # the 9 non-degenerate params
    q = normalize_x(Xm, dmin, dmax)[:, free]
    r = normalize_x(Xtr, dmin, dmax)[:, free]
    nn, cnt = [], []
    for k in range(0, len(q), 512):
        d = torch.cdist(q[k:k + 512], r)
        nn.append(d.min(dim=1).values)
        cnt.append((d < 0.25).float().sum(dim=1))
    nn, cnt = torch.cat(nn), torch.cat(cnt)
    return {"nn_median": float(nn.median()), "n_within_0.25_mean": float(cnt.mean())}


@click.command()
@click.option("--arm", type=click.Choice(ARMS), required=True)
@click.option("--seed", type=int, default=1, show_default=True,
              help="Model init / training seed; the data construction uses it too.")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/dispersion_experiment",
              show_default=True)
@click.option("--epochs", default=10000, show_default=True)
@click.option("--patience", default=100, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--batch-size", default=256, show_default=True)
@click.option("--jitter", default=1e-3, show_default=True)
@click.option("--num-samples", default=8, show_default=True)
@click.option("--smoke", is_flag=True,
              help="Tiny subsample and a few epochs, to check the plumbing.")
def main(arm, seed, output_dir, epochs, patience, lr, batch_size, jitter,
         num_samples, smoke):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"[cfg] arm={arm} seed={seed} device={device} smoke={smoke}")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dmin, dmax = build_norm_tensors()

    state = torch.load(REF_RUN / "state.pt", weights_only=False, map_location="cpu")
    Xp, Yp = _load_pool()
    Xtr, Ytr, Xva, Yva, n_ind, prov = _build_arm(arm, state, Xp, Yp, rng)

    if smoke:
        # Subsample at random, not from the head: every arm's set begins with
        # the shared initial random split, so a head slice would report the
        # pool's target distribution for every arm and hide a construction bug.
        s_tr = torch.tensor(rng.choice(len(Xtr), 600, replace=False), dtype=torch.long)
        s_va = torch.tensor(rng.choice(len(Xva), 150, replace=False), dtype=torch.long)
        Xtr, Ytr, Xva, Yva = Xtr[s_tr], Ytr[s_tr], Xva[s_va], Yva[s_va]
        epochs, n_ind = 30, 64

    t_tr = transform_y(Ytr, target="DMRD").view(-1)
    click.echo(f"[data] n_train={len(Xtr)} n_val={len(Xva)} num_inducing={n_ind} "
               f"median_Omega={float(Ytr.median()):.4f} std_t={float(t_tr.std()):.3f} "
               f"inband_frac={float(((Ytr-0.12).abs()/0.12 < 0.10).float().mean()):.4f}"
               + (f" prov={prov}" if prov else ""))

    model = create_gp_model(
        "deep_gp",
        normalize_x(Xtr, dmin, dmax), t_tr,
        normalize_x(Xva, dmin, dmax), transform_y(Yva, target="DMRD").view(-1),
        n_dim=Xtr.shape[1], kernel="RBF", lengthscale=1.0, noise=1e-2,
        num_hidden_dims=10, num_middle_dims=0, num_inducing_max=n_ind,
        num_samples=num_samples, seed=seed, device=device, target="DMRD",
    )
    model, train_losses, val_losses = train_gp_model(
        model, "deep_gp", lr=lr, iters=epochs, batch_size=batch_size,
        jitter=jitter, patience=patience if not smoke else None,
    )
    click.echo(f"[train] {len(train_losses)} epochs, final train={train_losses[-1]:.4f} "
               f"val={val_losses[-1]:.4f} ({time.time()-t0:.0f}s)")

    ev_rng = np.random.default_rng(0)   # same eval subsample for every arm
    Ym_all = np.load(MCMC_Y).ravel()
    im = ev_rng.choice(len(Ym_all), min(N_EVAL, len(Ym_all)), replace=False)
    Xm = torch.tensor(np.load(MCMC_X)[im], dtype=torch.float32)
    Ym = torch.tensor(Ym_all[im], dtype=torch.float32)
    ip = ev_rng.choice(len(Yp), min(N_EVAL, len(Yp)), replace=False)
    if smoke:
        Xm, Ym, ip = Xm[:400], Ym[:400], ip[:400]

    res = {
        "arm": arm, "seed": seed, "smoke": smoke,
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "num_inducing": int(n_ind),
        "train_median_omega": float(Ytr.median()),
        "train_std_t": float(t_tr.std()),
        "train_inband_frac": float(((Ytr - 0.12).abs() / 0.12 < 0.10).float().mean()),
        "provenance": prov,
        "epochs_run": int(len(train_losses)),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "geometry_vs_mcmc": _geometry(Xtr, Xm[:2000], dmin, dmax),
        "mcmc": _evaluate(model, Xm, Ym, dmin, dmax, num_samples, jitter),
        "pool": _evaluate(model, Xp[ip], Yp[ip], dmin, dmax, num_samples, jitter),
        "wall_seconds": round(time.time() - t0, 1),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{arm}_seed{seed}{'_smoke' if smoke else ''}.json"
    path.write_text(json.dumps(res, indent=2))

    m, p = res["mcmc"], res["pool"]
    click.echo(f"[mcmc] MSE={m['mse']:.4f} = bias^2 {m['bias_sq']:.4f} + "
               f"Var(pred) {m['pred_var']:.4f}   pred_std={m['pred_std']:.3f} "
               f"corr={m['corr']:+.3f}")
    click.echo(f"[pool] MSE={p['mse']:.4f} pred_std={p['pred_std']:.3f} "
               f"corr={p['corr']:+.3f}")
    click.echo(f"[done] {path} ({res['wall_seconds']}s)")


if __name__ == "__main__":
    main()
