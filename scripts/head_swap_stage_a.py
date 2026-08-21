"""Stage A of the classification-versus-regression question: does the acquisition
head change what the loop sees, at fixed data and with no simulator calls?

The literature's active-learning gains on high-dimensional BSM spaces are
reported for *classifiers* of a discrete label (Caron et al. 2019 label points
0/1 excluded/allowed and say explicitly that they consider classification
because it has a clear region of interest, the decision boundary; Goodsell and
Joury train binary cross-entropy on an oracle's good/bad verdict; Rocamonde et
al. discretise a continuous confidence level into three classes).  Our loop
regresses the observable and thresholds at read-out, and on the SModelS
exclusion target the dropout surrogates never reach their own random-selection
baseline (Transformer 0.8941 vs 0.8963, DNN 0.8551 vs 0.8595) while the two GPs
do (2.2x and 1.5x data efficiency).  Two hypotheses:

  H1 (framing)   the regression loss puts almost all of its squared-error mass
                 far from the contour -- the training target t = log r spans
                 27 nats while the band is |t| < 0.1 -- and MC-dropout sigma
                 ranks fit difficulty, which on r_exp = max_a r_a lives on the
                 argmax-switching ridges rather than at r = 1.  A classification
                 head on the same trunk addresses both: cross-entropy weights
                 points by verdict uncertainty, and the score is anchored on the
                 boundary.
  H0 (locality)  the defect is dropout's blindness to distance from labelled
                 data, which no head swap can repair (the K=5 ensemble and the
                 last-layer Laplace substitutions both failed to buy yield).

This script decides which hypothesis is worth an AL run, without spending a
single simulator call, by retraining both heads on training sets the AL runs
have *already* acquired and scoring the labelled random pool with each head's
own acquisition rule.  The pool is labelled, so "what would this head have
picked, and how good were those picks" is answerable offline.

Both arms use the shared, production implementations: the heads come from
:mod:`pmssm.heads`, the MC-dropout sampling from
:func:`pmssm.uncertainty.compute_uncertainty_head`, and the ranking from
:mod:`pmssm.selection`.  Nothing about the head lives only in this script, so
whatever Stage A measures is what a production run with that head would do.

Arms (identical trunk, identical optimiser, identical data; only the head
differs, and the read-out layer has the same shape in both):

  regression      MSELoss on t = log(Y / true_value)          [production]
  classification  BCEWithLogitsLoss on 1[t > 0], output read as a logit

Measurements
------------
M1  precision@500 for band membership on a held-out slice of the labelled pool,
    under each arm's own acquisition rule.  This is the acquisition step's yield
    per *valid* point, computed without the simulator.  Selectors:
      reg_topk      tolerance cut |t_hat| < 1.0, then proximity-weighted
                    variance exp(-t_hat^2 / 0.1) * sigma^2   [production top_k]
      reg_var_only  tolerance cut, then raw variance   [production top_k_tol_only]
      reg_entropy_batch  the production entropy_batch cell, which is the
                    paper's headline cell for the dropout surrogates
      prefilter_random   top_k_tol_only with the variance ranking replaced by a
                    uniform draw: what the cut is worth without any ranking
      cls_entropy   predictive entropy H[p_bar] of the MC-dropout mean
                    probability, i.e. "committee mean nearest 0.5"
      cls_bald      H[p_bar] - E_t H[p_t], the epistemic part alone
    The last two together test the decomposition H[E p] = I(y;theta) + E H[p]:
    if cls_entropy beats cls_bald, the classifier's signal is its mean, not its
    committee disagreement, which is what the ensemble null result predicts.
M2  verdict accuracy on one fixed static-random evaluation split, globally and
    restricted to shells in |t|, so "did the head buy boundary localisation" is
    separated from "did it buy bulk accuracy".  Both heads are scored by
    thresholding the raw output at zero, exactly as the per-iteration accuracy
    diagnostics do, which is why those diagnostics need no change for a
    classification-head run.
M3  one-step informativeness: retrain on L + the arm's own 500 picks and report
    the change in M2.  Optional (costs extra trainings per snapshot).
M4  band share of the regression loss: the fraction of the summed squared
    residual contributed by in-band points.  Quantifies H1's first half, and is
    comparable across targets when the script is run with --target DMRD.
M5  uncertainty profile against |t|: sigma (regression) or predictive entropy
    (classification) in bins of distance to the contour, which is where H1's
    second half is visible.

Usage
-----
    # ExpR, the target where the dropout surrogates fail
    python scripts/head_swap_stage_a.py --target ExpR \
        --models dnn_expr,transformer_expr --iterations 10,40 --n-inits 3

    # the relic-density comparison for M4 (loss mass) on the same footing
    python scripts/head_swap_stage_a.py --target DMRD \
        --manifest /ptmp/jwuerzin/analysis/all_runs/manifest_mainbody.csv \
        --models dnn,transformer --iterations 40 --n-inits 3

    # shape check on one snapshot, small pool, one init
    python scripts/head_swap_stage_a.py --smoke --models dnn_expr --seeds 1

Notes
-----
* Pool points are valid by construction (they passed the simulator), so M1 is a
  per-valid rate.  Multiply by p_valid (0.584 for ExpR, 0.445 for DMRD) for a
  per-attempt number comparable with Table 12 of the paper.
* Both arms are ranked with top-k style rules.  The batch-diversity term of
  entropy_batch is deliberately out of scope: it is a property of the selector,
  not of the head, and mixing the two would confound Stage A.
* Epochs/patience default to the production values, so the regression arm's M2
  should land near the published iteration-40 accuracy; that agreement is the
  built-in sanity check on the whole setup.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pmssm.config import TARGET_CONFIG  # noqa: E402
from pmssm.heads import get_head  # noqa: E402
from pmssm.models import PMSSMFeedForward, PMSSMTransformerTabular  # noqa: E402
from pmssm.selection import (  # noqa: E402
    select_entropy_batch_mc,
    select_top_score,
    select_top_uncertain_filtered,
    select_top_uncertain_tol_only,
)
from pmssm.uncertainty import compute_uncertainty_head  # noqa: E402

# Production hyperparameters, mirrored from pmssm.training.train_model_worker so
# the regression arm reproduces the runs it is compared against.
LR = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
GRAD_CLIP = 1.0
ETA_MIN = 1e-6
DNN_KW = dict(n_params=19, d_model=64, num_layers=4, dim_feedforward=256)
TF_KW = dict(d_model=128, nhead=4, num_layers=3, dim_feedforward=512)

# Production acquisition parameters (active_learning.py defaults).
TOLERANCE_SAMPLING = 1.0
PROXIMITY_SAMPLING = 0.1
N_SELECT = 500
MC_SAMPLES = 30
ENTROPY_BLUR = 0.15
ENTROPY_BETA = 50.0
ENTROPY_POOL = 5000

# Bands are the paper's: physical |Y/true_value - 1| <= tau.
TAUS = (0.1, 0.2, 0.5)
EVAL_SPLIT_SEED = 20260820

# Which selectors belong to which head.
SELECTORS = {
    "regression": ("reg_topk", "reg_var_only"),
    "classification": ("cls_entropy", "cls_bald"),
}

POOL_CACHE = {
    "ExpR": (
        "/ptmp/jwuerzin/analysis/expr_runs/x_full_ptmp_jwuerzin_data_260804_ExpR.npy",
        "/ptmp/jwuerzin/analysis/expr_runs/y_full_ptmp_jwuerzin_data_260804_ExpR.npy",
    ),
    "DMRD": (
        "/ptmp/jwuerzin/analysis/all_runs/x_full_ptmp_jwuerzin_data_18387358_DMRD.npy",
        "/ptmp/jwuerzin/analysis/all_runs/y_full_ptmp_jwuerzin_data_18387358_DMRD.npy",
    ),
}


# ----------------------------------------------------------------------------
# pool / snapshot plumbing
# ----------------------------------------------------------------------------
def _row_hash(X: np.ndarray) -> np.ndarray:
    """Collision-resistant 64-bit hash per row, exact on the float32 bit pattern.

    Used to remove every pool row a run has already seen from the evaluation and
    scoring slices.  Only the initial block can overlap (generated points are
    fresh Latin-hypercube draws), but hashing the whole labelled set is cheap and
    makes the guarantee unconditional.
    """
    A = np.ascontiguousarray(X, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rng = np.random.default_rng(0xA11CE)
    w = rng.integers(1, 2**63, size=A.shape[1], dtype=np.uint64) | np.uint64(1)
    return (A * w).sum(axis=1)


def _load_pool(target: str, pool_x: str | None, pool_y: str | None):
    px, py = POOL_CACHE[target]
    px, py = pool_x or px, pool_y or py
    X = np.load(px)
    Y = np.load(py).astype(np.float64).reshape(-1)
    if len(X) != len(Y):
        raise RuntimeError(f"pool cache mismatch: {len(X)} rows of X, {len(Y)} of Y")
    if (Y <= 0).any():
        raise RuntimeError(f"{int((Y <= 0).sum())} non-positive pool targets; the "
                           "log transform is undefined there")
    return X, Y


def _snapshot(state: dict, iteration: int):
    """The (train, val) labelled set as it stood at the end of `iteration`.

    `X`/`X_val` are appended in acquisition order and the run records the row
    counts per iteration, so the snapshot is a prefix.  Both counts are asserted
    against that bookkeeping: a silent misalignment here would invalidate every
    number downstream.
    """
    n_tr, n_va = state["al_n_train"], state["al_n_val"]
    if iteration > len(n_tr):
        raise IndexError(f"run reached iteration {len(n_tr)}, asked for {iteration}")
    k_tr, k_va = int(n_tr[iteration - 1]), int(n_va[iteration - 1])
    X, Y = state["X"], state["Y"].view(-1)
    Xv, Yv = state["X_val"], state["Y_val"].view(-1)
    if k_tr > len(X) or k_va > len(Xv):
        raise RuntimeError(f"prefix {k_tr}/{k_va} exceeds stored {len(X)}/{len(Xv)}")
    if min(k_tr, k_va) == 0:
        raise RuntimeError("empty snapshot")
    return X[:k_tr], Y[:k_tr], Xv[:k_va], Yv[:k_va]


def _manifest_rows(manifest: Path, models: list[str], seeds: list[int]):
    with open(manifest, newline="") as fh:
        return [r for r in csv.DictReader(fh)
                if r["model"] in models and int(r["seed"]) in seeds]


# ----------------------------------------------------------------------------
# training (one factor: the head)
# ----------------------------------------------------------------------------
def _build(arch: str, dropout: float, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if arch == "dnn":
        return PMSSMFeedForward(dropout=dropout, **DNN_KW)
    if arch == "transformer":
        return PMSSMTransformerTabular(dropout=dropout, **TF_KW)
    raise ValueError(f"unknown arch {arch!r}")


def _train(arch, head, X_tr, Y_tr, X_val, Y_val, stats, true_value, *,
           dropout, epochs, patience, device, init_seed, log):
    """Train one arm.  Trunk, optimiser, schedule and early stopping are shared;
    the head supplies the target and the loss."""
    mean_X, std_X = stats[0], stats[1]
    Xtr, Xva = (X_tr - mean_X) / std_X, (X_val - mean_X) / std_X
    ttr = head.make_targets(torch.log(Y_tr.view(-1, 1) / true_value))
    tva = head.make_targets(torch.log(Y_val.view(-1, 1) / true_value))

    model = _build(arch, dropout, init_seed).to(device)
    criterion = head.criterion()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=ETA_MIN)

    loader = DataLoader(TensorDataset(Xtr, ttr), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(init_seed))
    Xva_d, tva_d = Xva.to(device), tva.to(device)

    best, best_state, best_epoch, since, epoch = math.inf, None, -1, 0, -1
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(criterion(model(Xva_d), tva_d))
        if vl < best - 1e-9:
            best, best_epoch, since = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"      {head.name:14s} best val {best:.5f} @epoch {best_epoch} "
        f"({epoch + 1} run, {time.time() - t0:.0f}s)")
    return model, {"val_loss": best, "best_epoch": best_epoch, "epochs_run": epoch + 1}


def _summarise_predictions(model, X, stats, head, *, n_samples, device,
                           want_predictions=False):
    """MC-dropout summary through the production code path."""
    return compute_uncertainty_head(model, X, stats, n_samples, device, None, head,
                                    return_predictions=want_predictions)


# ----------------------------------------------------------------------------
# selection: each arm's own production-equivalent rule
# ----------------------------------------------------------------------------
def _picks(head, summary, X_pool, n_select, predictions=None, device="cpu",
           draw_seed=0):
    """Indices this head would have acquired, per selector, as long tensors.

    Beyond each head's production rule, two references decompose the regression
    path, because its measured precision came out *below* the random rate and the
    cut and the ranking have to be blamed separately:

      prefilter_random  the tolerance cut alone, then a random draw among the
                        survivors: what the mean-based anchor is worth without
                        any uncertainty ranking
      reg_var_nocut     raw variance over the whole pool, no cut: what the
                        ranking is worth without the anchor
      pool_random       a uniform draw, which must reproduce the band prevalence
                        and is therefore a check on the harness itself
    """
    # The random references are re-drawn per (snapshot, init) rather than fixed,
    # so their spread over replicas is a real binomial error bar instead of one
    # lucky or unlucky draw repeated everywhere.
    g = torch.Generator().manual_seed(EVAL_SPLIT_SEED + draw_seed)
    N = len(X_pool)
    out = {"pool_random": torch.randperm(N, generator=g)[:n_select]}

    if head.name == "regression":
        mean, var = summary["pred_mean"], summary["pred_var"]
        out["reg_topk"] = select_top_uncertain_filtered(
            X_pool, mean, var, n_select, threshold=0.0,
            tolerance_sampling=TOLERANCE_SAMPLING,
            proximity_sampling=PROXIMITY_SAMPLING)
        out["reg_var_only"] = select_top_uncertain_tol_only(
            X_pool, mean, var, n_select, threshold=0.0,
            tolerance_sampling=TOLERANCE_SAMPLING)
        out["reg_var_nocut"] = select_top_score(X_pool, var, n_select)
        if predictions is not None:
            # The paper's headline ExpR cell for the dropout surrogates is
            # entropy_batch, not top_k, and its per-valid yield (0.033 for the
            # DNN) sits at the pool prevalence. Without this arm a top_k-only
            # comparison cannot speak about the published configuration.
            out["reg_entropy_batch"] = select_entropy_batch_mc(
                X_pool, predictions, mean, var, n_select,
                blur=ENTROPY_BLUR, beta=ENTROPY_BETA, n_pool=ENTROPY_POOL,
                threshold=0.0, tolerance_sampling=TOLERANCE_SAMPLING,
                proximity_sampling=PROXIMITY_SAMPLING, device=device)
        surv = torch.where(mean.squeeze().abs() < TOLERANCE_SAMPLING)[0]
        if len(surv) == 0:
            surv = torch.arange(N)
        out["prefilter_random"] = surv[torch.randperm(
            len(surv),
            generator=torch.Generator().manual_seed(EVAL_SPLIT_SEED + 7919 + draw_seed)
        )[:n_select]]
    else:
        # A boundary-anchored score needs no mean-based pre-filter; that is the
        # point of it.  The tolerance cut stays available in select_top_score for
        # a matched-anchor variant.
        for name, which in (("cls_entropy", "entropy"), ("cls_bald", "bald")):
            out[name] = select_top_score(
                X_pool, head.acquisition_score(summary, which), n_select)
    return {k: torch.as_tensor(np.asarray(v), dtype=torch.long) for k, v in out.items()}


def _pick_profile(idx, t_pool, Xn_pool, Xn_train, chunk=256):
    """Where a selector's picks actually sit, in target space and input space.

    The two questions the precision number cannot answer: are the picks near the
    contour (|t| quantiles), and are they in regions the labelled set has not
    covered (distance to the nearest labelled point, in normalised input space).
    BALD outscoring the total predictive entropy predicts the second column
    should separate them, since only BALD is a disagreement signal.
    """
    a = t_pool[idx].abs()
    q = torch.quantile(a, torch.tensor([0.25, 0.5, 0.75], dtype=a.dtype))
    d_min = []
    P = Xn_pool[idx]
    for i in range(0, len(P), chunk):
        d_min.append(torch.cdist(P[i:i + chunk], Xn_train).min(dim=1).values)
    d_min = torch.cat(d_min) if d_min else torch.zeros(0)
    return {
        "abs_t_q25": float(q[0]), "abs_t_median": float(q[1]),
        "abs_t_q75": float(q[2]), "abs_t_mean": float(a.mean()),
        "nn_dist_mean": float(d_min.mean()) if len(d_min) else None,
        "nn_dist_median": float(d_min.median()) if len(d_min) else None,
    }


# ----------------------------------------------------------------------------
# measurements
# ----------------------------------------------------------------------------
def _band_masks(Y: torch.Tensor, true_value: float):
    r = Y.view(-1) / true_value
    return {f"tau{int(tau * 100)}": (r - 1.0).abs() <= tau for tau in TAUS}


def _precision(idx, bands):
    n = max(len(idx), 1)
    out = {k: float(m[idx].sum()) / n for k, m in bands.items()}
    out["n_picked"] = int(len(idx))
    return out


def _shells(t_true):
    a = t_true.abs()
    return {
        "shell_lt_ln1.1": a < math.log(1.1),
        "shell_lt_ln1.5": a < math.log(1.5),
        "shell_lt_ln3": a < math.log(3.0),
        "outside_ln3": a >= math.log(3.0),
    }


def _verdict_accuracy(raw_mean, t_true, shells):
    """Threshold the raw output at zero, for either head.

    This is deliberately the same rule the per-iteration accuracy diagnostics
    apply (`_classification_accuracy` in the multiseed plotter), because a logit
    is positive exactly when p > 0.5: it is what makes those diagnostics
    head-agnostic.
    """
    pred_pos, truth = raw_mean > 0, t_true > 0
    out = {"all": float((pred_pos == truth).float().mean())}
    for name, m in shells.items():
        out[name] = float((pred_pos[m] == truth[m]).float().mean()) if int(m.sum()) else None
    return out


def _loss_mass(mean_reg, t_true, bands):
    """M4: which points own the squared-error mass."""
    sq = (mean_reg - t_true) ** 2
    tot = float(sq.sum())
    out = {"mse": tot / len(sq)}
    for k, m in bands.items():
        out[f"share_{k}"] = float(sq[m].sum()) / tot if tot > 0 else None
        out[f"frac_points_{k}"] = float(m.float().mean())
    return out


def _uncertainty_profile(u, t_true, edges=(0.1, 0.3, 1.0, 3.0, 10.0)):
    """M5: mean uncertainty in bins of distance to the contour."""
    a, prof, lo = t_true.abs(), {}, 0.0
    for hi in edges:
        m = (a >= lo) & (a < hi)
        prof[f"|t|_{lo:g}-{hi:g}"] = float(u[m].mean()) if int(m.sum()) else None
        lo = hi
    m = a >= lo
    prof[f"|t|_>{lo:g}"] = float(u[m].mean()) if int(m.sum()) else None
    return prof


def _profile_quantity(head, summary):
    """The uncertainty a top-k rule would rank by, in the head's own units."""
    if head.name == "regression":
        return summary["pred_var"].view(-1).sqrt()
    return summary["entropy"].view(-1)


# ----------------------------------------------------------------------------
# one snapshot
# ----------------------------------------------------------------------------
def _run_snapshot(arch, X_tr, Y_tr, X_val, Y_val, *, true_value, pool, bands_pool,
                  eval_slice, heads, n_inits, dropout, epochs, patience, device,
                  n_select, mc_samples, one_step, log):
    X_pool, Y_pool = pool
    X_ev, Y_ev = eval_slice
    t_ev = torch.log(Y_ev.view(-1) / true_value)
    shells, bands_ev = _shells(t_ev), _band_masks(Y_ev, true_value)
    mean_X, std_X = X_tr.mean(dim=0), X_tr.std(dim=0) + 1e-8
    # compute_uncertainty_head takes the production 4-tuple; only the X stats are
    # used, since with y_transform='log' the target is never standardised.
    stats = (mean_X, std_X, torch.zeros(1), torch.ones(1))
    # For M6: pick positions in target space and in normalised input space.
    t_pool = torch.log(Y_pool.view(-1) / true_value)
    Xn_pool, Xn_train = (X_pool - mean_X) / std_X, (X_tr - mean_X) / std_X

    res = {"n_train": len(X_tr), "n_val": len(X_val), "arms": {}}
    for head_name in heads:
        head = get_head(head_name, threshold=0.0)
        per_init = []
        for k in range(n_inits):
            model, fit = _train(arch, head, X_tr, Y_tr, X_val, Y_val, stats, true_value,
                                dropout=dropout, epochs=epochs, patience=patience,
                                device=device, init_seed=1000 + k, log=log)
            s_pool, preds = _summarise_predictions(
                model, X_pool, stats, head, n_samples=mc_samples, device=device,
                want_predictions=True)
            picks = _picks(head, s_pool, X_pool, n_select,
                           predictions=preds, device=device,
                           draw_seed=hash((head.name, k, len(X_tr))) % 100_000)
            del preds
            s_ev = _summarise_predictions(model, X_ev, stats, head,
                                          n_samples=mc_samples, device=device)
            raw_mean = s_ev["pred_mean"].view(-1)

            entry = {
                "fit": fit,
                "M1_precision_at_500": {n: _precision(i, bands_pool)
                                        for n, i in picks.items()},
                "M2_verdict_accuracy": _verdict_accuracy(raw_mean, t_ev, shells),
                "M5_uncertainty_profile": _uncertainty_profile(
                    _profile_quantity(head, s_ev), t_ev),
                "M6_pick_profile": {n: _pick_profile(i, t_pool, Xn_pool, Xn_train)
                                    for n, i in picks.items()},
            }
            if head.value_metrics:
                entry["M4_loss_mass"] = _loss_mass(raw_mean, t_ev, bands_ev)

            if one_step and k == 0:
                entry["M3_one_step"] = {}
                for name, idx in picks.items():
                    X2 = torch.cat([X_tr, X_pool[idx]])
                    Y2 = torch.cat([Y_tr, Y_pool[idx]])
                    m2X, s2X = X2.mean(dim=0), X2.std(dim=0) + 1e-8
                    st2 = (m2X, s2X, torch.zeros(1), torch.ones(1))
                    m2, _ = _train(arch, head, X2, Y2, X_val, Y_val, st2, true_value,
                                   dropout=dropout, epochs=epochs, patience=patience,
                                   device=device, init_seed=1000, log=log)
                    s2 = _summarise_predictions(m2, X_ev, st2, head,
                                               n_samples=mc_samples, device=device)
                    acc2 = _verdict_accuracy(s2["pred_mean"].view(-1), t_ev, shells)
                    entry["M3_one_step"][name] = {
                        "accuracy_after": acc2,
                        "delta_all": acc2["all"] - entry["M2_verdict_accuracy"]["all"],
                    }
                    del m2
            per_init.append(entry)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        res["arms"][head_name] = per_init
    return res


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
@click.command()
@click.option("--target", default="ExpR", type=click.Choice(sorted(TARGET_CONFIG)))
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/joint/manifest_expr.csv",
              help="Best-cell manifest; rows give the runs whose acquired sets are reused.")
@click.option("--models", default="dnn_expr,transformer_expr",
              help="Comma list of manifest `model` values.")
@click.option("--seeds", default="1,2,3,4,5")
@click.option("--iterations", default="10,40", help="Comma list of snapshot iterations.")
@click.option("--heads", default="regression,classification",
              help="Comma list of head names from pmssm.heads.")
@click.option("--n-inits", default=3, show_default=True,
              help="Independent initialisations per arm (aggregated as mean +- SEM).")
@click.option("--eval-size", default=100_000, show_default=True)
@click.option("--score-pool-size", default=500_000, show_default=True,
              help="Pool slice each head scores; production sees 1e6 fresh candidates.")
@click.option("--n-select", default=N_SELECT, show_default=True)
@click.option("--mc-samples", default=MC_SAMPLES, show_default=True)
@click.option("--dropout", default=0.1, show_default=True)
@click.option("--epochs", default=10_000, show_default=True)
@click.option("--patience", default=200, show_default=True)
@click.option("--one-step/--no-one-step", default=True, show_default=True,
              help="M3: retrain on L + own picks (extra trainings per snapshot).")
@click.option("--pool-x", default=None)
@click.option("--pool-y", default=None)
@click.option("--pool-mask", default=None,
              help="Optional boolean .npy over pool rows (e.g. the neutralino-LSP "
                   "mask for DMRD, whose random-scan reference counts only those).")
@click.option("--output", default="/ptmp/jwuerzin/analysis/head_swap/stage_a.json",
              show_default=True)
@click.option("--device", default=None, help="cuda:0/cpu (default: auto).")
@click.option("--smoke", is_flag=True, help="One snapshot, one init, tiny pool, short fit.")
def main(target, manifest, models, seeds, iterations, heads, n_inits, eval_size,
         score_pool_size, n_select, mc_samples, dropout, epochs, patience,
         one_step, pool_x, pool_y, pool_mask, output, device, smoke):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    models = [m.strip() for m in models.split(",") if m.strip()]
    seeds = [int(s) for s in seeds.split(",") if s.strip()]
    iters = [int(i) for i in iterations.split(",") if i.strip()]
    heads = [h.strip() for h in heads.split(",") if h.strip()]
    if smoke:
        seeds, iters, n_inits = seeds[:1], iters[:1], 1
        eval_size, score_pool_size = 20_000, 50_000
        epochs, patience, mc_samples, one_step = 60, 20, 5, False

    def log(msg):
        click.echo(msg)
        sys.stdout.flush()

    true_value = float(TARGET_CONFIG[target]["true_value"])
    log(f"[cfg] target={target} true_value={true_value} device={device} heads={heads} "
        f"inits={n_inits} iters={iters} models={models} seeds={seeds}")

    Xp_np, Yp_np = _load_pool(target, pool_x, pool_y)
    log(f"[pool] {len(Xp_np)} valid points from cache")
    if pool_mask:
        keep = np.load(pool_mask).astype(bool).reshape(-1)
        if len(keep) != len(Xp_np):
            raise click.UsageError(f"mask has {len(keep)} rows, pool has {len(Xp_np)}")
        Xp_np, Yp_np = Xp_np[keep], Yp_np[keep]
        log(f"[pool] mask keeps {len(Xp_np)} ({keep.mean():.4f} of rows)")

    rows = _manifest_rows(Path(manifest), models, seeds)
    if not rows:
        raise click.UsageError(f"no manifest rows for models={models} seeds={seeds}")
    log(f"[manifest] {len(rows)} runs")

    # States first: the union of every labelled row is removed from the pool
    # before any split is drawn, so no arm is evaluated or scored on a point some
    # run already trained on.
    states, seen = {}, []
    for row in rows:
        p = Path(row["expected_run_dir"]) / "state.pt"
        if not p.exists():
            log(f"[skip] {row['model']} seed {row['seed']}: no state.pt")
            continue
        st = torch.load(p, weights_only=False, map_location="cpu")
        cell = (row["model"], row.get("strategy", "?"), row.get("warm_start", "?"),
                int(row["seed"]))
        if cell in states:
            log(f"[warn] duplicate manifest row for {cell}; keeping the first")
            continue
        states[cell] = st
        seen.extend([st["X"].numpy(), st["X_val"].numpy()])
    if not states:
        raise click.UsageError("no usable runs")

    free = np.flatnonzero(~np.isin(_row_hash(Xp_np),
                                   _row_hash(np.concatenate(seen, axis=0))))
    log(f"[pool] {len(Xp_np) - len(free)} rows seen by some run, {len(free)} free")

    perm = np.random.default_rng(EVAL_SPLIT_SEED).permutation(free)
    n_ev = min(eval_size, len(perm) // 2)
    ev_idx, sc_idx = perm[:n_ev], perm[n_ev:n_ev + score_pool_size]
    log(f"[split] eval {len(ev_idx)}, scoring pool {len(sc_idx)} (disjoint, seed "
        f"{EVAL_SPLIT_SEED})")

    def to_t(a):
        return torch.tensor(a, dtype=torch.float32)

    X_ev, Y_ev = to_t(Xp_np[ev_idx]), to_t(Yp_np[ev_idx])
    X_sc, Y_sc = to_t(Xp_np[sc_idx]), to_t(Yp_np[sc_idx])
    bands_pool = _band_masks(Y_sc, true_value)
    prevalence = {k: float(m.float().mean()) for k, m in bands_pool.items()}
    log("[reference] random-pick precision (pool band prevalence): "
        + ", ".join(f"{k}={v:.4f}" for k, v in prevalence.items()))

    results = {
        "config": {
            "target": target, "true_value": true_value, "manifest": manifest,
            "models": models, "seeds": seeds, "iterations": iters, "heads": heads,
            "n_inits": n_inits, "n_select": n_select, "mc_samples": mc_samples,
            "dropout": dropout, "epochs": epochs, "patience": patience,
            "eval_size": len(ev_idx), "score_pool_size": len(sc_idx),
            "tolerance_sampling": TOLERANCE_SAMPLING,
            "proximity_sampling": PROXIMITY_SAMPLING,
            "eval_split_seed": EVAL_SPLIT_SEED, "pool_mask": pool_mask,
            "smoke": smoke,
        },
        "random_pick_precision": prevalence,
        "snapshots": {},
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for (model_name, strategy, warm, seed), st in sorted(states.items()):
        arch = "transformer" if "transformer" in model_name else "dnn"
        for it in iters:
            key = f"{model_name}|{strategy}|{warm}|seed{seed}|iter{it}"
            try:
                X_tr, Y_tr, X_va, Y_va = _snapshot(st, it)
            except (IndexError, RuntimeError) as exc:
                log(f"[skip] {key}: {exc}")
                continue
            log(f"[run] {key}: train {len(X_tr)} val {len(X_va)} arch {arch}")
            results["snapshots"][key] = _run_snapshot(
                arch, X_tr, Y_tr, X_va, Y_va, true_value=true_value,
                pool=(X_sc, Y_sc), bands_pool=bands_pool, eval_slice=(X_ev, Y_ev),
                heads=heads, n_inits=n_inits, dropout=dropout, epochs=epochs,
                patience=patience, device=device, n_select=n_select,
                mc_samples=mc_samples, one_step=one_step, log=log)
            out.write_text(json.dumps(results, indent=2))  # checkpoint per snapshot
            log(f"[write] {out}")

    _summarise(results, log)
    out.write_text(json.dumps(results, indent=2))
    log(f"[done] {out}")


def _summarise(results, log):
    """Print the comparisons Stage A exists to make."""
    def agg(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        m = sum(vals) / len(vals)
        s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1) / len(vals)) ** 0.5 \
            if len(vals) > 1 else 0.0
        return m, s

    ref = results["random_pick_precision"].get("tau10")
    log("\n=== M1 precision@500 for the +-10% band (mean +- SEM over inits x seeds) ===")
    log(f"random pick reference: {ref:.4f}")
    by_sel = {}
    for key, snap in results["snapshots"].items():
        it = key.split("|")[-1]
        for inits in snap["arms"].values():
            for e in inits:
                for sel, prec in e["M1_precision_at_500"].items():
                    by_sel.setdefault((it, sel), []).append(prec["tau10"])
    for (it, sel), vals in sorted(by_sel.items()):
        m, s = agg(vals)
        log(f"  {it:8s} {sel:14s} {m:.4f} +- {s:.4f}  n={len(vals)}  "
            f"({m / ref:.1f}x random)")

    log("\n=== M2 verdict accuracy: all points, and the near-contour shell ===")
    by_head = {}
    for key, snap in results["snapshots"].items():
        it = key.split("|")[-1]
        for head, inits in snap["arms"].items():
            for e in inits:
                a = e["M2_verdict_accuracy"]
                by_head.setdefault((it, head), []).append((a["all"], a["shell_lt_ln1.1"]))
    for (it, head), vals in sorted(by_head.items()):
        ma, sa = agg([v[0] for v in vals])
        mb, sb = agg([v[1] for v in vals])
        log(f"  {it:8s} {head:14s} all {ma:.4f} +- {sa:.4f} | "
            f"|t|<ln1.1 {mb:.4f} +- {sb:.4f}  n={len(vals)}")

    log("\n=== M3 one-step gain in verdict accuracy from each arm's own 500 picks ===")
    by_sel = {}
    for key, snap in results["snapshots"].items():
        it = key.split("|")[-1]
        for inits in snap["arms"].values():
            for e in inits:
                for sel, d in e.get("M3_one_step", {}).items():
                    by_sel.setdefault((it, sel), []).append(d["delta_all"])
    for (it, sel), vals in sorted(by_sel.items()):
        m, s = agg(vals)
        log(f"  {it:8s} {sel:14s} {m:+.5f} +- {s:.5f}  n={len(vals)}")

    log("\n=== M6 where the picks sit: |t| median, and distance to labelled set ===")
    by_sel = {}
    for key, snap in results["snapshots"].items():
        it = key.split("|")[-1]
        for inits in snap["arms"].values():
            for e in inits:
                for sel, pr in e.get("M6_pick_profile", {}).items():
                    by_sel.setdefault((it, sel), []).append(
                        (pr["abs_t_median"], pr["nn_dist_mean"]))
    for (it, sel), vals in sorted(by_sel.items()):
        mt, st_ = agg([v[0] for v in vals])
        md, sd = agg([v[1] for v in vals])
        log(f"  {it:8s} {sel:16s} |t| median {mt:.3f} +- {st_:.3f} | "
            f"nn-dist {md:.3f} +- {sd:.3f}  n={len(vals)}")

    log("\n=== M4 share of the regression loss owned by in-band points ===")
    for key, snap in results["snapshots"].items():
        for e in snap["arms"].get("regression", [])[:1]:
            lm = e.get("M4_loss_mass")
            if lm:
                log(f"  {key}: mse {lm['mse']:.3f}; the tau10 band holds "
                    f"{lm['frac_points_tau10']:.4f} of points and "
                    f"{lm['share_tau10']:.5f} of the loss")


if __name__ == "__main__":
    main()
