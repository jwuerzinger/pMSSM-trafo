"""Confusion matrices for the exclusion verdict, AL model versus random baseline.

The support and discovery analyses ask where each run put its points. This asks
what the resulting SURROGATE would tell a physicist: thresholding the predicted
`r_exp` at 1 turns each model into a classifier of "excluded" against "viable",
and the two error directions cost very different things. A false exclusion
(predicted excluded, truly viable) discards live parameter space and, if a limit
contour were drawn from the surrogate, would be a wrong physics claim. A false
viability is caught by the next simulator call.

Four matrices, because two of them alone cannot be compared:

  AL model     on the static-random eval set   does AL overclaim exclusion over
                                               the prior-weighted space?
  baseline     on the static-random eval set   the fair mirror of the above
  baseline     on the AL-acquired points       would a random-trained surrogate
                                               discard viable models AL found?
  AL model     on the AL-acquired points       control for the above, since those
                                               points are intrinsically hard

The static-random set is the one the run itself carved and held out: the pool is
shuffled with the run's seed, rows [0, n_samples) are the shared initialisation
that AL trained on, and `static_eval_size` rows are drawn from the remainder with
a fixed generator (seed 123). Neither model ever saw it, so matrices one and two
are like-for-like. The AL-acquired points are every simulated point of the run,
train and validation both.

Both models come from the run's own checkpoints at the requested iteration, so
nothing is retrained: `al_model_checkpoint.pt` was fitted on the acquired points
and `baseline_model_checkpoint.pt` on an equal number of random pool rows.

False exclusions are also localised on the (IN_M_1, IN_M_2, IN_mu) quantile grid
of the support and island analyses, and reported for the rare sub-regions of
`find_islands.py`, which is where a surrogate that never sampled a corner would
be expected to fail.

Usage:

    P=/ptmp/jwuerzin/pixi-envs/pytorch-conda-forge-2863954108128992291/envs/rocm/bin/python
    $P scripts/exclusion_confusion.py \\
        --run-dir /ptmp/jwuerzin/output/active_learning_deep_gp_expr_..._seed1_... \\
        --output-dir /ptmp/jwuerzin/analysis/joint/expr_confusion
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts"),
           str(_REPO_ROOT / "al_pmssmwithgp" / "model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _confusion(y_true, y_pred, thr):
    """Counts and rates for the excluded/viable verdict at threshold ``thr``.

    "Excluded" is r > thr. Rates are per TRUE class, so ``false_exclusion_rate``
    is P(predicted excluded | truly viable), which is the quantity with physics
    consequences, and is unaffected by the class balance of the evaluation set.
    """
    t_ex, p_ex = y_true > thr, y_pred > thr
    tp = int((t_ex & p_ex).sum())        # excluded, called excluded
    fn = int((t_ex & ~p_ex).sum())       # excluded, called viable
    fp = int((~t_ex & p_ex).sum())       # viable, called EXCLUDED  <- the costly one
    tn = int((~t_ex & ~p_ex).sum())      # viable, called viable
    n_ex, n_vi = tp + fn, fp + tn
    return {"n": int(len(y_true)), "n_truly_excluded": n_ex, "n_truly_viable": n_vi,
            "prevalence_excluded": n_ex / max(1, len(y_true)),
            "tp_excluded_called_excluded": tp, "fn_excluded_called_viable": fn,
            "fp_viable_called_excluded": fp, "tn_viable_called_viable": tn,
            "false_exclusion_rate": fp / n_vi if n_vi else float("nan"),
            "false_viability_rate": fn / n_ex if n_ex else float("nan"),
            "accuracy": (tp + tn) / max(1, len(y_true))}


@click.command()
@click.option("--run-dir", required=True, help="AL run directory.")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/joint/expr_confusion",
              show_default=True)
@click.option("--iteration", default=0, show_default=True,
              help="Checkpoint iteration; 0 = the highest present.")
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/260804", show_default=True)
@click.option("--target", default="ExpR", show_default=True)
@click.option("--model-type", default="deep_gp", show_default=True)
@click.option("--seed", default=1, show_default=True,
              help="The run's seed: sets the pool shuffle, so it must match.")
@click.option("--n-samples-initial", default=2000, show_default=True,
              help="The run's --n-samples: the reserved initial block.")
@click.option("--static-eval-size", default=100_000, show_default=True)
@click.option("--eval-subsample", default=0, show_default=True,
              help="Score only this many static-random rows (0 = all). CPU "
                   "prediction over 100k deep-GP points is the slow step.")
@click.option("--kernel", default="RBF", show_default=True)
@click.option("--lengthscale", default=1.0, show_default=True)
@click.option("--noise", default=0.01, show_default=True)
@click.option("--jitter", default=0.001, show_default=True)
@click.option("--num-hidden-dims", default=10, show_default=True)
@click.option("--num-middle-dims", default=0, show_default=True)
@click.option("--num-inducing-max", default=256, show_default=True)
@click.option("--gp-num-samples", default=8, show_default=True)
@click.option("--n-train-init", default=1600, show_default=True,
              help="The run's initial TRAIN count (logged as 'Initial split'). "
                   "Needed to rebuild the baseline's own train/val split.")
@click.option("--n-val-init", default=400, show_default=True,
              help="The run's initial VAL count, shared by both models.")
@click.option("--islands-json", default="/ptmp/jwuerzin/analysis/joint/expr_islands/islands_ExpR.json",
              show_default=True, help="Sub-regions from find_islands.py, for the "
                                      "localised breakdown. Empty to skip.")
def main(run_dir, output_dir, iteration, baseline_data_dir, target, model_type,
         seed, n_samples_initial, static_eval_size, eval_subsample, kernel,
         lengthscale, noise, jitter, num_hidden_dims, num_middle_dims,
         num_inducing_max, gp_num_samples, n_train_init, n_val_init,
         islands_json):
    from active_learning_gp import cross_evaluate_gp
    from pmssm.config import PARAM_ORDER, TARGET_CONFIG
    from pmssm.data import build_norm_tensors, load_pmssm_data, transform_y
    from pmssm.training import create_gp_model

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = Path(run_dir)
    thr = float(TARGET_CONFIG[target]["true_value"])
    click.echo(f"[cm] run {run.name}")
    click.echo(f"[cm] target {target}, verdict: excluded iff r > {thr:g}")

    iters = sorted(int(p.name.split("_")[1]) for p in run.glob("iteration_*")
                   if (p / "al_model_checkpoint.pt").exists()
                   and (p / "baseline_model_checkpoint.pt").exists())
    if not iters:
        raise click.ClickException(f"no iteration with both checkpoints in {run}")
    it = iteration or iters[-1]
    it_dir = run / f"iteration_{it:03d}"
    click.echo(f"[cm] iteration {it} of {iters[0]}..{iters[-1]}")

    # ── the run's own data ───────────────────────────────────────────────────
    s = torch.load(run / "state.pt", map_location="cpu", weights_only=False)
    X_al = torch.cat([s["X"], s["X_val"]]).float()
    Y_al = torch.cat([s["Y"], s["Y_val"]]).float().view(-1, 1)
    base_idx = s.get("baseline_add_indices")
    click.echo(f"[cm] AL points {len(Y_al):,} (train {len(s['X']):,} + val "
               f"{len(s['X_val']):,}); baseline drew {0 if base_idx is None else len(base_idx):,} "
               f"pool rows")

    # ── reproduce the pool order, the reserved block and the static eval set ─
    Xf, Yf = load_pmssm_data(n_datasets=-1, data_dir=baseline_data_dir,
                             target=target, plot_dir=str(out))
    Xf = Xf.float()
    Yf = Yf.float().view(-1, 1)
    perm = torch.randperm(len(Xf), generator=torch.Generator().manual_seed(seed))
    Xf, Yf = Xf[perm], Yf[perm]
    avail = len(Xf) - n_samples_initial
    g_static = torch.Generator().manual_seed(123)
    st_idx = torch.randperm(avail, generator=g_static)[:min(static_eval_size, avail)] \
        + n_samples_initial
    X_st, Y_st = Xf[st_idx], Yf[st_idx]
    if eval_subsample and eval_subsample < len(Y_st):
        keep = torch.randperm(len(Y_st),
                              generator=torch.Generator().manual_seed(7))[:eval_subsample]
        X_st, Y_st = X_st[keep], Y_st[keep]
    # The baseline trains on pool rows; if any leaked into the static set the
    # mirror matrix would be scored partly on its own training data.
    if base_idx is not None:
        overlap = len(np.intersect1d(np.asarray(st_idx), np.asarray(base_idx)))
        click.echo(f"[cm] static set {len(Y_st):,} rows, overlap with the "
                   f"baseline's training rows: {overlap}")
    click.echo(f"[cm] static set truly excluded: "
               f"{float((Y_st.view(-1) > thr).float().mean()):.3%}; "
               f"AL points truly excluded: "
               f"{float((Y_al.view(-1) > thr).float().mean()):.3%}")

    # ── rebuild both models from their checkpoints ───────────────────────────
    data_min, data_max = build_norm_tensors()

    def _load(tag, x_fit, y_fit):
        """Instantiate the architecture and load the stored weights.

        The checkpoint holds only state dicts, so the model must be constructed
        first; the fitting data only fixes tensor SHAPES here (inducing points
        are capped at num_inducing_max), since every learnable value is then
        overwritten by the checkpoint. 'vanilla' inducing initialisation is used
        because k-means over tens of thousands of points would cost minutes and
        is discarded immediately.
        """
        from pmssm.data import normalize_x
        xn = normalize_x(x_fit, data_min, data_max)
        yt = transform_y(y_fit, target=target).view(-1)
        n_val = max(2, int(0.2 * len(xn)))
        m = create_gp_model(model_type, xn[:-n_val], yt[:-n_val], xn[-n_val:],
                            yt[-n_val:], len(PARAM_ORDER), kernel=kernel,
                            lengthscale=lengthscale, noise=noise, use_ard=True,
                            use_dkl=False, num_hidden_dims=num_hidden_dims,
                            num_middle_dims=num_middle_dims,
                            num_inducing_max=num_inducing_max,
                            num_samples=gp_num_samples, seed=seed, device="cpu",
                            target=target, inducing_strategy="vanilla")
        ck = torch.load(it_dir / f"{tag}_model_checkpoint.pt", map_location="cpu",
                        weights_only=False)
        m.load_state_dict(ck["model_state_dict"])
        if "likelihood_state_dict" in ck and hasattr(m, "likelihood"):
            m.likelihood.load_state_dict(ck["likelihood_state_dict"])
        m.eval()
        click.echo(f"[cm] loaded {tag} model from {it_dir.name}")
        return m

    X_base = Xf[base_idx] if base_idx is not None else X_al
    Y_base = Yf[base_idx] if base_idx is not None else Y_al
    models = {"al": _load("al", X_al, Y_al), "baseline": _load("baseline", X_base, Y_base)}

    # ── the validation populations ───────────────────────────────────────────
    # Each model early-stopped on its OWN validation set and neither fitted on
    # the other's, so a paired comparison over these is close to fair: the
    # leakage is weak (no gradient steps) and symmetric. The first --n-val-init
    # rows are the shared initialisation split, val for both models, so the
    # pooled set counts them once.
    #
    # The arrays can run past the last checkpoint when a run was killed
    # mid-iteration (here 37,713 train against the 37,526 the checkpoint was
    # fitted on), so the AL validation set is trimmed to the rows that existed
    # when this checkpoint was written.
    n_add_val = int(s.get("prev_n_add_val") or 0)
    n_add_train = int(s.get("prev_n_add_train") or 0)
    X_alv = s["X_val"][:n_val_init + n_add_val].float()
    Y_alv = s["Y_val"][:n_val_init + n_add_val].float().view(-1, 1)
    if base_idx is not None and len(base_idx) > n_add_train:
        b_val = np.asarray(base_idx)[n_add_train:]
        X_bv = torch.cat([s["X_val"][:n_val_init].float(), Xf[b_val]])
        Y_bv = torch.cat([s["Y_val"][:n_val_init].float().view(-1, 1), Yf[b_val]])
    else:
        X_bv, Y_bv = X_alv[:0], Y_alv[:0]
    click.echo(f"[cm] AL val {len(Y_alv):,} (shared init {n_val_init} + "
               f"{n_add_val:,} acquired); baseline val {len(Y_bv):,} "
               f"(shared init {n_val_init} + {len(Y_bv) - n_val_init:,} random)")

    # ── predict and score ────────────────────────────────────────────────────
    sets = {"static_random": (X_st, Y_st), "al_points": (X_al, Y_al),
            "al_val": (X_alv, Y_alv), "baseline_val": (X_bv, Y_bv)}
    res, preds = {}, {}
    for mname, model in models.items():
        for sname, (Xe, Ye) in sets.items():
            _mse, r2, y_true, y_pred = cross_evaluate_gp(
                model, Xe, Ye, data_min, data_max, model_type, jitter=jitter,
                num_samples=gp_num_samples, target=target,
                return_predictions=True)
            cm = _confusion(y_true.numpy(), y_pred.numpy(), thr)
            cm["r2_physical"] = float(r2)
            res[f"{mname}_on_{sname}"] = cm
            preds[(mname, sname)] = (y_true.numpy(), y_pred.numpy())
            click.echo(
                f"[cm] {mname:>8} on {sname:<14} n={cm['n']:>7,} "
                f"acc={cm['accuracy']:.3f} R2={r2:+.3f} | "
                f"false exclusion {cm['false_exclusion_rate']:.3%} "
                f"({cm['fp_viable_called_excluded']:,}/{cm['n_truly_viable']:,}) | "
                f"false viability {cm['false_viability_rate']:.3%} "
                f"({cm['fn_excluded_called_viable']:,}/{cm['n_truly_excluded']:,})")

    # Cache every prediction: the deep-GP pass over 100k rows is the expensive
    # step, and every follow-up question (which cells fail, other thresholds,
    # paired tests) is arithmetic on these arrays.
    np.savez_compressed(
        out / "predictions.npz",
        **{f"{m}__{sn}__{w}": arr
           for (m, sn), (yt, yp) in preds.items()
           for w, arr in (("true", yt), ("pred", yp))})
    click.echo(f"[cm] cached predictions to {out / 'predictions.npz'}")

    # ── paired comparison, restricted to truly VIABLE models ────────────────
    # The two models are scored on the same rows, so the discordant cells are
    # what carries the comparison (this is McNemar's setup). Restricting to
    # r <= threshold isolates the costly direction: every disagreement here is
    # one model discarding a viable model that the other keeps.
    def _paired(sname, extra=None):
        yt_a, yp_a = preds[("al", sname)]
        yt_b, yp_b = preds[("baseline", sname)]
        if extra is not None:
            yt_a, yp_a, yt_b, yp_b = (np.concatenate([a, b]) for a, b in
                                      zip((yt_a, yp_a, yt_b, yp_b), extra))
        assert np.allclose(yt_a, yt_b), f"{sname}: truth mismatch between models"
        vi = yt_a <= thr
        a_ex, b_ex = yp_a[vi] > thr, yp_b[vi] > thr
        n = int(vi.sum())
        cat = {"n_viable": n,
               "both_correct": int((~a_ex & ~b_ex).sum()),
               "both_exclude_difficult": int((a_ex & b_ex).sum()),
               "al_overclaims": int((a_ex & ~b_ex).sum()),
               "al_keeps_random_discards": int((~a_ex & b_ex).sum())}
        for k in ("both_correct", "both_exclude_difficult", "al_overclaims",
                  "al_keeps_random_discards"):
            cat[f"{k}_frac"] = cat[k] / n if n else float("nan")
        d1, d2 = cat["al_keeps_random_discards"], cat["al_overclaims"]
        # McNemar with continuity correction on the discordant pairs.
        cat["mcnemar_chi2"] = ((abs(d1 - d2) - 1) ** 2 / (d1 + d2)) if (d1 + d2) else float("nan")
        cat["discordant"] = d1 + d2
        return cat

    paired = {}
    for sname in ("static_random", "al_val", "baseline_val", "al_points"):
        if len(sets[sname][1]):
            paired[sname] = _paired(sname)
    # Pooled validation: AL's val plus the baseline's random-drawn val rows,
    # the shared initialisation block counted once (it is already inside al_val).
    if len(sets["baseline_val"][1]) > n_val_init:
        yt_b, yp_b = preds[("baseline", "baseline_val")]
        ya_b, yap_b = preds[("al", "baseline_val")]
        paired["pooled_val"] = _paired(
            "al_val", extra=(ya_b[n_val_init:], yap_b[n_val_init:],
                             yt_b[n_val_init:], yp_b[n_val_init:]))
    for sname, c in paired.items():
        click.echo(f"[cm] paired {sname:<14} viable n={c['n_viable']:>6,} | "
                   f"both ok {c['both_correct_frac']:.3%} | both exclude "
                   f"{c['both_exclude_difficult_frac']:.3%} | AL overclaims "
                   f"{c['al_overclaims_frac']:.3%} | AL keeps/random discards "
                   f"{c['al_keeps_random_discards_frac']:.3%} | "
                   f"McNemar chi2={c['mcnemar_chi2']:.1f}")

    # ── where do the false exclusions sit? ───────────────────────────────────
    # Same three axes and grid as the support and island analyses, built from the
    # static set's own viable models so the cells are defined by the population
    # being misclassified rather than by either model.
    ax = [PARAM_ORDER.index(a) for a in ("IN_M_1", "IN_M_2", "IN_mu")]
    loc = {}
    nb = 8
    Xs3 = X_st[:, ax].numpy()
    yt_st = preds[("al", "static_random")][0]
    viable = yt_st <= thr
    edges = [np.quantile(Xs3[viable][:, j], np.linspace(0, 1, nb + 1)) for j in range(3)]
    for e in edges:
        e[0], e[-1] = -np.inf, np.inf
    b = np.stack([np.clip(np.digitize(Xs3[:, j], e[1:-1]), 0, nb - 1)
                  for j, e in enumerate(edges)], axis=1)
    flat = (b[:, 0] * nb + b[:, 1]) * nb + b[:, 2]
    for mname in models:
        yt, yp = preds[(mname, "static_random")]
        vi = yt <= thr
        fp = vi & (yp > thr)
        n_vi = np.bincount(flat[vi], minlength=nb ** 3)
        n_fp = np.bincount(flat[fp], minlength=nb ** 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(n_vi >= 20, n_fp / np.maximum(n_vi, 1), np.nan)
        loc[mname] = {"cells_scored": int(np.isfinite(rate).sum()),
                      "worst_cell_rate": float(np.nanmax(rate)) if np.isfinite(rate).any() else None,
                      "cells_above_50pct": int(np.nansum(rate > 0.5)),
                      "cells_above_20pct": int(np.nansum(rate > 0.2))}
        click.echo(f"[cm] {mname:>8}: false-exclusion rate over {loc[mname]['cells_scored']} "
                   f"cells (>=20 viable models each), worst "
                   f"{loc[mname]['worst_cell_rate']:.1%}, "
                   f"{loc[mname]['cells_above_20pct']} cells above 20%, "
                   f"{loc[mname]['cells_above_50pct']} above 50%")

    # ── the rare sub-regions of find_islands.py ─────────────────────────────
    isl = {}
    if islands_json and Path(islands_json).exists():
        comps = json.load(open(islands_json))["components"]
        rare = [c for c in comps if c["p"] < 1e-4]
        for c in rare:
            box = np.array(c["box"])
            m = np.ones(len(Xs3), dtype=bool)
            for j in range(3):
                m &= (Xs3[:, j] >= box[j][0]) & (Xs3[:, j] <= box[j][1])
            rec = {"n_static_rows_in_box": int(m.sum())}
            for mname in models:
                yt, yp = preds[(mname, "static_random")]
                vi = m & (yt <= thr)
                rec[mname] = {
                    "n_viable": int(vi.sum()),
                    "false_exclusion_rate": float((vi & (yp > thr)).sum() / vi.sum())
                    if vi.sum() else None}
            isl[f"component_{c['component']}"] = rec
            click.echo(f"[cm] sub-region {c['component']} box: "
                       f"{rec['n_static_rows_in_box']:,} static rows, "
                       + ", ".join(
                           f"{mn} FE {rec[mn]['false_exclusion_rate']:.1%}"
                           if rec[mn]["false_exclusion_rate"] is not None else f"{mn} n/a"
                           for mn in models))

    payload = {"config": {"run": str(run), "iteration": it, "target": target,
                          "threshold": thr, "model_type": model_type,
                          "n_al_points": int(len(Y_al)),
                          "n_baseline_points": 0 if base_idx is None else int(len(base_idx)),
                          "static_eval_rows": int(len(Y_st))},
               "matrices": res, "paired": paired, "localised": loc,
               "sub_regions": isl}
    (out / "exclusion_confusion.json").write_text(json.dumps(payload, indent=1))
    click.echo(f"[cm] wrote {out / 'exclusion_confusion.json'}")


if __name__ == "__main__":
    main()
