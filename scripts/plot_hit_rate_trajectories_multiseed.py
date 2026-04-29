"""Plot mean trajectories with uncertainty bands over N seeds per config.

Reads `sweep_manifest.csv`, groups completed runs by (model, strategy, warm_start),
loads each seed's trajectory, and renders two metric families:

  - `hit_rate_*`         — fraction of *training-set* samples within tolerance
                           of the target (existing definition).
  - `hits_per_desired_*` — same numerator, but divided by the *requested*
                           sample count (CLI `--n-samples + k × --n-select`).
                           This naturally folds in the per-iteration physics
                           generation failure rate, since failed candidates
                           never make it into the training-set numerator.

Each metric produces three figures (one panel per tolerance):

  1. Models per strategy   — one figure per strategy, overlaying every
                              (model, warm) combo. Color = model, ls = warm.
  2. Best setting per model — one figure per metric with one curve per model,
                              picking the setting that maximises that
                              metric's strictest-tolerance final value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Make the repo root importable so we can reuse analyse_runs utilities.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analyse_runs import compute_hit_rate_trajectory, load_run  # noqa: E402
from pmssm import TARGET_CONFIG  # noqa: E402

# Defaults if a run's state.pt is missing the relevant fields.
_N_SAMPLES_DEFAULT = 2000
_N_SELECT_DEFAULT = 500


MODEL_COLORS = {
    "transformer": "tab:blue",
    "exact_gp":    "tab:orange",
    "deep_gp":     "tab:green",
    "tabpfn":      "tab:red",
}
STRATEGY_COLORS = {
    "top_k":          "tab:blue",
    "top_k_tol_only": "tab:orange",
    "entropy_batch":  "tab:green",
}
WARM_LS = {
    "warm":   "-",
    "cold":   "--",
    "tabpfn": "-",
}
WARM_MARKER = {
    "warm":   "o",
    "cold":   "s",
    "tabpfn": "^",
}


def _band(Y: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) uncertainty bands for an (n_seeds, n_iters) array.

    NaN-aware: at each iteration only the seeds that reported a value contribute.
    Iterations with zero or one valid seed get a zero-width band (band == mean).
    """
    n_per_iter = np.sum(~np.isnan(Y), axis=0)
    mean = np.nanmean(Y, axis=0)
    if mode == "sem":
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.nanstd(Y, axis=0, ddof=1)
        sd = np.where(n_per_iter > 1, sd, 0.0)
        half = sd / np.sqrt(np.clip(n_per_iter, 1, None))
        return mean - half, mean + half
    if mode == "sd":
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.nanstd(Y, axis=0, ddof=1)
        sd = np.where(n_per_iter > 1, sd, 0.0)
        return mean - sd, mean + sd
    if mode == "iqr":
        return np.nanpercentile(Y, 25, axis=0), np.nanpercentile(Y, 75, axis=0)
    raise ValueError(f"unknown uncertainty mode: {mode}")


def _desired_per_iter(run) -> list[int]:
    """Cumulative requested-sample count aligned with run.n_train_per_iter indices.

    desired[i] = n_samples + i × n_select_per_iter[0..i-1] sum
                = the total points the user asked the CLI tools to evaluate
                  by the time iteration (i+1)'s training step runs.

    `n_samples` is recovered as al_n_train[0] + al_n_val[0] (training-set
    initial size + validation-set initial size, since both sides of the
    train/val split come from the requested initial pool).

    `n_select` per iteration is recovered from `all_selected_points[k]["points"]`;
    we use the row count there, falling back to a default if the field is
    absent or empty.
    """
    state_path = Path(run.run_dir) / "state.pt"
    state = torch.load(state_path, weights_only=False, map_location="cpu")

    al_n_train = list(run.n_train_per_iter or [])
    al_n_val_raw = state.get("al_n_val") or []
    al_n_val = list(al_n_val_raw.tolist()) if hasattr(al_n_val_raw, "tolist") else list(al_n_val_raw)

    if al_n_train and al_n_val:
        n_samples = int(al_n_train[0]) + int(al_n_val[0])
    elif al_n_train:
        n_samples = int(al_n_train[0])
    else:
        n_samples = _N_SAMPLES_DEFAULT

    selected = state.get("all_selected_points") or []
    n_select_per_iter = []
    for entry in selected:
        pts = entry.get("points") if isinstance(entry, dict) else None
        if pts is None:
            n_select_per_iter.append(_N_SELECT_DEFAULT)
        else:
            n_select_per_iter.append(len(pts) or _N_SELECT_DEFAULT)

    desired = []
    cum = n_samples
    for i in range(len(al_n_train)):
        desired.append(cum)
        n_sel = n_select_per_iter[i] if i < len(n_select_per_iter) else _N_SELECT_DEFAULT
        cum += n_sel
    return desired


def _hits_per_desired_trajectory(run, true_value, tol):
    """Per-iteration cumulative hit count divided by cumulative requested count.

    Numerator at iter (i+1) = #{ Y[:al_n_train[i]] within `tol` of `true_value` }
    Denominator at iter (i+1) = `_desired_per_iter(run)[i]`

    Returns (iters, rates) with the same `iters` axis as `compute_hit_rate_trajectory`.
    """
    desired = _desired_per_iter(run)
    iters, rates = [], []
    for i, n in enumerate(run.n_train_per_iter):
        if n is None or n <= 0:
            continue
        n_clip = min(int(n), len(run.Y))
        Y_slice = run.Y[:n_clip]
        if hasattr(Y_slice, "numpy"):
            Y_slice = Y_slice.numpy()
        Y_slice = np.asarray(Y_slice).ravel()
        hits = int(np.sum(np.abs(Y_slice - true_value) / true_value < tol))
        denom = desired[i] if i < len(desired) else (n_clip or 1)
        if denom <= 0:
            continue
        iters.append(i + 1)
        rates.append(hits / denom)
    return iters, rates


def _load_y_full(data_dir: str, target: str, cache_dir: Path) -> np.ndarray:
    """Load (or read from .npy cache) the full unshuffled Y pool used by AL runs.

    The cache file is keyed by `data_dir` + `target` so changing either
    invalidates it. ROOT loading is slow (~1 min, 11 GB); the cache turns
    subsequent re-renders into a millisecond op.
    """
    safe = data_dir.replace("/", "_").strip("_")
    cache = cache_dir / f"y_full_{safe}_{target}.npy"
    if cache.exists():
        return np.load(cache)
    from pmssm.data import load_pmssm_data  # noqa: PLC0415
    _X, Y = load_pmssm_data(n_datasets=-1, target=target, data_dir=data_dir,
                             plot_dir=str(cache_dir))
    Y = Y.numpy().ravel().astype(np.float64) if hasattr(Y, "numpy") else np.asarray(Y).ravel()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, Y)
    return Y


def _seed_shuffled_y(Y_full: np.ndarray, seed: int) -> np.ndarray:
    """Replay the AL driver's per-seed `_load_perm` shuffle on Y_full."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(Y_full), generator=g).numpy()
    return Y_full[perm]


def _baseline_iter_y(state: dict, Y_full_shuffled: np.ndarray, iter_idx: int) -> np.ndarray | None:
    """Reconstruct the baseline cumulative training Y values at iteration `iter_idx`.

    Layout of `baseline_add_indices` (from active_learning.py): all train indices
    in addition order, followed by all val indices. So the first
    `n_added_train` entries always recover the baseline-train indices ever
    added by iteration i.
    """
    al_n_train = list(state.get("al_n_train") or [])
    base_n_train = list(state.get("baseline_n_train") or [])
    if iter_idx >= len(base_n_train) or not al_n_train:
        return None
    n_train_init = int(al_n_train[0])
    n_total = int(base_n_train[iter_idx])
    if n_total <= 0:
        return None
    n_added = max(0, n_total - n_train_init)

    Y_state = state.get("Y")
    Y_state = Y_state.numpy().ravel() if hasattr(Y_state, "numpy") else np.asarray(Y_state).ravel()
    Y_init = Y_state[:n_train_init]

    if n_added == 0:
        return Y_init
    add_idx = state.get("baseline_add_indices")
    add_idx = add_idx.numpy() if hasattr(add_idx, "numpy") else np.asarray(add_idx)
    if len(add_idx) < n_added:
        return None
    Y_added = Y_full_shuffled[add_idx[:n_added]]
    return np.concatenate([Y_init, Y_added])


def _baseline_hit_rate_trajectory(run_dir: str, seed: int, Y_full: np.ndarray,
                                  true_value: float, tol: float):
    state_path = Path(run_dir) / "state.pt"
    state = torch.load(state_path, weights_only=False, map_location="cpu")
    Y_full_shuffled = _seed_shuffled_y(Y_full, seed)
    base_n_train = list(state.get("baseline_n_train") or [])
    iters, rates = [], []
    for i in range(len(base_n_train)):
        Y_b = _baseline_iter_y(state, Y_full_shuffled, i)
        if Y_b is None or len(Y_b) == 0:
            continue
        hits = int(np.sum(np.abs(Y_b - true_value) / true_value < tol))
        iters.append(i + 1)
        rates.append(hits / len(Y_b))
    return iters, rates


def _baseline_hits_per_desired_trajectory(run_dir: str, seed: int, Y_full: np.ndarray,
                                          true_value: float, tol: float):
    """Random-scan reference for the hits/desired panel.

    Uses the *baseline's own sample count* as denominator (= the hit rate of
    the random-baseline samples). The AL hits/desired metric exists to
    penalise physics-generation failures, but the baseline draws from a
    pre-filtered pool with no failures — the honest baseline is therefore the
    pool prevalence (i.e. the same value as the hit_rate panel), not
    `hits / desired`, which artificially shrinks because `baseline_n_train` is
    coupled to AL's smaller `al_n_train`, not the full physics budget.
    """
    return _baseline_hit_rate_trajectory(run_dir, seed, Y_full, true_value, tol)


def _pool_prevalence(Y_full: np.ndarray, true_value: float, tols) -> dict:
    return {float(t): float(np.mean(np.abs(Y_full - true_value) / true_value < t)) for t in tols}


# Registry of metrics: name → (trajectory_fn, file_prefix, axis_label, title_word)
METRICS = {
    "hit_rate": (
        compute_hit_rate_trajectory,
        "hit_rate",
        "Hit rate",
        "Hit rate",
        _baseline_hit_rate_trajectory,
    ),
    "hits_per_desired": (
        _hits_per_desired_trajectory,
        "hits_per_desired",
        "Hits / Desired",
        "Hits / Desired",
        _baseline_hits_per_desired_trajectory,
    ),
}


def _collect_trajectories(df, true_val, tols, min_seeds, traj_fn):
    """Build {(model, strategy, warm): {tol: (iters_axis, Y[n_seeds, n_iters])}}.

    Each run is loaded once and re-used across tolerances. Trajectories of
    different lengths (e.g. partially-completed runs whose status is
    ``running`` or ``timeout``) are NaN-padded to the longest seed's length
    so that the per-iteration mean / band uses whichever seeds have data at
    that iter. Iterations where fewer than `min_seeds` seeds reported a value
    are dropped from the output, so the right-hand tail truncates cleanly
    when only one or two seeds got further than the rest.
    """
    out: dict = {}
    for (model, strat, warm), sub in df.groupby(["model", "strategy", "warm_start"]):
        runs = []
        for run_dir in sub["expected_run_dir"].dropna():
            try:
                runs.append((run_dir, load_run(run_dir)))
            except Exception as exc:
                click.echo(f"[warn] skip {run_dir}: {exc}", err=True)
        per_tol = {}
        for tol in tols:
            trajs = []
            for run_dir, run in runs:
                try:
                    iters, rates = traj_fn(run, true_val, tol)
                    if rates:
                        trajs.append((iters, rates))
                except Exception as exc:
                    click.echo(f"[warn] skip {run_dir} tol={tol}: {exc}", err=True)
            if len(trajs) < min_seeds:
                continue
            max_len = max(len(r) for _, r in trajs)
            Y = np.full((len(trajs), max_len), np.nan, dtype=float)
            for i, (_, rates) in enumerate(trajs):
                Y[i, :len(rates)] = rates
            longest_iters = next(its for its, r in trajs if len(r) == max_len)
            iters_ax = np.asarray(longest_iters[:max_len])
            n_per_iter = np.sum(~np.isnan(Y), axis=0)
            keep = n_per_iter >= min_seeds
            if not keep.any():
                continue
            per_tol[tol] = (iters_ax[keep], Y[:, keep])
        if per_tol:
            out[(model, strat, warm)] = per_tol
    return out


def _collect_baseline_trajectories(df, true_val, tols, min_seeds, traj_fn_baseline,
                                   Y_full):
    """Like `_collect_trajectories` but builds the random-baseline trajectories.

    Per-run: needs the seed (for replaying the AL driver's `_load_perm`) and
    the run_dir (for `state.pt`).
    """
    out: dict = {}
    for (model, strat, warm), sub in df.groupby(["model", "strategy", "warm_start"]):
        per_tol = {}
        run_seeds = list(zip(sub["expected_run_dir"].dropna(), sub["seed"]))
        for tol in tols:
            trajs = []
            for run_dir, seed in run_seeds:
                try:
                    iters, rates = traj_fn_baseline(run_dir, int(seed), Y_full,
                                                    true_val, tol)
                    if rates:
                        trajs.append((iters, rates))
                except Exception as exc:
                    click.echo(f"[warn] baseline skip {run_dir} tol={tol}: {exc}", err=True)
            if len(trajs) < min_seeds:
                continue
            max_len = max(len(r) for _, r in trajs)
            Y = np.full((len(trajs), max_len), np.nan, dtype=float)
            for i, (_, rates) in enumerate(trajs):
                Y[i, :len(rates)] = rates
            longest_iters = next(its for its, r in trajs if len(r) == max_len)
            iters_ax = np.asarray(longest_iters[:max_len])
            n_per_iter = np.sum(~np.isnan(Y), axis=0)
            keep = n_per_iter >= min_seeds
            if not keep.any():
                continue
            per_tol[tol] = (iters_ax[keep], Y[:, keep])
        if per_tol:
            out[(model, strat, warm)] = per_tol
    return out


def _draw_curve(ax, iters_ax, Y, *, color, linestyle, marker, label,
                uncertainty, linewidth=1.5, fill_alpha=0.15, alpha=1.0):
    lo, hi = _band(Y, uncertainty)
    mean = np.nanmean(Y, axis=0)
    ax.plot(iters_ax, mean, color=color, linestyle=linestyle, marker=marker,
            markersize=3, linewidth=linewidth, label=label, alpha=alpha)
    if fill_alpha > 0:
        ax.fill_between(iters_ax, lo, hi, color=color, alpha=fill_alpha)


def _setup_axes(axes, tols, true_val, title_word, ylabel):
    for ax, tol in zip(axes, tols):
        ax.set_title(f"{title_word} (|Ω − {true_val}| / {true_val} < {int(tol*100)}%)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)


def _finalize(fig, axes, out_path):
    for ax in axes:
        _, ymax = ax.get_ylim()
        ax.set_ylim(0, max(ymax, 0.05) * 1.05)

    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)

    fig.tight_layout()
    if seen:
        fig.subplots_adjust(right=0.80, wspace=0.28)
        fig.legend(seen.values(), seen.keys(),
                   loc="center left", bbox_to_anchor=(0.81, 0.5),
                   fontsize=8, frameon=True, borderaxespad=0.)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_models_per_strategy(traj, tols, uncertainty, true_val, out_dir,
                             file_prefix, title_word, ylabel):
    """One figure per strategy; lines are (model, warm) combos for that strategy.

    Each figure has 1 row × len(tols) cols (default 3 panels: 10/20/50%).
    Color encodes the model, linestyle encodes the warm-start variant.
    """
    written = []
    strategies = sorted({s for (_, s, _) in traj})
    for strat in strategies:
        cfgs = [(m, s, w) for (m, s, w) in traj if s == strat]
        if not cfgs:
            continue
        fig, axes = plt.subplots(1, len(tols), figsize=(6 * len(tols), 5),
                                 sharey=False, squeeze=False)
        axes = list(axes.flat)
        _setup_axes(axes, tols, true_val, title_word, ylabel)
        fig.suptitle(f"Strategy: {strat} — model & warm-start comparison",
                     fontsize=12)

        for ax, tol in zip(axes, tols):
            for (m, s, w) in sorted(cfgs):
                if tol not in traj[(m, s, w)]:
                    continue
                iters_ax, Y = traj[(m, s, w)][tol]
                _draw_curve(
                    ax, iters_ax, Y,
                    color=MODEL_COLORS.get(m, "gray"),
                    linestyle=WARM_LS.get(w, "-"),
                    marker=WARM_MARKER.get(w, "x"),
                    label=f"{m}-{w} (n={len(Y)})",
                    uncertainty=uncertainty,
                )

        out_path = out_dir / f"{file_prefix}_strategy_{strat}.png"
        _finalize(fig, axes, out_path)
        written.append(out_path)
    return written


def _best_setting_for_model(traj, model, tols):
    """Pick the (strategy, warm) for `model` with highest mean final-iter hit rate.

    Tries the strictest tolerance first; if no config has data there, falls back
    to progressively looser tolerances. Returns (strategy, warm, tol_used, score)
    or None if no eligible config exists.
    """
    candidates = [(m, s, w) for (m, s, w) in traj if m == model]
    if not candidates:
        return None
    for tol in sorted(tols):  # strictest first
        scored = []
        for (m, s, w) in candidates:
            if tol not in traj[(m, s, w)]:
                continue
            _, Y = traj[(m, s, w)][tol]
            scored.append(((s, w), float(np.nanmean(Y, axis=0)[-1])))
        if scored:
            (s, w), score = max(scored, key=lambda kv: kv[1])
            return s, w, tol, score
    return None


def plot_best_per_model(traj, tols, uncertainty, true_val, out_dir,
                        file_prefix, title_word, ylabel,
                        baseline_traj=None, prevalence=None):
    """Single figure: one curve per model using its best (strategy, warm) setting.

    If `baseline_traj` is provided (same {(model, strat, warm): {tol: ...}}
    structure), each picked config also gets a dashed random-baseline curve in
    its model colour. If `prevalence` is provided ({tol: rate}), a horizontal
    grey reference line is drawn on each panel.
    """
    models = sorted({m for (m, _, _) in traj})
    picks = []  # (model, strat, warm, tol_used, score)
    for model in models:
        chosen = _best_setting_for_model(traj, model, tols)
        if chosen is None:
            continue
        s, w, tol_used, score = chosen
        picks.append((model, s, w, tol_used, score))

    if not picks:
        return []

    fig, axes = plt.subplots(1, len(tols), figsize=(6 * len(tols), 5), sharey=False)
    if len(tols) == 1:
        axes = [axes]
    _setup_axes(axes, tols, true_val, title_word, ylabel)

    strict_tol = min(tols)
    fig.suptitle(
        f"Best setting per model "
        f"(picked by mean final {title_word.lower()} @ tol={int(strict_tol*100)}%)",
        fontsize=12,
    )

    for ax, tol in zip(axes, tols):
        for (m, s, w, _tu, _sc) in picks:
            cfg = (m, s, w)
            if tol not in traj[cfg]:
                continue
            iters_ax, Y = traj[cfg][tol]
            _draw_curve(
                ax, iters_ax, Y,
                color=MODEL_COLORS.get(m, "gray"),
                linestyle="-",
                marker="o",
                label=f"{m}: {s}-{w} (n={len(Y)})",
                uncertainty=uncertainty,
            )
            if baseline_traj is not None and cfg in baseline_traj \
                    and tol in baseline_traj[cfg]:
                b_iters, b_Y = baseline_traj[cfg][tol]
                _draw_curve(
                    ax, b_iters, b_Y,
                    color=MODEL_COLORS.get(m, "gray"),
                    linestyle="--",
                    marker=None,
                    label=f"{m}: random baseline",
                    uncertainty=uncertainty,
                    linewidth=1.8, fill_alpha=0.0, alpha=0.85,
                )
        if prevalence is not None and tol in prevalence:
            ax.axhline(prevalence[tol], color="black", linestyle=":", linewidth=1.4,
                       label="random scan (full pool)")
            ax.text(0.99, prevalence[tol], f" {prevalence[tol]:.4f}",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=8, color="black")

    out_path = out_dir / f"{file_prefix}_best_per_model.png"
    _finalize(fig, axes, out_path)

    click.echo(f"[best-per-model picks: {title_word}]")
    for (m, s, w, tu, sc) in picks:
        click.echo(f"  {m:12s} -> {s}-{w}  (final mean@{int(tu*100)}% = {sc:.4f})")

    return [out_path]


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--sweep-id", default=None,
              help="Filter to one sweep_id (default: use all completed rows).")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs/",
              show_default=True,
              help="Directory for the generated PNGs. For each metric "
                   "(hit_rate, hits_per_desired) the script writes "
                   "<metric>_strategy_<strategy>.png (one per strategy) and "
                   "<metric>_best_per_model.png.")
@click.option("--uncertainty", default="sem",
              type=click.Choice(["sem", "sd", "iqr"]), show_default=True,
              help="Band: SEM (default), SD, or IQR across seeds.")
@click.option("--target", default="DMRD", show_default=True,
              help="TARGET_CONFIG key (threshold + true_value source).")
@click.option("--tolerances", default="0.10,0.20,0.50", show_default=True,
              help="Comma-separated relative tolerances for hit-rate panels.")
@click.option("--min-seeds", default=2, type=int, show_default=True,
              help="Drop groups with fewer completed seeds than this.")
@click.option("--include-status", default="completed,running,timeout",
              show_default=True,
              help="Comma-separated statuses to include from the manifest. "
                   "`running` and `timeout` rows surface partial trajectories "
                   "alongside completed seeds; the per-iteration band uses "
                   "whichever seeds have data at that iter.")
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358",
              show_default=True,
              help="ROOT data directory used to compute the random-scan baseline "
                   "(pool prevalence + per-run baseline trajectories). Set to "
                   "empty string to disable the baseline overlay.")
def main(manifest, sweep_id, output_dir, uncertainty, target, tolerances,
         min_seeds, include_status, baseline_data_dir):
    df = pd.read_csv(manifest)
    if sweep_id:
        df = df[df["sweep_id"].astype(str) == str(sweep_id)]
    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    df = df[df["status"].isin(allowed)]
    if df.empty:
        raise click.ClickException("no rows matched manifest filter")

    tols = [float(t) for t in tolerances.split(",")]
    true_val = TARGET_CONFIG[target]["true_value"]

    out_dir = Path(output_dir)

    Y_full = None
    prevalence = None
    if baseline_data_dir:
        try:
            Y_full = _load_y_full(baseline_data_dir, target, out_dir)
            prevalence = _pool_prevalence(Y_full, true_val, tols)
            with open(out_dir / "random_baseline_prevalence.json", "w") as fh:
                json.dump({"data_dir": baseline_data_dir, "target": target,
                           "true_value": true_val, "n_pool": int(len(Y_full)),
                           "prevalence": {f"{t:.4f}": v for t, v in prevalence.items()}},
                          fh, indent=2)
            click.echo(f"[baseline] pool prevalence (n={len(Y_full)}): "
                       + ", ".join(f"tol={int(t*100)}%→{r:.4f}" for t, r in prevalence.items()))
        except Exception as exc:
            click.echo(f"[warn] could not load random baseline pool from {baseline_data_dir}: {exc}",
                       err=True)
            Y_full = None
            prevalence = None

    written = []
    for metric_name, (traj_fn, file_prefix, ylabel, title_word, traj_fn_baseline) in METRICS.items():
        traj = _collect_trajectories(df, true_val, tols, min_seeds, traj_fn)
        if not traj:
            click.echo(f"[warn] metric '{metric_name}': no groups passed min-seeds filter; skipping",
                       err=True)
            continue
        baseline_traj = None
        if Y_full is not None:
            baseline_traj = _collect_baseline_trajectories(
                df, true_val, tols, min_seeds, traj_fn_baseline, Y_full,
            )
        written += plot_models_per_strategy(
            traj, tols, uncertainty, true_val, out_dir,
            file_prefix=file_prefix, title_word=title_word, ylabel=ylabel,
        )
        written += plot_best_per_model(
            traj, tols, uncertainty, true_val, out_dir,
            file_prefix=file_prefix, title_word=title_word, ylabel=ylabel,
            baseline_traj=baseline_traj, prevalence=prevalence,
        )

    if not written:
        raise click.ClickException("no plots produced — every metric had too few seeds")

    click.echo(f"[plot] wrote {len(written)} file(s) to {out_dir}")
    for p in written:
        click.echo(f"  {p}")


if __name__ == "__main__":
    main()
