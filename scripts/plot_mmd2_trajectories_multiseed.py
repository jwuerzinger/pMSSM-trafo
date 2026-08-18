"""Plot mean MMD² trajectories vs MCMC reference, with uncertainty bands.

For each iteration of every (model, strategy, warm) seed in the sweep manifest,
compute MMD² between the AL training set so far (free params, normalized) and
a fixed MCMC reference subsample. Aggregate across seeds and plot the
multi-seed mean ± band, mirroring `plot_r2_mcmc_trajectories_multiseed.py`.

MMD² is *not* logged per iteration by the AL pipeline — it's only computed
once at end-of-run by `analyse_runs.py`. This script reconstructs each
training-set snapshot from `state.pt` and computes MMD² post-hoc. Results are
cached in `<output-dir>/mmd2_cache.csv` so subsequent runs are instant; only
new (run_dir, iteration) pairs are recomputed.

A *single* bandwidth (median heuristic) is computed once across the pooled
training + MCMC sample at first run and reused (cached separately) so MMD²
values stay comparable across runs.

Lower MMD² ⇒ training set looks more like the MCMC posterior.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analyse_runs import (  # noqa: E402
    _mmd_gaussian,
    load_mcmc_numpy,
    load_run,
    normalise_free_params,
)


MODEL_COLORS = {
    "transformer": "tab:blue",
    "exact_gp":    "tab:orange",
    "deep_gp":     "tab:green",
    "tabpfn":      "tab:red",
    "dnn":         "tab:purple",
    "transformer_oracle": "tab:blue",
    "deep_gp_oracle":     "tab:green",
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


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_COLUMNS = ["run_dir", "iteration", "mmd2"]


def _load_cache(path: Path) -> dict[tuple[str, int], float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        (str(row["run_dir"]), int(row["iteration"])): float(row["mmd2"])
        for _, row in df.iterrows()
    }


def _append_cache(path: Path, new_rows: list[dict]) -> None:
    if not new_rows:
        return
    df_new = pd.DataFrame(new_rows, columns=_CACHE_COLUMNS)
    if path.exists():
        df_existing = pd.read_csv(path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["run_dir", "iteration"], keep="last")
    else:
        df = df_new
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_or_compute_bandwidth(
    sentinel_path: Path,
    mcmc_X_free_norm: np.ndarray,
    sample_train: np.ndarray | None,
) -> float:
    """Compute & cache the median-heuristic bandwidth once across the project."""
    if sentinel_path.exists():
        with open(sentinel_path) as f:
            return float(json.load(f)["bandwidth"])

    pool = mcmc_X_free_norm
    if sample_train is not None and len(sample_train) > 0:
        pool = np.concatenate([mcmc_X_free_norm, sample_train], axis=0)
    # Replicate the median-heuristic bandwidth used by `_mmd_gaussian`
    # (see analyse_runs.py:_mmd_gaussian) so we can expose & cache the value.
    from sklearn.metrics import pairwise_distances
    if len(pool) > 2000:
        idx = np.random.default_rng(0).choice(len(pool), 2000, replace=False)
        pool = pool[idx]
    d = pairwise_distances(pool, metric="euclidean")
    bw = float(np.median(d[d > 0])) or 1.0

    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sentinel_path, "w") as f:
        json.dump({"bandwidth": bw}, f)
    return bw


# ──────────────────────────────────────────────────────────────────────────────
# Per-run trajectory computation
# ──────────────────────────────────────────────────────────────────────────────

def _mmd2_trajectory(
    run,
    mcmc_X_free_norm: np.ndarray,
    bandwidth: float,
    cache: dict,
    new_cache_rows: list[dict],
    n_subsample: int,
    rng_seed: int = 0,
) -> tuple[list[int], list[float]]:
    """Per-iteration MMD² between cumulative training set and MCMC reference.

    Subsamples both sides to `n_subsample` points (with a deterministic RNG)
    so the kernel matrix stays tractable across thousands of MMD evaluations.
    Each (run_dir, iteration) result is cached; cache hits avoid recomputation.
    """
    run_dir_str = str(run.run_dir)
    rng = np.random.default_rng(rng_seed)
    if len(mcmc_X_free_norm) > n_subsample:
        mcmc_idx = np.random.default_rng(rng_seed).choice(
            len(mcmc_X_free_norm), n_subsample, replace=False
        )
        mcmc_sub = mcmc_X_free_norm[mcmc_idx]
    else:
        mcmc_sub = mcmc_X_free_norm

    iters, mmds = [], []
    for i, n in enumerate(run.n_train_per_iter or []):
        if n is None or n <= 0:
            continue
        n_clip = min(int(n), len(run.X_free_norm))
        if n_clip <= 0:
            continue
        iter_idx = i + 1
        cache_key = (run_dir_str, iter_idx)

        if cache_key in cache:
            mmds.append(cache[cache_key])
            iters.append(iter_idx)
            continue

        X_iter = run.X_free_norm[:n_clip]
        if len(X_iter) > n_subsample:
            sub_idx = rng.choice(len(X_iter), n_subsample, replace=False)
            X_sub = X_iter[sub_idx]
        else:
            X_sub = X_iter
        mmd = _mmd_gaussian(X_sub, mcmc_sub, bandwidth=bandwidth)
        cache[cache_key] = mmd
        new_cache_rows.append(
            {"run_dir": run_dir_str, "iteration": iter_idx, "mmd2": mmd}
        )
        iters.append(iter_idx)
        mmds.append(mmd)
    return iters, mmds


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed aggregation (NaN-padded across seeds, partial-run aware)
# ──────────────────────────────────────────────────────────────────────────────

def _band(Y: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
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


def _collect_trajectories(
    df,
    min_seeds,
    mcmc_X_free_norm,
    bandwidth,
    cache,
    new_cache_rows,
    n_subsample,
):
    out: dict = {}
    for (model, strat, warm), sub in df.groupby(["model", "strategy", "warm_start"]):
        trajs = []
        for run_dir in sub["expected_run_dir"].dropna():
            try:
                run = load_run(run_dir)
                iters, mmds = _mmd2_trajectory(
                    run, mcmc_X_free_norm, bandwidth, cache, new_cache_rows,
                    n_subsample=n_subsample,
                )
                if mmds:
                    trajs.append((iters, mmds))
            except Exception as exc:
                click.echo(f"[warn] skip {run_dir}: {exc}", err=True)
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
        out[(model, strat, warm)] = (iters_ax[keep], Y[:, keep])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

# Markers are unreadable once a trajectory runs past a few tens of iterations,
# so above this many points a curve is drawn as a line only. Kept in step with
# the canonical definition in `plot_hit_rate_trajectories_multiseed.py`.
MARKER_MAX_POINTS = 50


def _markers_on(n_points) -> bool:
    """True when a series is short enough for per-point markers to be legible."""
    try:
        return int(n_points) <= MARKER_MAX_POINTS
    except (TypeError, ValueError):
        return True


def _draw_curve(ax, iters_ax, Y, *, color, linestyle, marker, label, uncertainty):
    # The legend on these figures is built from the curve labels, so dropping
    # the marker here keeps the key and the plot in agreement automatically.
    if not _markers_on(len(np.atleast_1d(iters_ax))):
        marker = None
    lo, hi = _band(Y, uncertainty)
    mean = np.nanmean(Y, axis=0)
    ax.plot(iters_ax, mean, color=color, linestyle=linestyle, marker=marker,
            markersize=3, linewidth=1.5, label=label)
    ax.fill_between(iters_ax, lo, hi, color=color, alpha=0.15)


def _setup_axes(ax, y_min, y_max):
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MMD² (vs MCMC, lower = more MCMC-like)")
    ax.grid(alpha=0.3)
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
    if y_min is not None or y_max is not None:
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(cur_lo if y_min is None else y_min,
                    cur_hi if y_max is None else y_max)


def _finalize(fig, ax, out_path, y_min, y_max):
    _setup_axes(ax, y_min, y_max)
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    if handles:
        fig.subplots_adjust(right=0.78)
        fig.legend(handles, labels,
                   loc="center left", bbox_to_anchor=(0.79, 0.5),
                   fontsize=8, frameon=True, borderaxespad=0.)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_models_per_strategy(traj, uncertainty, out_dir, y_min, y_max):
    written = []
    for strat in sorted({s for (_, s, _) in traj}):
        cfgs = [(m, s, w) for (m, s, w) in traj if s == strat]
        if not cfgs:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"Strategy: {strat} — MMD² vs MCMC", fontsize=12)
        for (m, s, w) in sorted(cfgs):
            iters_ax, Y = traj[(m, s, w)]
            _draw_curve(
                ax, iters_ax, Y,
                color=MODEL_COLORS.get(m, "gray"),
                linestyle=WARM_LS.get(w, "-"),
                marker=WARM_MARKER.get(w, "x"),
                label=f"{m}-{w} (n={len(Y)})",
                uncertainty=uncertainty,
            )
        out_path = out_dir / f"mmd2_strategy_{strat}.png"
        _finalize(fig, ax, out_path, y_min, y_max)
        written.append(out_path)
    return written


def _best_setting_for_model(traj, model):
    """Lower MMD² is better, so pick the (s, w) with the SMALLEST mean final value."""
    candidates = [(m, s, w) for (m, s, w) in traj if m == model]
    if not candidates:
        return None
    scored = []
    for (m, s, w) in candidates:
        _, Y = traj[(m, s, w)]
        scored.append(((s, w), float(np.nanmean(Y, axis=0)[-1])))
    if not scored:
        return None
    (s, w), score = min(scored, key=lambda kv: kv[1])
    return s, w, score


def plot_best_per_model(traj, uncertainty, out_dir, y_min, y_max):
    models = sorted({m for (m, _, _) in traj})
    picks = []
    for model in models:
        chosen = _best_setting_for_model(traj, model)
        if chosen is None:
            continue
        s, w, score = chosen
        picks.append((model, s, w, score))
    if not picks:
        return []
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Best setting per model — lowest mean final MMD² vs MCMC",
                 fontsize=12)
    for (m, s, w, _sc) in picks:
        iters_ax, Y = traj[(m, s, w)]
        _draw_curve(
            ax, iters_ax, Y,
            color=MODEL_COLORS.get(m, "gray"),
            linestyle="-",
            marker="o",
            label=f"{m}: {s}-{w} (n={len(Y)})",
            uncertainty=uncertainty,
        )
    out_path = out_dir / "mmd2_best_per_model.png"
    _finalize(fig, ax, out_path, y_min, y_max)
    click.echo("[best-per-model picks: MMD² (smaller = better)]")
    for (m, s, w, sc) in picks:
        click.echo(f"  {m:12s} -> {s}-{w}  (final mean MMD² = {sc:.6f})")
    return [out_path]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True,
              help="Directory of MCMC ROOT files used as the reference distribution.")
@click.option("--mcmc-max-samples", default=500_000, type=int, show_default=True,
              help="Seeded uniform subsample cap on the MCMC reference "
                   "(preserves chain multiplicity weighting). 0 disables.")
@click.option("--sweep-id", default=None,
              help="Filter to one sweep_id (default: all).")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs/",
              show_default=True)
@click.option("--uncertainty", default="sem",
              type=click.Choice(["sem", "sd", "iqr"]), show_default=True)
@click.option("--min-seeds", default=2, type=int, show_default=True)
@click.option("--include-status", default="completed,running,timeout",
              show_default=True,
              help="Comma-separated statuses to include from the manifest.")
@click.option("--n-subsample", default=2000, type=int, show_default=True,
              help="Per-side subsample size for MMD² (kernel matrix is "
                   "n_subsample × n_subsample; cost ~ N²).")
@click.option("--y-min", default=None, type=float)
@click.option("--y-max", default=None, type=float)
def main(manifest, mcmc_data_dir, mcmc_max_samples, sweep_id, output_dir,
         uncertainty, min_seeds, include_status, n_subsample, y_min, y_max):
    df = pd.read_csv(manifest)
    if sweep_id:
        df = df[df["sweep_id"].astype(str) == str(sweep_id)]
    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    df = df[df["status"].isin(allowed)]
    if df.empty:
        raise click.ClickException("no rows matched manifest filter")

    out_dir = Path(output_dir)
    cache_path = out_dir / "mmd2_cache.csv"
    bandwidth_sentinel = out_dir / "mmd2_bandwidth.json"

    click.echo(f"[mmd2] loading MCMC reference from {mcmc_data_dir} ...")
    mcmc_X, _ = load_mcmc_numpy(mcmc_data_dir, max_samples=mcmc_max_samples or None)
    mcmc_X_free_norm = normalise_free_params(mcmc_X)
    click.echo(f"[mmd2] MCMC: {len(mcmc_X_free_norm)} samples, "
               f"{mcmc_X_free_norm.shape[1]} free params")

    bandwidth = _load_or_compute_bandwidth(
        bandwidth_sentinel, mcmc_X_free_norm, None
    )
    click.echo(f"[mmd2] bandwidth (median heuristic, cached) = {bandwidth:.4f}")

    cache = _load_cache(cache_path)
    click.echo(f"[mmd2] cache: {len(cache)} (run_dir, iter) entries already computed")

    new_cache_rows: list[dict] = []
    traj = _collect_trajectories(
        df, min_seeds, mcmc_X_free_norm, bandwidth, cache, new_cache_rows,
        n_subsample=n_subsample,
    )
    if new_cache_rows:
        _append_cache(cache_path, new_cache_rows)
        click.echo(f"[mmd2] appended {len(new_cache_rows)} new entries to cache")

    if not traj:
        raise click.ClickException("no (model, strategy, warm) groups had enough seeds")

    written = []
    written += plot_models_per_strategy(traj, uncertainty, out_dir, y_min, y_max)
    written += plot_best_per_model(traj, uncertainty, out_dir, y_min, y_max)

    click.echo(f"[plot] wrote {len(written)} file(s) to {out_dir}")
    for p in written:
        click.echo(f"  {p}")


if __name__ == "__main__":
    main()
