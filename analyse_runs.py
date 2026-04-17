#!/usr/bin/env python
"""
analyse_runs.py — Cross-run quality, diversity and physical-property analysis
for pMSSM active learning pipelines.

Overview
--------
Each active learning run writes a ``state.pt`` checkpoint that accumulates the
full training dataset and per-iteration evaluation metrics.  This script loads
one or more such runs, computes a standardised set of quality and diversity
metrics, and writes comparison plots plus a ``summary.csv``.

Metrics
-------
Quality (does the run find physically viable models?)
  * Hit rate: fraction of training points within ε of the target relic density
    (|Ω − 0.12| / 0.12 < ε) at tolerances 10 %, 20 %, 50 %.
  * Static-random eval R²: generalisation to a held-out random sample —
    stored per iteration in ``state.pt``.
  * MCMC eval R²: generalisation to the physically motivated MCMC posterior.
  * Convergence speed: earliest iteration at which R² exceeds 0.5 / 0.7 / 0.9.

Physical properties of accumulated training data
  * Mean and std of Ω across all training points.
  * Per-parameter sample variance (free parameters only).
  * Log-generalised variance: log-determinant of the covariance matrix of the
    free parameters — a single number summarising the "volume" of parameter
    space covered.

Diversity (does the run explore the parameter space broadly?)
  * Per-parameter Shannon entropy: computed from 20-bin histograms of each
    free parameter in normalised [0, 1] space.  Maximum entropy is
    log₂(20) ≈ 4.32 bits (uniform distribution).
  * k-NN mean distance: for each training point, the mean Euclidean distance
    to its k=5 nearest neighbours in normalised free-parameter space.  Higher
    values indicate sparser, more spread-out coverage.
  * MMD vs MCMC: Maximum Mean Discrepancy with a Gaussian kernel between the
    training data and the MCMC posterior.  Lower MMD means the selected models
    are distributed more like physically motivated models.
  * MCMC cluster coverage: K-means is fit once on the MCMC data; coverage is
    the fraction of clusters that contain at least one AL training point.

All scalar metrics are accompanied by ±1σ bootstrap uncertainty estimates
(n=500 resamples).  MMD uncertainty is derived from a permutation test.

Usage
-----
    python analyse_runs.py \\
        --run-dirs  /ptmp/jwuerzin/output/run_A  /ptmp/jwuerzin/output/run_B \\
        --labels    transformer_entropy  transformer_top_k \\
        --mcmc-data-dir /ptmp/jwuerzin/data/19250082 \\
        --output-dir analysis_output

    # Include baseline model metrics alongside AL metrics
    python analyse_runs.py --run-dirs ... --include-baseline

    # Skip bootstrap (faster but no error bars)
    python analyse_runs.py --run-dirs ... --n-bootstrap 0
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm as scipy_norm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

# ── project imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from pmssm.config import PARAM_ORDER, PARAM_RANGES, TARGET_CONFIG
from pmssm.resume import load_state

# ── free-parameter bookkeeping ──────────────────────────────────────────────────
# Parameters whose lower and upper range bounds are equal are fixed in the
# current pMSSM scan.  They carry no information about model diversity and are
# excluded from all diversity and variance metrics.
FREE_PARAM_INDICES: list[int] = [
    i for i, p in enumerate(PARAM_ORDER)
    if PARAM_RANGES[p][0] != PARAM_RANGES[p][1]
]
FREE_PARAM_NAMES: list[str] = [
    PARAM_ORDER[i].replace("IN_", "") for i in FREE_PARAM_INDICES
]
FREE_PARAM_LO: np.ndarray = np.array(
    [PARAM_RANGES[PARAM_ORDER[i]][0] for i in FREE_PARAM_INDICES], dtype=np.float64
)
FREE_PARAM_HI: np.ndarray = np.array(
    [PARAM_RANGES[PARAM_ORDER[i]][1] for i in FREE_PARAM_INDICES], dtype=np.float64
)

# 2-D scatter projections to draw in pairwise_scatter.png
SCATTER_PAIRS: list[tuple[str, str]] = [
    ("M_1", "M_2"),
    ("M_1", "mu"),
    ("M_2", "mu"),
    ("meL", "meR"),
    ("M_1", "tanb"),
    ("At", "tanb"),
]

# ── colour palette ──────────────────────────────────────────────────────────────
PALETTE = plt.cm.tab10.colors  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunData:
    """Container for one active learning run's data and per-iteration metrics.

    Attributes
    ----------
    label : str
        Human-readable identifier (derived from directory name or --labels).
    run_dir : Path
        Path to the run's output directory.
    X : np.ndarray, shape (N, 19)
        Accumulated training inputs in physical units at end of run.
    Y : np.ndarray, shape (N,)
        Accumulated training targets (relic density Ω) in physical units.
    X_free_norm : np.ndarray, shape (N, n_free)
        Free parameters only, normalised to [0, 1] using PARAM_RANGES.
    n_iters : int
        Number of completed AL iterations.
    n_train_per_iter : list[int]
        Cumulative training-set size after each iteration (``al_n_train``).
    al_r2 : list[float]
        AL validation R² per iteration.
    baseline_r2 : list[float]
        Baseline validation R² per iteration.
    al_static_r2 : list[float | None]
        AL model R² on the static random eval set per iteration (may be empty).
    baseline_static_r2 : list[float | None]
        Baseline model R² on the static random eval set per iteration.
    al_mcmc_r2 : list[float | None]
        AL model R² on the MCMC eval set per iteration (may be empty).
    baseline_mcmc_r2 : list[float | None]
        Baseline model R² on the MCMC eval set per iteration.
    """
    label: str
    run_dir: Path
    X: np.ndarray
    Y: np.ndarray
    X_free_norm: np.ndarray
    n_iters: int
    n_train_per_iter: list
    al_r2: list
    baseline_r2: list
    al_static_r2: list
    baseline_static_r2: list
    al_mcmc_r2: list
    baseline_mcmc_r2: list


# ══════════════════════════════════════════════════════════════════════════════
# Helper: normalisation
# ══════════════════════════════════════════════════════════════════════════════

def normalise_free_params(X: np.ndarray) -> np.ndarray:
    """Extract and min-max normalise the free pMSSM parameters to [0, 1].

    Parameters
    ----------
    X : np.ndarray, shape (N, 19)
        Full parameter matrix in physical units.

    Returns
    -------
    np.ndarray, shape (N, n_free)
        Free parameters only, each scaled to [0, 1] using ``PARAM_RANGES``.
    """
    X_free = X[:, FREE_PARAM_INDICES].astype(np.float64)
    return (X_free - FREE_PARAM_LO) / (FREE_PARAM_HI - FREE_PARAM_LO)


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════

def _derive_label(run_dir: Path) -> str:
    """Derive a short human-readable label from the run directory name.

    Strips leading ``active_learning_`` / ``al_`` prefixes and trailing
    timestamp suffixes (``_YYYYMMDD_HHMMSS``).

    Parameters
    ----------
    run_dir : Path
        Run output directory path.

    Returns
    -------
    str
        Shortened label, e.g. ``"exact_gp_top_k_single"``.
    """
    name = run_dir.name
    name = re.sub(r"_(output|slurm)_?", "_", name)
    name = re.sub(r"_\d{8}_\d{6}$", "", name)
    name = re.sub(r"^active_learning_", "", name)
    name = re.sub(r"^al_", "", name)
    name = name.strip("_")
    return name


def _safe_list(state: dict, key: str) -> list:
    """Return ``state[key]`` as a Python list, or ``[]`` if the key is absent."""
    val = state.get(key, [])
    if isinstance(val, torch.Tensor):
        val = val.tolist()
    return list(val)


def load_run(run_dir: str | Path, label: str | None = None) -> RunData:
    """Load a completed (or partial) active learning run from its output directory.

    Reads ``state.pt`` and extracts the accumulated training data and all
    per-iteration evaluation metrics stored by the AL driver.

    Parameters
    ----------
    run_dir : str or Path
        Path to the run's output directory (must contain ``state.pt``).
    label : str, optional
        Human-readable name.  If ``None``, derived automatically from the
        directory name.

    Returns
    -------
    RunData

    Raises
    ------
    FileNotFoundError
        If ``state.pt`` is not found in ``run_dir``.
    """
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    if state is None:
        raise FileNotFoundError(f"state.pt not found in {run_dir}")

    label = label or _derive_label(run_dir)

    # ── training data ──────────────────────────────────────────────────────────
    X_t = state.get("X")
    Y_t = state.get("Y")
    if X_t is None or Y_t is None:
        raise KeyError(f"state.pt in {run_dir} is missing 'X' or 'Y'")

    X = X_t.numpy().astype(np.float64) if isinstance(X_t, torch.Tensor) else np.asarray(X_t, dtype=np.float64)
    Y = Y_t.numpy().ravel().astype(np.float64) if isinstance(Y_t, torch.Tensor) else np.asarray(Y_t, dtype=np.float64).ravel()
    X_free_norm = normalise_free_params(X)

    n_iters = int(state.get("iteration", 0))

    return RunData(
        label=label,
        run_dir=run_dir,
        X=X,
        Y=Y,
        X_free_norm=X_free_norm,
        n_iters=n_iters,
        n_train_per_iter=_safe_list(state, "al_n_train"),
        al_r2=_safe_list(state, "al_r2_scores"),
        baseline_r2=_safe_list(state, "baseline_r2_scores"),
        al_static_r2=_safe_list(state, "al_on_static_random_r2"),
        baseline_static_r2=_safe_list(state, "baseline_on_static_random_r2"),
        al_mcmc_r2=_safe_list(state, "al_on_mcmc_r2"),
        baseline_mcmc_r2=_safe_list(state, "baseline_on_mcmc_r2"),
    )


def load_mcmc_numpy(mcmc_data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load MCMC evaluation data, using the project's ``load_mcmc_data`` helper.

    Parameters
    ----------
    mcmc_data_dir : str or Path
        Directory containing MCMC ROOT files (e.g. ``/ptmp/.../19250082``).

    Returns
    -------
    X_mcmc : np.ndarray, shape (N, 19)
        MCMC parameter samples in physical units.
    Y_mcmc : np.ndarray, shape (N,)
        MCMC relic density values in physical units.
    """
    from pmssm.data import load_mcmc_data  # noqa: PLC0415 (lazy import — uproot is optional)
    X_t, Y_t = load_mcmc_data(data_dir=str(mcmc_data_dir))
    X = X_t.numpy().astype(np.float64) if isinstance(X_t, torch.Tensor) else np.asarray(X_t, dtype=np.float64)
    Y = Y_t.numpy().ravel().astype(np.float64) if isinstance(Y_t, torch.Tensor) else np.asarray(Y_t, dtype=np.float64).ravel()
    return X, Y


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap helpers
# ══════════════════════════════════════════════════════════════════════════════

def _bootstrap_scalar(
    data: np.ndarray,
    func,
    n_bootstrap: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Estimate a scalar statistic and its bootstrap standard error.

    Parameters
    ----------
    data : np.ndarray, shape (N, ...)
        Data to resample.  Resampling is along axis 0.
    func : callable
        Function mapping an (N, ...) array to a scalar.
    n_bootstrap : int
        Number of bootstrap resamples.  Set to 0 to skip (returns NaN SE).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    value : float
        Statistic computed on the original data.
    se : float
        Bootstrap standard error (std of bootstrap distribution).  NaN when
        ``n_bootstrap == 0``.
    """
    value = float(func(data))
    if n_bootstrap == 0:
        return value, float("nan")
    rng = rng or np.random.default_rng(0)
    n = len(data)
    stats = [float(func(data[rng.integers(0, n, n)])) for _ in range(n_bootstrap)]
    return value, float(np.std(stats, ddof=1))


def _bootstrap_vector(
    data: np.ndarray,
    func,
    n_bootstrap: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a vector-valued statistic and its per-element bootstrap SE.

    Parameters
    ----------
    data : np.ndarray, shape (N, ...)
    func : callable
        Maps (N, ...) → 1-D array of fixed length.
    n_bootstrap : int
    rng : np.random.Generator, optional

    Returns
    -------
    values : np.ndarray
    ses : np.ndarray
        Per-element standard errors (NaN when ``n_bootstrap == 0``).
    """
    values = np.asarray(func(data), dtype=np.float64)
    if n_bootstrap == 0:
        return values, np.full_like(values, float("nan"))
    rng = rng or np.random.default_rng(0)
    n = len(data)
    boot = np.stack([func(data[rng.integers(0, n, n)]) for _ in range(n_bootstrap)])
    return values, np.std(boot, axis=0, ddof=1)


# ══════════════════════════════════════════════════════════════════════════════
# Quality metrics
# ══════════════════════════════════════════════════════════════════════════════

HIT_TOLERANCES: list[float] = [0.10, 0.20, 0.50]
R2_THRESHOLDS: list[float] = [0.5, 0.7, 0.9]


def compute_hit_rate(Y: np.ndarray, true_value: float, tol: float) -> float:
    """Fraction of samples within ``tol`` relative tolerance of ``true_value``.

    Parameters
    ----------
    Y : np.ndarray, shape (N,)
        Target values in physical units.
    true_value : float
        Reference value (e.g. 0.12 for Ω).
    tol : float
        Relative tolerance, e.g. 0.10 for 10 %.

    Returns
    -------
    float
        Hit rate ∈ [0, 1].
    """
    return float(np.mean(np.abs(Y - true_value) / true_value < tol))


def compute_quality_metrics(
    run: RunData,
    true_value: float,
    n_bootstrap: int = 500,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute quality metrics for a single run.

    Parameters
    ----------
    run : RunData
    true_value : float
        Physical target value (e.g. 0.12 for DMRD).
    n_bootstrap : int
        Bootstrap resamples for uncertainty estimation.
    rng : np.random.Generator, optional

    Returns
    -------
    dict with keys:
        ``hit_rate_{tol}`` and ``hit_rate_{tol}_se`` for each tolerance,
        ``final_al_static_r2``, ``final_al_mcmc_r2``,
        ``convergence_iter_{thresh}`` for each R² threshold.
    """
    rng = rng or np.random.default_rng(0)
    result: dict = {}

    # Hit rate (bootstrap over training samples)
    for tol in HIT_TOLERANCES:
        key = f"hit_rate_{int(tol * 100)}pct"
        val, se = _bootstrap_scalar(
            run.Y,
            lambda y, t=tol, tv=true_value: compute_hit_rate(y, tv, t),
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        result[key] = val
        result[f"{key}_se"] = se

    # Final static-random and MCMC R²
    result["final_al_static_r2"] = run.al_static_r2[-1] if run.al_static_r2 else float("nan")
    result["final_baseline_static_r2"] = run.baseline_static_r2[-1] if run.baseline_static_r2 else float("nan")
    result["final_al_mcmc_r2"] = run.al_mcmc_r2[-1] if run.al_mcmc_r2 else float("nan")
    result["final_baseline_mcmc_r2"] = run.baseline_mcmc_r2[-1] if run.baseline_mcmc_r2 else float("nan")
    result["final_al_val_r2"] = run.al_r2[-1] if run.al_r2 else float("nan")
    result["final_baseline_val_r2"] = run.baseline_r2[-1] if run.baseline_r2 else float("nan")

    # Convergence speed: first iteration where R² exceeds threshold.
    # Use static R² if available, fall back to validation R².
    r2_traj = run.al_static_r2 if run.al_static_r2 else run.al_r2
    for thresh in R2_THRESHOLDS:
        key = f"convergence_iter_r2_{int(thresh * 10)}"
        iters = [i + 1 for i, v in enumerate(r2_traj) if v is not None and v >= thresh]
        result[key] = int(iters[0]) if iters else float("nan")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Physical property metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_physical_props(
    run: RunData,
    true_value: float,
    n_bootstrap: int = 500,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute physical properties of the accumulated training dataset.

    Parameters
    ----------
    run : RunData
    true_value : float
        Target relic density (used only for labelling context; not subtracted here).
    n_bootstrap : int
    rng : np.random.Generator, optional

    Returns
    -------
    dict with keys:
        ``mean_relic_density``, ``mean_relic_density_se``,
        ``std_relic_density``,  ``std_relic_density_se``,
        ``param_variance_{name}``, ``param_variance_{name}_se`` for each free param,
        ``log_generalised_variance``, ``log_generalised_variance_se``.

    Notes
    -----
    *Log-generalised variance* is log det(Cov(X_free)) in normalised [0,1]
    space.  It is a scalar measure of the "volume" of parameter space covered
    by the training data.  Using the log avoids extreme values in 9-D space.
    A degenerate covariance matrix (fewer points than free dims) yields -inf.
    """
    rng = rng or np.random.default_rng(0)
    result: dict = {}

    # Mean and std of relic density
    result["mean_relic_density"], result["mean_relic_density_se"] = _bootstrap_scalar(
        run.Y, np.mean, n_bootstrap, rng
    )
    result["std_relic_density"], result["std_relic_density_se"] = _bootstrap_scalar(
        run.Y, np.std, n_bootstrap, rng
    )

    # Per-parameter variance in normalised space
    def _per_param_var(xfn: np.ndarray) -> np.ndarray:
        return np.var(xfn, axis=0, ddof=1)

    var_vals, var_ses = _bootstrap_vector(
        run.X_free_norm, _per_param_var, n_bootstrap, rng
    )
    for name, val, se in zip(FREE_PARAM_NAMES, var_vals, var_ses):
        result[f"param_variance_{name}"] = float(val)
        result[f"param_variance_{name}_se"] = float(se)

    # Log-generalised variance (log det of covariance matrix)
    def _log_gen_var(xfn: np.ndarray) -> float:
        if len(xfn) <= xfn.shape[1]:
            return -np.inf
        cov = np.cov(xfn.T)
        sign, logdet = np.linalg.slogdet(cov)
        return float(logdet) if sign > 0 else -np.inf

    result["log_generalised_variance"], result["log_generalised_variance_se"] = (
        _bootstrap_scalar(run.X_free_norm, _log_gen_var, n_bootstrap, rng)
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Diversity metrics
# ══════════════════════════════════════════════════════════════════════════════

N_ENTROPY_BINS: int = 20


def _entropy(x: np.ndarray, n_bins: int = N_ENTROPY_BINS) -> float:
    """Shannon entropy (bits) of ``x`` estimated via a histogram.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        1-D array of values in [0, 1].
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        Shannon entropy in bits.  Maximum value is log₂(n_bins).
    """
    counts, _ = np.histogram(x, bins=n_bins, range=(0.0, 1.0))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _per_param_entropy(X_free_norm: np.ndarray) -> np.ndarray:
    """Shannon entropy for each free parameter column.

    Parameters
    ----------
    X_free_norm : np.ndarray, shape (N, n_free)

    Returns
    -------
    np.ndarray, shape (n_free,)
    """
    return np.array([_entropy(X_free_norm[:, j]) for j in range(X_free_norm.shape[1])])


def _knn_mean_dist(X_free_norm: np.ndarray, k: int = 5) -> float:
    """Mean k-nearest-neighbour distance in normalised free-parameter space.

    For each point the mean distance to its ``k`` nearest neighbours is
    computed; the result is the average over all points.  Higher values
    indicate sparser coverage (points are spread further apart).

    Parameters
    ----------
    X_free_norm : np.ndarray, shape (N, n_free)
    k : int
        Number of nearest neighbours (excluding the point itself).

    Returns
    -------
    float
        Mean k-NN distance.
    """
    n = len(X_free_norm)
    k_eff = min(k + 1, n)  # +1 because the point itself is included
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(X_free_norm)
    dists, _ = nbrs.kneighbors(X_free_norm)
    # dists[:, 0] is self-distance (0); exclude it
    return float(dists[:, 1:].mean())


def _mmd_gaussian(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidth: float | None = None,
) -> float:
    """Maximum Mean Discrepancy with a Gaussian (RBF) kernel.

    MMD²(X, Y) = E[k(X, X)] + E[k(Y, Y)] − 2 E[k(X, Y)]
    where k(a, b) = exp(−‖a − b‖² / (2σ²)).

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    Y : np.ndarray, shape (m, d)
    bandwidth : float, optional
        Kernel bandwidth σ.  If ``None``, the median heuristic is used:
        σ = median of all pairwise distances in the pooled sample.

    Returns
    -------
    float
        MMD² estimate (can be slightly negative due to finite-sample noise).
    """
    if bandwidth is None:
        pooled = np.concatenate([X, Y], axis=0)
        # Subsample for speed when the pool is large
        if len(pooled) > 2000:
            idx = np.random.default_rng(0).choice(len(pooled), 2000, replace=False)
            pooled = pooled[idx]
        d = pairwise_distances(pooled, metric="euclidean")
        bandwidth = float(np.median(d[d > 0]))
        if bandwidth == 0:
            bandwidth = 1.0

    def _rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = pairwise_distances(A, B, metric="sqeuclidean")
        return np.exp(-d2 / (2.0 * bandwidth ** 2))

    return float(_rbf(X, X).mean() + _rbf(Y, Y).mean() - 2.0 * _rbf(X, Y).mean())


def compute_diversity_metrics(
    run: RunData,
    mcmc_X_free_norm: np.ndarray,
    kmeans: KMeans,
    mmd_bandwidth: float | None = None,
    n_bootstrap: int = 500,
    n_permutations: int = 200,
    knn_k: int = 5,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute diversity metrics for a single run.

    Parameters
    ----------
    run : RunData
    mcmc_X_free_norm : np.ndarray, shape (M, n_free)
        MCMC free parameters normalised to [0, 1].
    kmeans : sklearn.cluster.KMeans
        K-means model fitted on ``mcmc_X_free_norm``.  Using a shared model
        ensures consistent cluster definitions across all runs.
    mmd_bandwidth : float, optional
        MMD kernel bandwidth.  ``None`` → median heuristic (computed once
        from pooled AL + MCMC data).
    n_bootstrap : int
        Bootstrap resamples for entropy and k-NN uncertainty.
    n_permutations : int
        Permutation-test resamples for MMD uncertainty.
    knn_k : int
        Number of nearest neighbours for the k-NN distance metric.
    rng : np.random.Generator, optional

    Returns
    -------
    dict with keys:
        ``entropy_{param}``, ``entropy_{param}_se`` for each free param,
        ``mean_entropy``, ``mean_entropy_se``,
        ``knn_mean_dist``, ``knn_mean_dist_se``,
        ``mmd_vs_mcmc``, ``mmd_vs_mcmc_se``, ``mmd_pvalue``,
        ``cluster_coverage``, ``cluster_coverage_se``.

    Notes
    -----
    *k-NN distance* is O(N · k) after fitting the index.  *MMD* is O(n · m)
    for the cross-term; both AL and MCMC samples are capped at 5000 points
    for the permutation test to remain tractable.
    """
    rng = rng or np.random.default_rng(0)
    result: dict = {}
    X = run.X_free_norm

    # ── per-parameter entropy ──────────────────────────────────────────────────
    ent_vals, ent_ses = _bootstrap_vector(X, _per_param_entropy, n_bootstrap, rng)
    for name, val, se in zip(FREE_PARAM_NAMES, ent_vals, ent_ses):
        result[f"entropy_{name}"] = float(val)
        result[f"entropy_{name}_se"] = float(se)
    result["mean_entropy"] = float(ent_vals.mean())
    result["mean_entropy_se"] = float(np.sqrt((ent_ses ** 2).mean()))  # propagate errors

    # ── k-NN mean distance ─────────────────────────────────────────────────────
    result["knn_mean_dist"], result["knn_mean_dist_se"] = _bootstrap_scalar(
        X,
        lambda xfn: _knn_mean_dist(xfn, k=knn_k),
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    # ── MMD vs MCMC ───────────────────────────────────────────────────────────
    # Subsample both to 5000 pts so the permutation test stays fast
    max_pts = 5000
    X_sub = X[rng.choice(len(X), min(max_pts, len(X)), replace=False)]
    M_sub = mcmc_X_free_norm[rng.choice(len(mcmc_X_free_norm), min(max_pts, len(mcmc_X_free_norm)), replace=False)]

    observed_mmd = _mmd_gaussian(X_sub, M_sub, bandwidth=mmd_bandwidth)
    result["mmd_vs_mcmc"] = observed_mmd

    if n_permutations > 0:
        pooled = np.concatenate([X_sub, M_sub], axis=0)
        n_x = len(X_sub)
        perm_mmds = []
        for _ in range(n_permutations):
            shuffled = rng.permutation(pooled)
            perm_mmds.append(_mmd_gaussian(shuffled[:n_x], shuffled[n_x:], bandwidth=mmd_bandwidth))
        perm_mmds = np.array(perm_mmds)
        result["mmd_vs_mcmc_se"] = float(np.std(perm_mmds, ddof=1))
        # p-value: fraction of permuted MMDs ≥ observed
        result["mmd_pvalue"] = float(np.mean(perm_mmds >= observed_mmd))
    else:
        result["mmd_vs_mcmc_se"] = float("nan")
        result["mmd_pvalue"] = float("nan")

    # ── MCMC cluster coverage ─────────────────────────────────────────────────
    # Assign each training point to its nearest MCMC cluster centre.
    al_labels = kmeans.predict(X)
    n_clusters = kmeans.n_clusters
    n_covered = int(len(np.unique(al_labels)))
    coverage = n_covered / n_clusters
    result["cluster_coverage"] = coverage
    # Binomial confidence interval (Wilson interval)
    if n_clusters > 0:
        ci_lo, ci_hi = _wilson_ci(n_covered, n_clusters)
        result["cluster_coverage_se"] = float((ci_hi - ci_lo) / 2)
    else:
        result["cluster_coverage_se"] = float("nan")

    return result


def _wilson_ci(k: int, n: int, confidence: float = 0.68) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    confidence : float
        Confidence level (default 0.68 ≈ ±1σ).

    Returns
    -------
    (lower, upper) : tuple[float, float]
    """
    z = float(scipy_norm.ppf(0.5 + confidence / 2))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


# ══════════════════════════════════════════════════════════════════════════════
# Per-iteration hit rate (trajectory)
# ══════════════════════════════════════════════════════════════════════════════

def compute_hit_rate_trajectory(
    run: RunData,
    true_value: float,
    tol: float,
) -> tuple[list[int], list[float]]:
    """Compute hit rate at each iteration using the cumulative training dataset.

    The training data ``(X, Y)`` in ``state.pt`` is accumulated sequentially.
    The list ``al_n_train`` records the cumulative training-set size after each
    iteration, allowing us to reconstruct per-iteration slices of Y.

    Parameters
    ----------
    run : RunData
    true_value : float
    tol : float
        Relative tolerance.

    Returns
    -------
    iters : list[int]
        Iteration indices (1-based).
    rates : list[float]
        Hit rate at each iteration.
    """
    iters, rates = [], []
    for i, n in enumerate(run.n_train_per_iter):
        if n is None or n <= 0:
            continue
        n_clip = min(int(n), len(run.Y))
        rate = compute_hit_rate(run.Y[:n_clip], true_value, tol)
        iters.append(i + 1)
        rates.append(rate)
    return iters, rates


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_r2_trajectories(
    runs: list[RunData],
    out_dir: Path,
    include_baseline: bool = False,
) -> None:
    """Line charts of static-random and MCMC eval R² vs iteration for all runs.

    Parameters
    ----------
    runs : list[RunData]
    out_dir : Path
    include_baseline : bool
        Also draw baseline curves alongside AL curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    axes[0].set_title("Static-random eval R²")
    axes[1].set_title("MCMC eval R²")

    for ax, (al_key, base_key, ylabel) in zip(
        axes,
        [
            ("al_static_r2", "baseline_static_r2", "R² (static random)"),
            ("al_mcmc_r2",   "baseline_mcmc_r2",   "R² (MCMC)"),
        ],
    ):
        for idx, run in enumerate(runs):
            color = PALETTE[idx % len(PALETTE)]
            al_traj = getattr(run, al_key)
            if not al_traj:
                continue
            iters = list(range(1, len(al_traj) + 1))
            ax.plot(iters, al_traj, color=color, lw=2, label=run.label)
            if include_baseline:
                base_traj = getattr(run, base_key)
                if base_traj:
                    ax.plot(
                        range(1, len(base_traj) + 1),
                        base_traj,
                        color=color,
                        lw=1.5,
                        ls="--",
                        alpha=0.6,
                        label=f"{run.label} (baseline)",
                    )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5, ls=":")

    fig.tight_layout()
    _save(fig, out_dir / "r2_trajectories.png")


def plot_hit_rate_trajectories(
    runs: list[RunData],
    true_value: float,
    out_dir: Path,
    tolerances: list[float] | None = None,
) -> None:
    """Line charts of cumulative hit rate vs iteration for all runs.

    Parameters
    ----------
    runs : list[RunData]
    true_value : float
    out_dir : Path
    tolerances : list[float], optional
        Relative tolerances to plot.  Defaults to HIT_TOLERANCES.
    """
    tolerances = tolerances or HIT_TOLERANCES
    fig, axes = plt.subplots(1, len(tolerances), figsize=(6 * len(tolerances), 5), sharey=False)
    if len(tolerances) == 1:
        axes = [axes]

    for ax, tol in zip(axes, tolerances):
        ax.set_title(f"Hit rate (|Ω − {true_value}| / {true_value} < {int(tol*100)} %)")
        for idx, run in enumerate(runs):
            color = PALETTE[idx % len(PALETTE)]
            iters, rates = compute_hit_rate_trajectory(run, true_value, tol)
            if iters:
                ax.plot(iters, rates, color=color, lw=2, marker="o", ms=4, label=run.label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Hit rate")
        ax.set_ylim(0, 0.4)
        ax.grid(True, alpha=0.3)

    # Single shared legend outside the rightmost axis
    handles, lbls = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, fontsize=8, loc="center left",
                   bbox_to_anchor=(1.0, 0.5), borderaxespad=0.)

    fig.tight_layout(rect=(0, 0, 0.88, 1))
    _save(fig, out_dir / "hit_rate_trajectories.png")


def plot_quality_summary(
    runs: list[RunData],
    quality_metrics: list[dict],
    out_dir: Path,
) -> None:
    """Bar chart of final quality metrics across all runs.

    Shows hit rates at each tolerance and final R² (static-random and MCMC).

    Parameters
    ----------
    runs : list[RunData]
    quality_metrics : list[dict]
        One dict per run (output of ``compute_quality_metrics``).
    out_dir : Path
    """
    labels = [r.label for r in runs]
    n_runs = len(runs)
    x = np.arange(n_runs)
    w = 0.18  # bar width

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── hit rates ──────────────────────────────────────────────────────────────
    ax = axes[0]
    offsets = np.linspace(-w, w, len(HIT_TOLERANCES))
    for k, (tol, off) in enumerate(zip(HIT_TOLERANCES, offsets)):
        key = f"hit_rate_{int(tol * 100)}pct"
        vals = [m[key] for m in quality_metrics]
        ses  = [m[f"{key}_se"] for m in quality_metrics]
        ax.bar(x + off, vals, width=w, yerr=ses, capsize=4,
               label=f"±{int(tol*100)}%", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, 0.4)
    ax.set_title("Hit rate at different tolerances (final)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── final R² ──────────────────────────────────────────────────────────────
    ax = axes[1]
    r2_keys = [
        ("final_al_static_r2",    "AL static-random R²"),
        ("final_al_mcmc_r2",      "AL MCMC R²"),
        ("final_al_val_r2",       "AL val R²"),
    ]
    offsets = np.linspace(-w, w, len(r2_keys))
    r2_clip = -10.0
    for (key, lbl), off in zip(r2_keys, offsets):
        vals = [m[key] for m in quality_metrics]
        clipped = [max(v, r2_clip) for v in vals]
        bars = ax.bar(x + off, clipped, width=w, label=lbl, alpha=0.8)
        # Annotate bars whose true value was clipped so no information is lost.
        for xi, v_true, v_clip in zip(x + off, vals, clipped):
            if v_true < r2_clip:
                ax.text(xi, v_clip + 0.1, f"{v_true:.0f}",
                        ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_ylim(r2_clip, 1.0)
    ax.set_title("Final R² scores (clipped at R²=−10)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)

    fig.tight_layout()
    _save(fig, out_dir / "quality_summary.png")


def plot_relic_density_summary(
    runs: list[RunData],
    physical_metrics: list[dict],
    true_value: float,
    out_dir: Path,
) -> None:
    """Bar chart of mean ± std relic density per run.

    Parameters
    ----------
    runs : list[RunData]
    physical_metrics : list[dict]
    true_value : float
    out_dir : Path
    """
    labels = [r.label for r in runs]
    x = np.arange(len(runs))
    means = [m["mean_relic_density"] for m in physical_metrics]
    stds  = [m["std_relic_density"]  for m in physical_metrics]
    mean_ses = [m["mean_relic_density_se"] for m in physical_metrics]

    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 1.2), 5))
    bars = ax.bar(x, means, yerr=mean_ses, capsize=5, color=PALETTE[:len(runs)], alpha=0.8)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="k", capsize=6, lw=1.5,
                label="±1 std (spread of Ω)")
    ax.axhline(true_value, color="red", lw=1.5, ls="--", label=f"Target Ω = {true_value}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Relic density Ω")
    ax.set_title("Mean (±SE bars) and spread (±1 std whiskers) of Ω in training data")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "relic_density_summary.png")


def plot_diversity_summary(
    runs: list[RunData],
    diversity_metrics: list[dict],
    out_dir: Path,
) -> None:
    """Bar charts of scalar diversity metrics across all runs.

    Parameters
    ----------
    runs : list[RunData]
    diversity_metrics : list[dict]
    out_dir : Path
    """
    labels = [r.label for r in runs]
    x = np.arange(len(runs))

    scalar_metrics = [
        ("mean_entropy",     "Mean per-param entropy (bits)",   f"max = {np.log2(N_ENTROPY_BINS):.2f} bits"),
        ("knn_mean_dist",    "Mean k-NN distance (normalised)", None),
        ("mmd_vs_mcmc",      "MMD² vs MCMC",                    "lower = more MCMC-like"),
        ("cluster_coverage", "MCMC cluster coverage",           "fraction of clusters hit"),
    ]

    fig, axes = plt.subplots(1, len(scalar_metrics), figsize=(5 * len(scalar_metrics), 5))

    for ax, (key, ylabel, note) in zip(axes, scalar_metrics):
        vals = [m.get(key, float("nan")) for m in diversity_metrics]
        ses  = [m.get(f"{key}_se", float("nan")) for m in diversity_metrics]
        ax.bar(x, vals, yerr=ses, capsize=5, color=PALETTE[:len(runs)], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if note:
            ax.set_xlabel(note, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir / "diversity_summary.png")


def plot_param_entropy_heatmap(
    runs: list[RunData],
    diversity_metrics: list[dict],
    out_dir: Path,
) -> None:
    """Heatmap of per-parameter entropy for each run.

    Rows = runs, columns = free parameters.  Colour encodes entropy (bits).

    Parameters
    ----------
    runs : list[RunData]
    diversity_metrics : list[dict]
    out_dir : Path
    """
    labels = [r.label for r in runs]
    data = np.array([
        [m.get(f"entropy_{p}", float("nan")) for p in FREE_PARAM_NAMES]
        for m in diversity_metrics
    ])
    max_ent = np.log2(N_ENTROPY_BINS)

    fig, ax = plt.subplots(figsize=(max(10, len(FREE_PARAM_NAMES) * 0.9), max(4, len(runs) * 0.7)))
    im = ax.imshow(data, vmin=0, vmax=max_ent, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(FREE_PARAM_NAMES)))
    ax.set_xticklabels(FREE_PARAM_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(f"Per-parameter Shannon entropy (bits, max = {max_ent:.2f})")
    plt.colorbar(im, ax=ax, label="Entropy (bits)")

    # Annotate cells
    for i in range(len(runs)):
        for j in range(len(FREE_PARAM_NAMES)):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > max_ent * 0.6 else "black")

    fig.tight_layout()
    _save(fig, out_dir / "param_entropy_heatmap.png")


def plot_param_variance_heatmap(
    runs: list[RunData],
    physical_metrics: list[dict],
    out_dir: Path,
) -> None:
    """Heatmap of per-parameter sample variance for each run.

    Parameters
    ----------
    runs : list[RunData]
    physical_metrics : list[dict]
    out_dir : Path
    """
    labels = [r.label for r in runs]
    data = np.array([
        [m.get(f"param_variance_{p}", float("nan")) for p in FREE_PARAM_NAMES]
        for m in physical_metrics
    ])

    fig, ax = plt.subplots(figsize=(max(10, len(FREE_PARAM_NAMES) * 0.9), max(4, len(runs) * 0.7)))
    im = ax.imshow(data, vmin=0, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(FREE_PARAM_NAMES)))
    ax.set_xticklabels(FREE_PARAM_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Per-parameter sample variance (normalised [0,1] space)")
    plt.colorbar(im, ax=ax, label="Variance")

    for i in range(len(runs)):
        for j in range(len(FREE_PARAM_NAMES)):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    _save(fig, out_dir / "param_variance_heatmap.png")


def plot_pairwise_scatter(
    runs: list[RunData],
    mcmc_X_free_norm: np.ndarray,
    out_dir: Path,
    max_pts: int = 2000,
) -> None:
    """2-D scatter plots of selected training points overlaid with MCMC samples.

    One panel per parameter pair defined in ``SCATTER_PAIRS``.  MCMC samples
    are shown as a grey background density; each run's training data is
    overlaid in colour.

    Parameters
    ----------
    runs : list[RunData]
    mcmc_X_free_norm : np.ndarray, shape (M, n_free)
    out_dir : Path
    max_pts : int
        Cap on the number of points plotted per run to avoid overplotting.
    """
    # Build index: free param name → column index in X_free_norm
    name_to_col = {n: i for i, n in enumerate(FREE_PARAM_NAMES)}
    valid_pairs = [(a, b) for a, b in SCATTER_PAIRS if a in name_to_col and b in name_to_col]
    if not valid_pairs:
        return

    ncols = min(3, len(valid_pairs))
    nrows = (len(valid_pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).ravel()

    rng = np.random.default_rng(42)

    for ax, (xname, yname) in zip(axes_flat, valid_pairs):
        xi = name_to_col[xname]
        yi = name_to_col[yname]

        # MCMC background
        m_sub = mcmc_X_free_norm[:, [xi, yi]]
        if len(m_sub) > 5000:
            m_sub = m_sub[rng.choice(len(m_sub), 5000, replace=False)]
        ax.scatter(m_sub[:, 0], m_sub[:, 1], c="lightgrey", s=2, alpha=0.4,
                   label="MCMC", zorder=1, rasterized=True)

        # AL training data per run
        for idx, run in enumerate(runs):
            pts = run.X_free_norm[:, [xi, yi]]
            if len(pts) > max_pts:
                pts = pts[rng.choice(len(pts), max_pts, replace=False)]
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=[PALETTE[idx % len(PALETTE)]],
                       s=10, alpha=0.6, label=run.label, zorder=2 + idx,
                       rasterized=True)

        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)

    # Shared legend on first axis
    handles, lbls = axes_flat[0].get_legend_handles_labels()
    axes_flat[0].legend(handles, lbls, fontsize=7, loc="upper right")

    # Hide unused panels
    for ax in axes_flat[len(valid_pairs):]:
        ax.set_visible(False)

    fig.suptitle("Training data vs MCMC posterior (normalised parameter space)", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "pairwise_scatter.png")


def plot_pairwise_scatter_per_run(
    runs: list[RunData],
    mcmc_X_free_norm: np.ndarray,
    out_dir: Path,
    max_pts: int = 5000,
) -> None:
    """Same panels as :func:`plot_pairwise_scatter`, but one figure per run.

    Avoids the overplotting mess that results from stacking all runs on the
    same axes. MCMC samples remain the grey background; each figure shows
    exactly one run's training data on top.
    """
    name_to_col = {n: i for i, n in enumerate(FREE_PARAM_NAMES)}
    valid_pairs = [(a, b) for a, b in SCATTER_PAIRS if a in name_to_col and b in name_to_col]
    if not valid_pairs:
        return

    sub_dir = out_dir / "pairwise_scatter_per_run"
    sub_dir.mkdir(parents=True, exist_ok=True)

    ncols = min(3, len(valid_pairs))
    nrows = (len(valid_pairs) + ncols - 1) // ncols
    rng = np.random.default_rng(42)

    # Pre-subsample MCMC once
    mcmc_plot = mcmc_X_free_norm
    if len(mcmc_plot) > 5000:
        mcmc_plot = mcmc_plot[rng.choice(len(mcmc_plot), 5000, replace=False)]

    for idx, run in enumerate(runs):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = np.array(axes).ravel()
        colour = PALETTE[idx % len(PALETTE)]

        for ax, (xname, yname) in zip(axes_flat, valid_pairs):
            xi = name_to_col[xname]
            yi = name_to_col[yname]

            if len(mcmc_plot):
                ax.scatter(mcmc_plot[:, xi], mcmc_plot[:, yi],
                           c="lightgrey", s=2, alpha=0.4, label="MCMC",
                           zorder=1, rasterized=True)

            pts = run.X_free_norm[:, [xi, yi]]
            if len(pts) > max_pts:
                pts = pts[rng.choice(len(pts), max_pts, replace=False)]
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=[colour], s=10, alpha=0.6, label=run.label,
                       zorder=2, rasterized=True)

            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.2)

        handles, lbls = axes_flat[0].get_legend_handles_labels()
        axes_flat[0].legend(handles, lbls, fontsize=9, loc="upper right")

        for ax in axes_flat[len(valid_pairs):]:
            ax.set_visible(False)

        fig.suptitle(
            f"Training data vs MCMC posterior — {run.label} "
            f"(N={len(run.X_free_norm)})",
            y=1.01,
        )
        fig.tight_layout()
        safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in run.label)
        _save(fig, sub_dir / f"pairwise_scatter_{safe_label}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_summary_csv(
    runs: list[RunData],
    quality_metrics: list[dict],
    physical_metrics: list[dict],
    diversity_metrics: list[dict],
    out_dir: Path,
) -> None:
    """Write ``summary.csv`` with one row per run and all scalar metrics.

    Parameters
    ----------
    runs : list[RunData]
    quality_metrics, physical_metrics, diversity_metrics : list[dict]
        One dict per run.
    out_dir : Path
    """
    rows = []
    for run, qm, pm, dm in zip(runs, quality_metrics, physical_metrics, diversity_metrics):
        row = {
            "label": run.label,
            "run_dir": str(run.run_dir),
            "n_iters": run.n_iters,
            "n_train_final": int(len(run.Y)),
        }
        row.update(qm)
        row.update(pm)
        row.update(dm)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dirs", nargs="+", required=True,
        help="One or more active learning output directories (each must contain state.pt).",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Human-readable labels for each run (same order as --run-dirs). "
             "Derived from directory names if omitted.",
    )
    parser.add_argument(
        "--mcmc-data-dir", default=None,
        help="Directory with MCMC ROOT files for MMD and cluster-coverage metrics. "
             "If omitted, MMD and cluster-coverage are skipped.",
    )
    parser.add_argument(
        "--output-dir", default="analysis_output",
        help="Directory where plots and summary.csv are written (default: analysis_output).",
    )
    parser.add_argument(
        "--target", default="DMRD", choices=list(TARGET_CONFIG.keys()),
        help="Target variable (default: DMRD).",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=20,
        help="Number of K-means clusters for MCMC cluster-coverage metric (default: 20).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=500,
        help="Bootstrap resamples for uncertainty estimation (0 = skip, default: 500).",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=200,
        help="Permutation-test resamples for MMD uncertainty (default: 200).",
    )
    parser.add_argument(
        "--knn-k", type=int, default=5,
        help="Number of nearest neighbours for k-NN distance metric (default: 5).",
    )
    parser.add_argument(
        "--include-baseline", action="store_true",
        help="Also draw baseline model curves in trajectory plots.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for bootstrap and permutation tests (default: 0).",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    true_value = TARGET_CONFIG[args.target]["true_value"]

    # ── load runs ──────────────────────────────────────────────────────────────
    labels = args.labels or [None] * len(args.run_dirs)
    if len(labels) != len(args.run_dirs):
        raise ValueError(
            f"--labels has {len(labels)} entries but --run-dirs has {len(args.run_dirs)}"
        )

    runs: list[RunData] = []
    for rd, lbl in zip(args.run_dirs, labels):
        print(f"Loading {rd} ...")
        try:
            run = load_run(rd, label=lbl)
            print(f"  label={run.label!r}  n_iters={run.n_iters}  n_train={len(run.Y)}")
            runs.append(run)
        except Exception as exc:
            warnings.warn(f"  SKIPPED {rd}: {exc}")

    if not runs:
        print("No runs loaded — exiting.")
        return

    # ── load MCMC data (optional) ─────────────────────────────────────────────
    mcmc_X_free_norm: np.ndarray | None = None
    kmeans: KMeans | None = None
    mmd_bandwidth: float | None = None

    if args.mcmc_data_dir is not None:
        print(f"\nLoading MCMC data from {args.mcmc_data_dir} ...")
        mcmc_X, mcmc_Y = load_mcmc_numpy(args.mcmc_data_dir)
        mcmc_X_free_norm = normalise_free_params(mcmc_X)
        print(f"  {len(mcmc_X_free_norm)} MCMC samples loaded")

        print(f"  Fitting K-means (k={args.n_clusters}) on MCMC free params ...")
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=0, n_init="auto")
        kmeans.fit(mcmc_X_free_norm)

        # Compute a shared MMD bandwidth from pooled AL + MCMC data
        all_X = np.concatenate([r.X_free_norm for r in runs] + [mcmc_X_free_norm])
        if len(all_X) > 3000:
            all_X = all_X[rng.choice(len(all_X), 3000, replace=False)]
        d = pairwise_distances(all_X, metric="euclidean")
        mmd_bandwidth = float(np.median(d[d > 0]))
        print(f"  MMD bandwidth (median heuristic): {mmd_bandwidth:.4f}")

    # ── compute metrics ────────────────────────────────────────────────────────
    quality_metrics, physical_metrics, diversity_metrics = [], [], []

    for run in runs:
        print(f"\nComputing metrics for {run.label!r} ...")

        q = compute_quality_metrics(run, true_value, n_bootstrap=args.n_bootstrap, rng=rng)
        quality_metrics.append(q)
        print(
            f"  hit@10%={q['hit_rate_10pct']:.3f}±{q['hit_rate_10pct_se']:.3f}  "
            f"hit@20%={q['hit_rate_20pct']:.3f}±{q['hit_rate_20pct_se']:.3f}  "
            f"static_r2={q['final_al_static_r2']:.3f}  mcmc_r2={q['final_al_mcmc_r2']:.3f}"
        )

        p = compute_physical_props(run, true_value, n_bootstrap=args.n_bootstrap, rng=rng)
        physical_metrics.append(p)
        print(
            f"  mean_Ω={p['mean_relic_density']:.4f}±{p['mean_relic_density_se']:.4f}  "
            f"std_Ω={p['std_relic_density']:.4f}  "
            f"log_genvar={p['log_generalised_variance']:.2f}"
        )

        d: dict = {}
        if mcmc_X_free_norm is not None and kmeans is not None:
            d = compute_diversity_metrics(
                run,
                mcmc_X_free_norm=mcmc_X_free_norm,
                kmeans=kmeans,
                mmd_bandwidth=mmd_bandwidth,
                n_bootstrap=args.n_bootstrap,
                n_permutations=args.n_permutations,
                knn_k=args.knn_k,
                rng=rng,
            )
        else:
            # Compute entropy and k-NN without MCMC
            ent_vals, ent_ses = _bootstrap_vector(
                run.X_free_norm, _per_param_entropy, args.n_bootstrap, rng
            )
            for name, val, se in zip(FREE_PARAM_NAMES, ent_vals, ent_ses):
                d[f"entropy_{name}"] = float(val)
                d[f"entropy_{name}_se"] = float(se)
            d["mean_entropy"] = float(ent_vals.mean())
            d["mean_entropy_se"] = float(np.sqrt((ent_ses ** 2).mean()))
            d["knn_mean_dist"], d["knn_mean_dist_se"] = _bootstrap_scalar(
                run.X_free_norm,
                lambda xfn: _knn_mean_dist(xfn, k=args.knn_k),
                args.n_bootstrap,
                rng,
            )
            for k_mmd in ("mmd_vs_mcmc", "mmd_vs_mcmc_se", "mmd_pvalue",
                          "cluster_coverage", "cluster_coverage_se"):
                d[k_mmd] = float("nan")

        diversity_metrics.append(d)
        print(
            f"  mean_entropy={d.get('mean_entropy', float('nan')):.3f}  "
            f"knn_dist={d.get('knn_mean_dist', float('nan')):.4f}  "
            f"mmd={d.get('mmd_vs_mcmc', float('nan')):.5f}  "
            f"coverage={d.get('cluster_coverage', float('nan')):.3f}"
        )

    # ── plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_r2_trajectories(runs, out_dir, include_baseline=args.include_baseline)
    plot_hit_rate_trajectories(runs, true_value, out_dir)
    plot_quality_summary(runs, quality_metrics, out_dir)
    plot_relic_density_summary(runs, physical_metrics, true_value, out_dir)
    plot_diversity_summary(runs, diversity_metrics, out_dir)
    plot_param_entropy_heatmap(runs, diversity_metrics, out_dir)
    plot_param_variance_heatmap(runs, physical_metrics, out_dir)
    if mcmc_X_free_norm is not None:
        plot_pairwise_scatter(runs, mcmc_X_free_norm, out_dir)
        plot_pairwise_scatter_per_run(runs, mcmc_X_free_norm, out_dir)
    else:
        # Still plot scatter without MCMC background
        dummy_mcmc = np.empty((0, len(FREE_PARAM_INDICES)))
        plot_pairwise_scatter(runs, dummy_mcmc, out_dir)
        plot_pairwise_scatter_per_run(runs, dummy_mcmc, out_dir)

    # ── summary CSV ────────────────────────────────────────────────────────────
    save_summary_csv(runs, quality_metrics, physical_metrics, diversity_metrics, out_dir)
    print(f"\nDone. Results written to {out_dir}/")


if __name__ == "__main__":
    main()
