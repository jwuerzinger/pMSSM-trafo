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
import re
import sys
import time
from pathlib import Path

# Force line-buffered stdout/stderr so SLURM log files reflect progress in real
# time even when PYTHONUNBUFFERED is missed (e.g. inside subprocesses).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

import click
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

# Make the repo root importable so we can reuse analyse_runs utilities.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analyse_runs import compute_hit_rate_trajectory, load_run, filter_run_neutralino_lsp  # noqa: E402


# Module-level toggle for the post-hoc neutralino-LSP veto. Set from main().
_REQUIRE_NEUTRALINO_LSP = False


def _load_run(run_dir):
    """load_run + optional sneutrino veto controlled by main()'s CLI flag."""
    run = load_run(run_dir)
    if _REQUIRE_NEUTRALINO_LSP:
        run = filter_run_neutralino_lsp(run)
    return run
from pmssm import TARGET_CONFIG  # noqa: E402

# Defaults if a run's state.pt is missing the relevant fields.
_N_SAMPLES_DEFAULT = 2000
_N_SELECT_DEFAULT = 500

# Run3ModelGen validity rate, set by main() after parsing the AL log.
# When set, _desired_per_iter divides the initial-set count by this rate so the
# initial chunk is charged in attempt-units (consistent with the per-iter
# `n_select` increments which are already in attempt-units). Without this the
# initial 2000 valid samples are a free pass that biases iter-1 hits/desired
# upward by ~1/p_valid and only fully dilutes by iter ~40.
_DESIRED_P_VALID: float | None = None


MODEL_COLORS = {
    "transformer": "tab:blue",
    "exact_gp":    "tab:orange",
    "deep_gp":     "tab:green",
    "tabpfn":      "tab:red",
    "dnn":         "tab:purple",
    "dnn_match_trafo": "tab:pink",
    # Oracle (theoretical-limit) variants — same colour as parent so curves
    # group visually; linestyle disambiguates (see ORACLE_LS below).
    "transformer_oracle": "tab:blue",
    "deep_gp_oracle":     "tab:green",
}
# Linestyle override for *_oracle model rows (dotted, regardless of warm/cold).
ORACLE_LS = ":"
# Friendly display labels used in legends.
MODEL_DISPLAY = {
    "transformer":         "Transformer",
    "exact_gp":            "Exact GP",
    "deep_gp":             "Deep GP",
    "tabpfn":              "TabPFN",
    "dnn":                 "DNN",
    "dnn_match_trafo":     "DNN (matched)",
    "transformer_oracle":  "Transformer (oracle)",
    "deep_gp_oracle":      "Deep GP (oracle)",
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
    cum = n_samples / _DESIRED_P_VALID if _DESIRED_P_VALID is not None else float(n_samples)
    for i in range(len(al_n_train)):
        desired.append(cum)
        n_sel = n_select_per_iter[i] if i < len(n_select_per_iter) else _N_SELECT_DEFAULT
        cum += n_sel
    return desired


def _hits_per_desired_trajectory(run, true_value, tol):
    """Per-iteration cumulative hit count divided by cumulative requested count.

    Numerator at iter (i+1) = #{ Y[:al_n_train[i]] within `tol` of `true_value` }
    Denominator at iter (i+1) = `_desired_per_iter(run)[i]`

    Iter-1 override: at i=0 the surrogate has not yet selected anything, so the
    initial training set is i.i.d. with the random-scan baseline. To keep the
    comparison apples-to-apples there, the iter-1 denominator drops the val
    chunk and uses `al_n_train[0] / p_valid` only — the same train-only
    denominator the random-baseline trajectory uses (which is then scaled by
    p_valid in main()). Without this override, AL iter-1 sits ~20% below the
    random line purely because val samples cost budget without contributing to
    the numerator. From iter 2 on the full denominator (initial in
    attempt-units + cumulative attempt-unit selections) is used as before.

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
        if i == 0 and _DESIRED_P_VALID is not None:
            denom = n_clip / _DESIRED_P_VALID
        else:
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


def _load_xy_full(data_dir: str, target: str, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (or read from .npy cache) the full unshuffled (X, Y) pool used by AL runs.

    Y reuses `_load_y_full`'s cache. X is cached separately as a (N, 19) float32
    array and read with mmap on hit, so the 11 GB ROOT pool doesn't bloat
    resident memory if multiple workers reuse it.
    """
    Y_full = _load_y_full(data_dir, target, cache_dir)
    safe = data_dir.replace("/", "_").strip("_")
    x_cache = cache_dir / f"x_full_{safe}_{target}.npy"
    if x_cache.exists():
        X_full = np.load(x_cache, mmap_mode="r")
        return X_full, Y_full
    from pmssm.data import load_pmssm_data  # noqa: PLC0415
    X, _Y = load_pmssm_data(n_datasets=-1, target=target, data_dir=data_dir,
                             plot_dir=str(cache_dir))
    X = X.numpy().astype(np.float32) if hasattr(X, "numpy") else np.asarray(X, dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_cache, X)
    X_full = np.load(x_cache, mmap_mode="r")
    return X_full, Y_full


def _seed_shuffled_y(Y_full: np.ndarray, seed: int) -> np.ndarray:
    """Replay the AL driver's per-seed `_load_perm` shuffle on Y_full."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(Y_full), generator=g).numpy()
    return Y_full[perm]


def _seed_perm(n: int, seed: int) -> np.ndarray:
    """Permutation indices for a given seed (matches AL driver's `_load_perm`)."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(int(n), generator=g).numpy()


def _static_random_indices(n_pool: int, n_initial_reserved: int,
                           static_eval_size: int = 100_000) -> np.ndarray:
    """Reproduce the static-random eval indices into the unshuffled X_full pool.

    Mirrors active_learning.py:474-482: a torch.Generator with manual_seed(123)
    permutes the post-reserved tail of the pool, and the first
    `static_eval_size` of those become the static eval set.

    Returns indices into the unshuffled pool (0-based).
    """
    available = n_pool - n_initial_reserved
    actual = min(int(static_eval_size), max(0, available))
    if actual <= 0:
        return np.empty(0, dtype=np.int64)
    g_static = torch.Generator().manual_seed(123)
    perm_static = torch.randperm(int(available), generator=g_static).numpy()
    return perm_static[:actual] + int(n_initial_reserved)


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


def _baseline_iter_xy(state: dict, X_full_shuffled: np.ndarray, Y_full_shuffled: np.ndarray,
                      iter_idx: int, role: str = "train"
                      ) -> tuple[np.ndarray, np.ndarray] | None:
    """Reconstruct the baseline cumulative (X, Y) at iteration `iter_idx` for `role`.

    `role` is "train" or "val". Layout of `baseline_add_indices` is
    [train-additions ..., val-additions ...] (see active_learning.py:691-693);
    state["X"], state["Y"] hold AL train data only — for the baseline we re-use
    the same INITIAL train pool (the al_n_train[0] / al_n_val[0] heads of
    state.X / state.X_val are also baseline init since both drivers share the
    same first split), then append the seed-shuffled `baseline_add_indices`
    slices for the requested role.
    """
    al_n_train_list = list(state.get("al_n_train") or [])
    al_n_val_list = list(state.get("al_n_val") or [])
    base_n_train = list(state.get("baseline_n_train") or [])
    base_n_val = list(state.get("baseline_n_val") or [])
    if not al_n_train_list:
        return None

    n_train_init = int(al_n_train_list[0])
    n_val_init = int(al_n_val_list[0]) if al_n_val_list else 0

    if role == "train":
        if iter_idx >= len(base_n_train):
            return None
        n_total = int(base_n_train[iter_idx])
        n_init = n_train_init
        X_init_src, Y_init_src = state.get("X"), state.get("Y")
        add_offset = 0  # baseline_add_indices[0:n_added_train]
    elif role == "val":
        if iter_idx >= len(base_n_val):
            return None
        n_total = int(base_n_val[iter_idx])
        n_init = n_val_init
        X_init_src, Y_init_src = state.get("X_val"), state.get("Y_val")
        # baseline_add_indices layout: train-adds prefix, then val-adds. Skip
        # past every train-add in the indices array (= the *cumulative* number
        # of train-adds across all iterations, which at iter `iter_idx` is
        # base_n_train[iter_idx] - n_train_init).
        n_added_train_total = max(0, int(base_n_train[iter_idx]) - n_train_init) \
            if base_n_train and iter_idx < len(base_n_train) else 0
        add_offset = n_added_train_total
    else:
        raise ValueError(f"role must be 'train' or 'val', got {role}")

    if n_total <= 0 or X_init_src is None or Y_init_src is None:
        return None

    X_init = X_init_src.numpy() if hasattr(X_init_src, "numpy") else np.asarray(X_init_src)
    Y_init = Y_init_src.numpy().ravel() if hasattr(Y_init_src, "numpy") else np.asarray(Y_init_src).ravel()
    X_init = np.asarray(X_init[:n_init], dtype=np.float64)
    Y_init = Y_init[:n_init]

    n_added = max(0, n_total - n_init)
    if n_added == 0:
        return X_init, Y_init

    add_idx = state.get("baseline_add_indices")
    add_idx = add_idx.numpy() if hasattr(add_idx, "numpy") else np.asarray(add_idx)
    if len(add_idx) < add_offset + n_added:
        return None
    sel = add_idx[add_offset:add_offset + n_added]
    X_added = np.asarray(X_full_shuffled[sel], dtype=np.float64)
    Y_added = np.asarray(Y_full_shuffled[sel], dtype=np.float64).ravel()
    return np.concatenate([X_init, X_added], axis=0), np.concatenate([Y_init, Y_added], axis=0)


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

    Returns the *per-valid-sample* hit rate, identical to the hit_rate
    baseline. Callers that want a per-attempt rate (apples-to-apples with the
    AL hits/desired numerator, which is divided by the surrogate's requested
    count and therefore implicitly pays for Run3ModelGen failures) must
    multiply the returned rates by `p_valid` (Run3ModelGen validity rate from
    the AL log: `Filter ...: N_valid / N_total samples kept`). The main()
    plotting loop does this scaling for the hits_per_desired metric.
    """
    return _baseline_hit_rate_trajectory(run_dir, seed, Y_full, true_value, tol)


def _pool_prevalence(Y_full: np.ndarray, true_value: float, tols) -> dict:
    return {float(t): float(np.mean(np.abs(Y_full - true_value) / true_value < t)) for t in tols}


_VALIDITY_RE = re.compile(
    r"Filter \(MO_Omega > 0 & < 1 & SP_m_h != -1\):\s*(\d+)\s*/\s*(\d+)\s*samples kept"
)


def _extract_validity_rate(run_dirs) -> tuple[float | None, int | None, int | None, str | None]:
    """Parse Run3ModelGen validity rate from the first AL log we can find.

    The filter line is logged once per run at startup. All runs that share a
    raw-pool data dir share the same validity rate, so we return the first
    match. Returns (p_valid, n_valid, n_total, source_log) or all-None.
    """
    for run_dir in run_dirs:
        log_path = Path(run_dir) / "active_learning.log"
        if not log_path.exists():
            continue
        try:
            with open(log_path) as fh:
                for line in fh:
                    m = _VALIDITY_RE.search(line)
                    if m:
                        n_valid, n_total = int(m.group(1)), int(m.group(2))
                        if n_total <= 0:
                            continue
                        return n_valid / n_total, n_valid, n_total, str(log_path)
        except Exception:
            continue
    return None, None, None, None


# ════════════════════════════════════════════════════════════════════════════
# Classification-accuracy trajectories (recompute from per-iter checkpoints)
# ════════════════════════════════════════════════════════════════════════════

ACC_DATASETS = ("static_random", "mcmc", "train", "val")
ACC_ROLES = ("al", "baseline")

# Regex for active_learning.log startup banner. Matches lines like
# "2026-04-28 01:33:27 [info     ]   kernel: RBF" — captures key+value.
# The 3-space indent inside `[level    ]` is the banner-only signature; later
# log lines use a single space after `]` and won't match.
import re as _re  # noqa: E402
_LOG_KV_RE = _re.compile(r"\[\w+\s*\]\s{3}(\w+):\s*(.+?)\s*$")

_GP_KWARG_KEYS = {
    "kernel", "lengthscale", "noise", "use_ard", "m_nu", "num_mixtures",
    "use_dkl", "feature_dim", "num_hidden_dims", "num_middle_dims",
    "num_inducing_max", "inducing_strategy",
}


def _coerce_log_value(v: str):
    """Coerce a stringified log value (`"True"`, `"0.001"`, `"RBF"`) to its type."""
    if v in ("True", "False"):
        return v == "True"
    if v in ("None", "null"):
        return None
    try:
        if "." in v or "e" in v or "E" in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _parse_run_kwargs_from_log(run_dir: Path) -> dict:
    """Parse the AL/AL-GP/AL-TabPFN startup banner for run-config kwargs.

    Reads `<run_dir>/active_learning.log` (only the head — the banner is at the
    top, before any iteration starts), pulls every `key: value` line whose
    indentation matches a config row, and returns a flat dict. Used to source
    GP kernel/ARD/inducing settings that are not saved to state.pt.

    Returns an empty dict on any read/parse failure.
    """
    log_path = Path(run_dir) / "active_learning.log"
    if not log_path.exists():
        return {}
    try:
        with open(log_path) as fh:
            head = []
            for line in fh:
                head.append(line)
                if "Starting active learning" in line or len(head) > 400:
                    break
        out: dict = {}
        for line in head:
            m = _LOG_KV_RE.search(line)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
            if k in _GP_KWARG_KEYS or k in {"y_transform", "model_type", "target",
                                            "num_inducing_max", "gp_num_samples"}:
                out[k] = _coerce_log_value(v)
        return out
    except Exception:
        return {}


def _classification_accuracy(y_pred_t: np.ndarray, y_true_t: np.ndarray,
                             threshold: float) -> float:
    """Binary classification accuracy at `threshold` in transformed space."""
    if len(y_pred_t) == 0:
        return float("nan")
    return float(np.mean((y_pred_t >= threshold) == (y_true_t >= threshold)))


def _load_accuracy_cache(run_dir: Path) -> dict:
    """Load `<run_dir>/accuracy_trajectory.json`, or return an empty cache."""
    p = Path(run_dir) / "accuracy_trajectory.json"
    if p.exists():
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def _save_accuracy_cache(run_dir: Path, cache: dict) -> None:
    p = Path(run_dir) / "accuracy_trajectory.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
    tmp.replace(p)


def _predict_transformer_t(model, X: torch.Tensor, stats, device: str,
                           batch_size: int = 1024) -> np.ndarray:
    """Predict transformer outputs in transformed (log) space."""
    mean_X, std_X, _mean_Y, _std_Y = stats
    X_norm = (X - mean_X) / std_X
    model.eval()
    model.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            out.append(model(X_norm[i:i + batch_size].to(device)).cpu())
    return torch.cat(out, dim=0).numpy().ravel().astype(np.float64)


def _predict_gp_t(model, model_type: str, X: torch.Tensor, data_min, data_max,
                  jitter: float, num_samples: int, device: str,
                  batch_size: int | None = None) -> np.ndarray:
    """Predict GP mean in transformed space (matches gp_predict path)."""
    from pmssm.visualization import gp_predict  # noqa: PLC0415
    from pmssm.data import normalize_x  # noqa: PLC0415
    x_norm = normalize_x(X, data_min, data_max)
    if batch_size is None:
        batch_size = 5000 if model_type == "exact_gp" else 10_000
    out = []
    for i in range(0, len(x_norm), batch_size):
        chunk = x_norm[i:i + batch_size].to(device)
        y_t = gp_predict(model, chunk, model_type, jitter=jitter, num_samples=num_samples)
        out.append(y_t.detach().cpu().numpy().ravel())
    return np.concatenate(out).astype(np.float64)


def _build_eval_sets(run_dir: Path, state: dict, seed: int,
                     X_full: np.ndarray, Y_full: np.ndarray,
                     X_static: torch.Tensor, Y_static: torch.Tensor,
                     X_mcmc: torch.Tensor, Y_mcmc: torch.Tensor,
                     iter_idx: int) -> dict | None:
    """Build the four (X, Y) eval sets for one (run, iter) pair, all torch tensors.

    Returns `{"al": {dataset: (X, Y)}, "baseline": {dataset: (X, Y)}}` or None
    if the iteration's data can't be reconstructed.
    """
    al_n_train = list(state.get("al_n_train") or [])
    al_n_val = list(state.get("al_n_val") or [])
    if iter_idx >= len(al_n_train) or iter_idx >= len(al_n_val):
        return None

    n_tr = int(al_n_train[iter_idx])
    n_val = int(al_n_val[iter_idx])
    X_state = state.get("X")
    Y_state = state.get("Y")
    X_val_state = state.get("X_val")
    Y_val_state = state.get("Y_val")
    if X_state is None or Y_state is None or X_val_state is None or Y_val_state is None:
        return None

    def _t(arr) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().to(torch.float32)
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))

    al_X_tr = _t(X_state)[:n_tr]
    al_Y_tr = _t(Y_state).view(-1)[:n_tr]
    al_X_val = _t(X_val_state)[:n_val]
    al_Y_val = _t(Y_val_state).view(-1)[:n_val]

    perm = _seed_perm(len(Y_full), seed)
    X_full_shuf = np.asarray(X_full)[perm]
    Y_full_shuf = np.asarray(Y_full)[perm]

    base_tr = _baseline_iter_xy(state, X_full_shuf, Y_full_shuf, iter_idx, role="train")
    base_va = _baseline_iter_xy(state, X_full_shuf, Y_full_shuf, iter_idx, role="val")
    if base_tr is None or base_va is None:
        return None
    bX_tr, bY_tr = base_tr
    bX_va, bY_va = base_va

    return {
        "al": {
            "static_random": (X_static, Y_static),
            "mcmc": (X_mcmc, Y_mcmc),
            "train": (al_X_tr, al_Y_tr),
            "val": (al_X_val, al_Y_val),
        },
        "baseline": {
            "static_random": (X_static, Y_static),
            "mcmc": (X_mcmc, Y_mcmc),
            "train": (_t(bX_tr), _t(bY_tr)),
            "val": (_t(bX_va), _t(bY_va)),
        },
    }


def _load_iter_model(model_type: str, role: str, iter_dir: Path,
                     X_train_i: torch.Tensor, Y_train_i: torch.Tensor,
                     X_val_i: torch.Tensor, Y_val_i: torch.Tensor,
                     run_kwargs: dict, device: str, dropout: float = 0.1):
    """Construct the model used for evaluation at this iteration.

    Always loads from the per-iter checkpoint saved by the AL driver:
    ``al_model_checkpoint.pt`` (AL) or ``baseline_model_checkpoint.pt``
    (random baseline). For transformer the state_dict is loaded into a
    freshly-instantiated model whose architecture matches active_learning.py
    (d_model=128, nhead=4, num_layers=3, dim_feedforward=512). For GPs the
    architecture is rebuilt with the iteration's train/val data (inducing-point
    selection depends on it) using kernel/ARD/noise kwargs parsed from the
    run's startup banner, then state_dict + likelihood_state_dict are loaded.

    TabPFN is not supported here: AL TabPFN runs save no weight file (TabPFN
    is in-context-fit per call), so reproducing its AL-time predictions would
    require re-running the AL pipeline. Skip those picks at the caller.
    """
    ckpt_name = "al_model_checkpoint.pt" if role == "al" else "baseline_model_checkpoint.pt"
    ckpt_path = iter_dir / ckpt_name

    if model_type == "transformer":
        if not ckpt_path.exists():
            return None
        from pmssm.models import PMSSMTransformerTabular  # noqa: PLC0415
        model = PMSSMTransformerTabular(d_model=128, nhead=4, num_layers=3,
                                        dim_feedforward=512, dropout=dropout)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model.to(device)

    if model_type == "dnn":
        if not ckpt_path.exists():
            return None
        from pmssm.models import PMSSMFeedForward  # noqa: PLC0415
        # Architecture must match active_learning_dnn.py defaults.
        model = PMSSMFeedForward(n_params=19, d_model=64, num_layers=4,
                                 dim_feedforward=256, dropout=dropout)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model.to(device)

    if model_type == "dnn_match_trafo":
        if not ckpt_path.exists():
            return None
        from pmssm.models import PMSSMFeedForward  # noqa: PLC0415
        # Hyperparams chosen to roughly match transformer's parameter budget.
        model = PMSSMFeedForward(n_params=19, d_model=64, num_layers=3,
                                 dim_feedforward=400, dropout=dropout)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model.to(device)

    if model_type in ("exact_gp", "deep_gp", "sparse_gp"):
        if not ckpt_path.exists():
            return None
        from pmssm.training import create_gp_model  # noqa: PLC0415
        from pmssm.data import normalize_x, transform_y, build_norm_tensors, PARAM_ORDER  # noqa: PLC0415
        target = run_kwargs.get("target", "DMRD") or "DMRD"
        data_min, data_max = build_norm_tensors()
        x_tr = normalize_x(X_train_i, data_min, data_max).to(device)
        y_tr = transform_y(Y_train_i, target=target).view(-1).to(device)
        x_va = normalize_x(X_val_i, data_min, data_max).to(device)
        y_va = transform_y(Y_val_i, target=target).view(-1).to(device)
        gp_kwargs = {k: v for k, v in run_kwargs.items() if k in _GP_KWARG_KEYS}
        # Sensible defaults if log parsing missed something
        gp_kwargs.setdefault("kernel", "RBF")
        gp_kwargs.setdefault("use_ard", True)
        gp_kwargs.setdefault("noise", 1e-2)
        gp_kwargs.setdefault("lengthscale", 1.0)
        if model_type == "deep_gp":
            gp_kwargs.setdefault("num_inducing_max", 256)
            gp_kwargs.setdefault("num_hidden_dims", 10)
            gp_kwargs.setdefault("num_middle_dims", 0)
        num_samples = int(run_kwargs.get("gp_num_samples", 8) or 8)
        model = create_gp_model(model_type, x_tr, y_tr, x_va, y_va,
                                n_dim=len(PARAM_ORDER), num_samples=num_samples,
                                target=target, device=device, **gp_kwargs)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if hasattr(model, "likelihood") and "likelihood_state_dict" in ckpt:
            model.likelihood.load_state_dict(ckpt["likelihood_state_dict"])
            model.likelihood = model.likelihood.to(device)
        return model.to(device)

    raise ValueError(f"unsupported model_type for checkpoint reload: {model_type}")


def _accuracy_for_iter(model_type: str, role: str, iter_dir: Path,
                       eval_sets: dict, run_kwargs: dict, threshold_t: float,
                       device: str, dropout: float = 0.1) -> dict | None:
    """Run inference and return `{dataset: accuracy}` for one (iter, role) pair.

    Returns None if the model can't be loaded for this role/iter.
    """
    sets_role = eval_sets[role]
    X_tr, Y_tr = sets_role["train"]
    X_va, Y_va = sets_role["val"]
    if len(X_tr) == 0 or len(X_va) == 0:
        return None

    model = _load_iter_model(model_type, role, iter_dir, X_tr, Y_tr, X_va, Y_va,
                             run_kwargs, device, dropout=dropout)
    if model is None:
        return None

    target = run_kwargs.get("target", "DMRD") or "DMRD"
    from pmssm.data import transform_y  # noqa: PLC0415

    if model_type in ("transformer", "dnn"):
        from pmssm.data import compute_stats  # noqa: PLC0415
        # Match active_learning(_dnn).py: stats from the train set with idx_train = arange(n_train)
        idx_tr = torch.arange(len(X_tr))
        stats = compute_stats(X_tr, Y_tr.unsqueeze(-1), idx_tr)
    else:
        stats = None

    out = {}
    for ds in ACC_DATASETS:
        X_d, Y_d = sets_role[ds]
        if len(X_d) == 0:
            continue  # eval set absent for this run (e.g. MCMC dir not provided)
        Y_true_t = transform_y(Y_d.float(), target=target).numpy().ravel().astype(np.float64)

        ds_t0 = time.time()
        if model_type in ("transformer", "dnn"):
            # PMSSMFeedForward shares the (B,19)->(B,1) interface so the same
            # mean/std-normalised forward path works for both architectures.
            y_pred_t = _predict_transformer_t(model, X_d.float(), stats, device)
        elif model_type in ("exact_gp", "deep_gp", "sparse_gp"):
            jitter = float(run_kwargs.get("jitter", 1e-3) or 1e-3)
            num_samples = int(run_kwargs.get("gp_num_samples", 8) or 8)
            y_pred_t = _predict_gp_t(model, model_type, X_d.float(), *_GP_NORM,
                                     jitter=jitter, num_samples=num_samples,
                                     device=device)
        else:
            raise ValueError(f"unsupported model_type: {model_type}")
        click.echo(f"[accuracy]         {role}/{ds}: n={len(X_d)} "
                   f"predicted in {time.time()-ds_t0:5.1f}s", err=False)
        sys.stdout.flush()

        out[ds] = _classification_accuracy(y_pred_t, Y_true_t, threshold_t)

    # Free GPU memory between iterations.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# Cached GP normalization tensors (populated once per process — they're constant).
_GP_NORM: tuple = (None, None)


def _classification_accuracy_trajectory(run, run_dir: str, seed: int,
                                        model_type: str,
                                        X_full: np.ndarray, Y_full: np.ndarray,
                                        X_static: torch.Tensor, Y_static: torch.Tensor,
                                        X_mcmc: torch.Tensor, Y_mcmc: torch.Tensor,
                                        threshold_t: float, device: str,
                                        refresh: bool = False,
                                        dropout: float = 0.1) -> dict:
    """Build {role: {dataset: [(iter, acc), ...]}} for one seed run.

    Cache hits short-circuit per iteration; misses load the iter's checkpoints
    and run inference. The result is also written back to
    `<run_dir>/accuracy_trajectory.json`.
    """
    # Oracle / theoretical-limit runs are tagged "<model>_oracle" in the
    # manifest so they appear as separate plot entries, but their on-disk
    # checkpoints are saved by the same per-model AL pipeline, so the
    # _load_iter_model dispatch only knows the bare model name. Strip the
    # _oracle suffix before any downstream dispatch.
    if model_type.endswith("_oracle"):
        model_type = model_type[: -len("_oracle")]

    run_dir_p = Path(run_dir)
    cache = {} if refresh else _load_accuracy_cache(run_dir_p)

    state_path = run_dir_p / "state.pt"
    if not state_path.exists():
        return {role: {ds: [] for ds in ACC_DATASETS} for role in ACC_ROLES}
    state = torch.load(state_path, weights_only=False, map_location="cpu")
    run_kwargs = _parse_run_kwargs_from_log(run_dir_p)

    al_n_train = list(state.get("al_n_train") or [])
    out = {role: {ds: [] for ds in ACC_DATASETS} for role in ACC_ROLES}

    eval_sets_cached: dict[int, dict] = {}
    n_iters_total = len(al_n_train)
    n_computed = 0
    n_cached = 0
    seed_t0 = time.time()
    click.echo(f"[accuracy]     seed={seed} run={run_dir_p.name}: "
               f"{n_iters_total} iters, "
               f"cache_entries={len(cache)}, refresh={refresh}, "
               f"model_type={model_type}", err=False)
    sys.stdout.flush()

    # Datasets the *current* invocation can actually evaluate. Datasets with
    # zero-length tensors (e.g. MCMC not provided this run) are skipped from
    # the cache-completeness check so the worker doesn't loop forever.
    available_static = len(X_static) > 0
    available_mcmc = len(X_mcmc) > 0
    available_per_role = {
        "static_random": available_static,
        "mcmc": available_mcmc,
        "train": True,
        "val": True,
    }
    required_datasets = [ds for ds in ACC_DATASETS if available_per_role[ds]]

    for i in range(len(al_n_train)):
        iter_no = i + 1
        iter_dir = run_dir_p / f"iteration_{iter_no:03d}"
        if not iter_dir.exists():
            continue
        key = str(iter_no)
        cache_iter = cache.get(key, {})
        if not isinstance(cache_iter, dict):
            cache_iter = {}

        # Decide whether we need any compute for this iter.
        needs_compute = refresh
        if not needs_compute:
            for role in ACC_ROLES:
                role_cache = cache_iter.get(role) or {}
                if not all(ds in role_cache for ds in required_datasets):
                    needs_compute = True
                    break

        if needs_compute:
            iter_t0 = time.time()
            click.echo(f"[accuracy]       iter {iter_no}/{n_iters_total} "
                       f"start (n_train={al_n_train[i]})", err=False)
            sys.stdout.flush()
            sets = eval_sets_cached.get(i)
            if sets is None:
                sets = _build_eval_sets(run_dir_p, state, seed, X_full, Y_full,
                                        X_static, Y_static, X_mcmc, Y_mcmc, i)
                if sets is None:
                    click.echo(f"[accuracy]       iter {iter_no}/{n_iters_total} "
                               "skipped: could not build eval sets", err=True)
                    sys.stdout.flush()
                    continue
                eval_sets_cached[i] = sets

            roles_done = []
            roles_acc_summary: list[str] = []
            for role in ACC_ROLES:
                role_cache = cache_iter.get(role) or {}
                if not refresh and all(ds in role_cache for ds in required_datasets):
                    continue
                role_t0 = time.time()
                click.echo(f"[accuracy]         loading {role} checkpoint", err=False)
                sys.stdout.flush()
                role_acc = _accuracy_for_iter(model_type, role, iter_dir, sets,
                                              run_kwargs, threshold_t, device,
                                              dropout=dropout)
                if role_acc is None:
                    click.echo(f"[accuracy]         {role}: checkpoint missing or "
                               "eval set empty -- skipped", err=True)
                    sys.stdout.flush()
                    continue
                role_cache.update({k: float(v) for k, v in role_acc.items()})
                cache_iter[role] = role_cache
                roles_done.append(role)
                roles_acc_summary.append(
                    f"{role}=" + "/".join(f"{ds}={role_acc[ds]:.4f}"
                                          for ds in ACC_DATASETS if ds in role_acc)
                )
                click.echo(f"[accuracy]         {role} done in "
                           f"{time.time()-role_t0:5.1f}s", err=False)
                sys.stdout.flush()

            cache[key] = cache_iter
            _save_accuracy_cache(run_dir_p, cache)
            n_computed += 1
            click.echo(f"[accuracy]       iter {iter_no}/{n_iters_total} "
                       f"computed in {time.time()-iter_t0:5.1f}s "
                       f"(roles={','.join(roles_done) or 'none'}, "
                       f"n_train={al_n_train[i]}) "
                       + " | ".join(roles_acc_summary), err=False)
            sys.stdout.flush()
        else:
            n_cached += 1

        for role in ACC_ROLES:
            role_acc = (cache_iter.get(role) or {})
            for ds in ACC_DATASETS:
                if ds in role_acc:
                    out[role][ds].append((iter_no, float(role_acc[ds])))

    click.echo(f"[accuracy]     seed={seed} done in {time.time()-seed_t0:5.1f}s "
               f"(computed={n_computed}, cached={n_cached})", err=False)
    sys.stdout.flush()
    return out


def _collect_accuracy_trajectories(df, picks, target: str, X_full: np.ndarray,
                                   Y_full: np.ndarray, X_static: torch.Tensor,
                                   Y_static: torch.Tensor, X_mcmc: torch.Tensor,
                                   Y_mcmc: torch.Tensor, threshold_t: float,
                                   device: str, min_seeds: int,
                                   refresh: bool = False,
                                   dropout: float = 0.1) -> dict:
    """Build accuracy trajectories for every picked (model, strategy, warm)+seed.

    TabPFN picks are skipped: AL TabPFN runs save no per-iteration weight file,
    so reproducing AL-time predictions would require re-running the pipeline.

    Returns
        {(model, strat, warm): {role: {dataset: (iters_axis, Y[n_seeds, n_iters])}}}
    """
    global _GP_NORM
    if _GP_NORM == (None, None):
        from pmssm.data import build_norm_tensors  # noqa: PLC0415
        _GP_NORM = build_norm_tensors()

    click.echo(f"[accuracy] starting collection over {len(picks)} pick(s); "
               f"min_seeds={min_seeds}, refresh={refresh}, device={device}")
    sys.stdout.flush()

    out: dict = {}
    for pick_idx, (model, strat, warm, _tu, _sc) in enumerate(picks, start=1):
        if model == "tabpfn":
            click.echo(f"[accuracy] [{pick_idx}/{len(picks)}] skipping {model}-{strat}-{warm}: "
                       "TabPFN saves no per-iter checkpoint, would require re-running AL.",
                       err=True)
            sys.stdout.flush()
            continue
        sub = df[(df["model"] == model) & (df["strategy"] == strat)
                 & (df["warm_start"] == warm)]
        click.echo(f"[accuracy] [{pick_idx}/{len(picks)}] pick {model}-{strat}-{warm}: "
                   f"{len(sub)} seed runs to process")
        sys.stdout.flush()
        pick_t0 = time.time()
        per_role_ds: dict = {role: {ds: [] for ds in ACC_DATASETS} for role in ACC_ROLES}
        valid_seeds = 0
        n_seeds_total = len(sub)
        for seed_idx, (_, row) in enumerate(sub.iterrows(), start=1):
            run_dir = row.get("expected_run_dir")
            seed = int(row.get("seed"))
            if not isinstance(run_dir, str) or not run_dir:
                continue
            click.echo(f"[accuracy]   [{pick_idx}/{len(picks)}] "
                       f"seed {seed_idx}/{n_seeds_total} (seed={seed}) -> {run_dir}")
            sys.stdout.flush()
            try:
                run = _load_run(run_dir)
            except Exception as exc:
                click.echo(f"[warn] accuracy: skip {run_dir}: {exc}", err=True)
                sys.stdout.flush()
                continue
            try:
                seed_traj = _classification_accuracy_trajectory(
                    run, run_dir, seed, model, X_full, Y_full,
                    X_static, Y_static, X_mcmc, Y_mcmc, threshold_t,
                    device=device, refresh=refresh, dropout=dropout,
                )
            except Exception as exc:
                click.echo(f"[warn] accuracy: failed for {run_dir}: {exc}", err=True)
                sys.stdout.flush()
                continue
            recorded = False
            for role in ACC_ROLES:
                for ds in ACC_DATASETS:
                    pts = seed_traj[role][ds]
                    if pts:
                        per_role_ds[role][ds].append(pts)
                        recorded = True
            if recorded:
                valid_seeds += 1

        click.echo(f"[accuracy]   pick {model}-{strat}-{warm} done in "
                   f"{time.time()-pick_t0:5.1f}s ({valid_seeds} valid seeds)")
        sys.stdout.flush()

        if valid_seeds < min_seeds:
            click.echo(f"[accuracy] {model}-{strat}-{warm}: "
                       f"only {valid_seeds} seeds with data — dropped (min={min_seeds})",
                       err=True)
            continue

        cfg_out: dict = {}
        for role in ACC_ROLES:
            cfg_out[role] = {}
            for ds in ACC_DATASETS:
                trajs = per_role_ds[role][ds]
                if len(trajs) < min_seeds:
                    continue
                max_len = max(len(t) for t in trajs)
                Y = np.full((len(trajs), max_len), np.nan, dtype=np.float64)
                iters_ax = None
                for j, t in enumerate(trajs):
                    if len(t) == max_len and iters_ax is None:
                        iters_ax = np.asarray([it for it, _ in t])
                    Y[j, :len(t)] = [a for _, a in t]
                if iters_ax is None:
                    iters_ax = np.asarray([it for it, _ in trajs[0]])
                n_per_iter = np.sum(~np.isnan(Y), axis=0)
                keep = n_per_iter >= min_seeds
                if not keep.any():
                    continue
                cfg_out[role][ds] = (iters_ax[keep], Y[:, keep])
        if any(cfg_out[r] for r in ACC_ROLES):
            out[(model, strat, warm)] = cfg_out

    return out


def plot_classification_accuracy_oracle_comparison(traj_acc: dict, picks,
                                                   out_dir: Path,
                                                   uncertainty: str) -> list:
    """Oracle counterpart of `plot_classification_accuracy_best_per_model`.

    For each regular best-per-model pick whose `<model>_oracle` counterpart has
    accuracy data in `traj_acc`, render the same per-dataset accuracy plot
    showing the regular AL+baseline curves (solid/dashed) alongside the oracle
    AL+baseline curves (dotted/dash-dot). One PNG per eval dataset:
    `accuracy_oracle_comparison_<ds>.png`.

    Returns the written paths.
    """
    written = []
    dataset_titles = {
        "static_random": "Static random eval set",
        "mcmc": "MCMC eval set",
        "train": "Per-model own train set",
        "val": "Per-model own validation set",
    }

    # Identify oracle picks: for each base model whose <model>_oracle has
    # entries in traj_acc, find the (s, w) cell present (there should only
    # be one — submit_strategy_sweep.sh OUTPUT_TAG=oracle launched a single
    # cell per oracle model). Score by picking the same way as for regular
    # models would be unnecessary: just take the single (s, w) available.
    oracle_models = sorted({m for (m, _, _) in traj_acc.keys()
                            if m.endswith("_oracle")})
    oracle_picks = []  # (base_model, s, w)
    for om in oracle_models:
        base = om[: -len("_oracle")]
        oracle_cfgs = [(s, w) for (m, s, w) in traj_acc.keys() if m == om]
        if oracle_cfgs:
            s, w = sorted(oracle_cfgs)[0]
            oracle_picks.append((base, s, w))

    if not oracle_picks:
        return written

    # Only keep regular picks whose base model has an oracle counterpart, so
    # the plot stays focused on the comparison the user asked for.
    pickable_models = {base for (base, _, _) in oracle_picks}
    regular_picks = [(m, s, w, _tu, _sc) for (m, s, w, _tu, _sc) in picks
                     if m in pickable_models]
    if not regular_picks:
        return written

    for ds in ACC_DATASETS:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        ax.set_title(f"Classification accuracy (oracle comparison) — {dataset_titles[ds]}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy (Ωh² ≷ 0.12)")
        ax.grid(alpha=0.3)

        any_data = False
        y_lo, y_hi = float("inf"), float("-inf")
        model_order: list[str] = []
        # Keys are (base_model, is_oracle, role) → seed count for that curve.
        n_seeds_per_curve: dict[tuple[str, bool, str], int] = {}

        def _draw_pair(m_for_color: str, cfg: tuple, is_oracle: bool):
            nonlocal any_data, y_lo, y_hi
            if cfg not in traj_acc:
                return
            for role in ACC_ROLES:
                role_traj = traj_acc[cfg].get(role, {})
                if ds not in role_traj:
                    continue
                iters_ax, Y = role_traj[ds]
                if m_for_color not in model_order:
                    model_order.append(m_for_color)
                n_seeds_per_curve[(m_for_color, is_oracle, role)] = int(
                    (~np.isnan(Y)).any(axis=1).sum()
                )
                # Regular: solid (AL) / dashed (baseline). Oracle: dotted (AL)
                # / dash-dot (baseline). Same colour per base model.
                if not is_oracle:
                    ls = "-" if role == "al" else "--"
                    mk = "o" if role == "al" else "s"
                    lw = 1.6 if role == "al" else 1.3
                else:
                    ls = ":" if role == "al" else (0, (3, 1, 1, 1))
                    mk = "*" if role == "al" else "x"
                    lw = 2.0 if role == "al" else 1.3
                _draw_curve(
                    ax, iters_ax, Y,
                    color=MODEL_COLORS.get(m_for_color, "gray"),
                    linestyle=ls,
                    marker=mk,
                    label=None,
                    uncertainty=uncertainty,
                    linewidth=lw,
                    fill_alpha=0.14 if role == "al" else 0.06,
                    alpha=1.0 if role == "al" else 0.85,
                )
                any_data = True
                mean = np.nanmean(Y, axis=0)
                if np.isfinite(mean).any():
                    y_lo = min(y_lo, float(np.nanmin(mean)))
                    y_hi = max(y_hi, float(np.nanmax(mean)))

        for (m, s, w, _tu, _sc) in regular_picks:
            _draw_pair(m, (m, s, w), is_oracle=False)
        for (base, s, w) in oracle_picks:
            _draw_pair(base, (f"{base}_oracle", s, w), is_oracle=True)

        if not any_data:
            plt.close(fig)
            continue

        if np.isfinite(y_lo) and np.isfinite(y_hi):
            pad = max(0.02, (y_hi - y_lo) * 0.15)
            ax.set_ylim(max(0.0, y_lo - pad), min(1.0, y_hi + pad))
        else:
            ax.set_ylim(0, 1.02)

        # Split legend: "Model" (colour swatches) + "Curve" (linestyle key).
        model_handles = []
        for m in model_order:
            counts = [n for (mm, _o, _r), n in n_seeds_per_curve.items() if mm == m]
            n_str = f"n={max(counts)}" if counts else "n=0"
            model_handles.append(Line2D(
                [0], [0], color=MODEL_COLORS.get(m, "gray"), lw=2.4,
                label=f"{MODEL_DISPLAY.get(m, m)} ({n_str})",
            ))
        role_handles = [
            Line2D([0], [0], color="black", linestyle="-",  marker="o",
                   markersize=5, label="AL"),
            Line2D([0], [0], color="black", linestyle="--", marker="s",
                   markersize=5, label="baseline"),
            Line2D([0], [0], color="black", linestyle=":",  marker="*",
                   markersize=6, label="AL (oracle)"),
            Line2D([0], [0], color="black", linestyle=(0, (3, 1, 1, 1)),
                   marker="x", markersize=5, label="baseline (oracle)"),
        ]
        leg_model = ax.legend(
            handles=model_handles, loc="lower right",
            fontsize=9, frameon=True, framealpha=0.9,
            title="Model", title_fontsize=10,
        )
        ax.add_artist(leg_model)
        ax.legend(
            handles=role_handles, loc="upper left",
            fontsize=9, frameon=True, framealpha=0.9,
            title="Curve", title_fontsize=10,
        )

        fig.tight_layout()
        out_path = out_dir / f"accuracy_oracle_comparison_{ds}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def plot_classification_accuracy_best_per_model(traj_acc: dict, picks, out_dir: Path,
                                                uncertainty: str) -> list:
    """Render four PNGs (static_random, mcmc, train, val) — accuracy vs iteration.

    Mirrors `plot_best_per_model`'s layout: solid line per model (AL), dashed
    line per model (baseline) in the same colour, SEM band across seeds.
    Y-axis auto-zooms with padding so MCMC and other narrow ranges are
    readable. Pick details (strategy/warm) move out of the legend into a
    footnote, leaving short `model (AL)` / `model (baseline)` labels.
    """
    written = []
    dataset_titles = {
        "static_random": "Static random eval set",
        "mcmc": "MCMC eval set",
        "train": "Per-model own train set",
        "val": "Per-model own validation set",
    }
    data_efficiency_all: dict[str, dict[str, dict]] = {}
    for ds in ACC_DATASETS:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        ax.set_title(f"Classification accuracy — {dataset_titles[ds]}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy (Ωh² ≷ 0.12)")
        ax.grid(alpha=0.3)

        any_data = False
        y_lo, y_hi = float("inf"), float("-inf")
        model_order: list[str] = []
        n_seeds_per_role: dict[tuple[str, str], int] = {}
        # Per-model curve cache for the data-efficiency calculation.
        # m -> role -> (iters_ax, mean_per_iter)
        per_model_means: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for (m, s, w, _tu, _sc) in picks:
            cfg = (m, s, w)
            if cfg not in traj_acc:
                continue
            for role in ACC_ROLES:
                role_traj = traj_acc[cfg].get(role, {})
                if ds not in role_traj:
                    continue
                iters_ax, Y = role_traj[ds]
                if m not in model_order:
                    model_order.append(m)
                n_seeds_per_role[(m, role)] = int((~np.isnan(Y)).any(axis=1).sum())
                _draw_curve(
                    ax, iters_ax, Y,
                    color=MODEL_COLORS.get(m, "gray"),
                    linestyle="-" if role == "al" else "--",
                    marker="o" if role == "al" else "s",
                    label=None,
                    uncertainty=uncertainty,
                    linewidth=1.6 if role == "al" else 1.3,
                    fill_alpha=0.14 if role == "al" else 0.06,
                    alpha=1.0 if role == "al" else 0.85,
                )
                any_data = True
                mean = np.nanmean(Y, axis=0)
                per_model_means.setdefault(m, {})[role] = (np.asarray(iters_ax), mean)
                if np.isfinite(mean).any():
                    y_lo = min(y_lo, float(np.nanmin(mean)))
                    y_hi = max(y_hi, float(np.nanmax(mean)))

        if not any_data:
            plt.close(fig)
            continue

        # ── Data-efficiency: iter at which AL mean accuracy first matches the
        # baseline's final-iteration mean accuracy. Gain = baseline_iter /
        # al_match_iter. Printed and saved to JSON.
        ds_eff: dict[str, dict] = {}
        eff_lines: list[str] = []
        for m in model_order:
            roles = per_model_means.get(m, {})
            if "al" not in roles or "baseline" not in roles:
                continue
            bl_iters, bl_mean = roles["baseline"]
            al_iters, al_mean = roles["al"]
            finite_bl = np.where(np.isfinite(bl_mean))[0]
            if not len(finite_bl):
                continue
            last_idx = finite_bl[-1]
            target_iter = int(bl_iters[last_idx])
            target_acc = float(bl_mean[last_idx])
            valid_al = np.isfinite(al_mean) & (al_mean >= target_acc)
            if valid_al.any():
                match_idx = int(np.argmax(valid_al))  # first True
                match_iter = int(al_iters[match_idx])
                gain = target_iter / max(match_iter, 1)
                eff_lines.append(
                    f"  {MODEL_DISPLAY.get(m, m):<22s} target acc {target_acc:.4f} @ "
                    f"baseline iter {target_iter:>3d} → AL reaches it at iter "
                    f"{match_iter:>3d}  (gain {gain:5.2f}x)"
                )
                ds_eff[m] = {
                    "target_iter": target_iter,
                    "target_acc": target_acc,
                    "al_match_iter": match_iter,
                    "gain": gain,
                }
            else:
                al_best = float(np.nanmax(al_mean)) if np.isfinite(al_mean).any() else float("nan")
                eff_lines.append(
                    f"  {MODEL_DISPLAY.get(m, m):<22s} target acc {target_acc:.4f} @ "
                    f"baseline iter {target_iter:>3d} → AL never matched "
                    f"(AL best mean {al_best:.4f})"
                )
                ds_eff[m] = {
                    "target_iter": target_iter,
                    "target_acc": target_acc,
                    "al_match_iter": None,
                    "gain": None,
                    "al_best_mean": al_best,
                }
        if eff_lines:
            click.echo(f"[data-efficiency] {dataset_titles[ds]}")
            for line in eff_lines:
                click.echo(line)
        if ds_eff:
            data_efficiency_all[ds] = ds_eff

        # Auto-zoom y-axis with 10% padding (or at least 0.02), clipped to [0, 1].
        if np.isfinite(y_lo) and np.isfinite(y_hi):
            pad = max(0.02, (y_hi - y_lo) * 0.15)
            ax.set_ylim(max(0.0, y_lo - pad), min(1.0, y_hi + pad))
        else:
            ax.set_ylim(0, 1.02)

        # Split legend into two compact blocks placed outside the axes:
        #   "Model"  — one colour swatch per model, with seed count(s).
        #   "Curve"  — linestyle key (solid=AL, dashed=baseline).
        model_handles = []
        for m in model_order:
            n_al = n_seeds_per_role.get((m, "al"), 0)
            n_bl = n_seeds_per_role.get((m, "baseline"), 0)
            if n_al and n_bl and n_al != n_bl:
                n_str = f"n={n_al}/{n_bl}"
            else:
                n_str = f"n={n_al or n_bl}"
            model_handles.append(Line2D(
                [0], [0], color=MODEL_COLORS.get(m, "gray"), lw=2.4,
                label=f"{MODEL_DISPLAY.get(m, m)} ({n_str})",
            ))
        role_handles = [
            Line2D([0], [0], color="black", linestyle="-",  marker="o",
                   markersize=5, label="AL"),
            Line2D([0], [0], color="black", linestyle="--", marker="s",
                   markersize=5, label="baseline"),
        ]
        leg_model = ax.legend(
            handles=model_handles, loc="lower right",
            fontsize=9, frameon=True, framealpha=0.9,
            title="Model", title_fontsize=10,
        )
        ax.add_artist(leg_model)
        ax.legend(
            handles=role_handles, loc="upper left",
            fontsize=9, frameon=True, framealpha=0.9,
            title="Curve", title_fontsize=10,
        )

        fig.tight_layout()
        out_path = out_dir / f"accuracy_best_per_model_{ds}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    if data_efficiency_all:
        eff_path = out_dir / "data_efficiency_best_per_model.json"
        with open(eff_path, "w") as fh:
            json.dump(data_efficiency_all, fh, indent=2, default=str)
        click.echo(f"[data-efficiency] saved {eff_path}")
    return written


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
                runs.append((run_dir, _load_run(run_dir)))
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
        ax.set_title(f"{title_word} (|Ω − {true_val}| / {true_val} < {int(tol*100)}%)",
                     fontsize=15)
        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
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
        fig.subplots_adjust(right=0.74, wspace=0.28)
        fig.legend(seen.values(), seen.keys(),
                   loc="center left", bbox_to_anchor=(0.755, 0.5),
                   fontsize=14, frameon=True, borderaxespad=0.)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _finalize_split_legend(fig, axes, *, color_handles, style_handles,
                           out_path, color_title="Model", style_title="Curve",
                           color_loc="lower right", style_loc="upper left",
                           panel_index=-1, fontsize=11, title_fontsize=12):
    """Mirror of `_finalize` that places the legends *inside* the plot area
    using the accuracy-plot style: a Model/colour-key legend in the lower-right
    corner of one panel, and a Curve/linestyle-key legend in the upper-left.

    `color_handles`/`style_handles` are pre-built Line2D lists. Pass an empty
    list to skip the corresponding legend.
    """
    for ax in axes:
        _, ymax = ax.get_ylim()
        ax.set_ylim(0, max(ymax, 0.05) * 1.05)

    target_ax = axes[panel_index]
    fig.tight_layout()
    if color_handles:
        leg_c = target_ax.legend(
            handles=color_handles, loc=color_loc,
            fontsize=fontsize, frameon=True, framealpha=0.9,
            title=color_title, title_fontsize=title_fontsize,
        )
        target_ax.add_artist(leg_c)
    if style_handles:
        target_ax.legend(
            handles=style_handles, loc=style_loc,
            fontsize=fontsize, frameon=True, framealpha=0.9,
            title=style_title, title_fontsize=title_fontsize,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_models_per_strategy(traj, tols, uncertainty, true_val, out_dir,
                             file_prefix, title_word, ylabel,
                             baseline_traj=None, prevalence=None,
                             prevalence_label="random scan (full pool)"):
    """One figure per strategy; lines are (model, warm) combos for that strategy.

    Each figure has 1 row × len(tols) cols (default 3 panels: 10/20/50%).
    Color encodes the model, linestyle encodes the warm-start variant.

    If `baseline_traj` is provided (same {(model, strat, warm): {tol: ...}}
    structure), each plotted config also gets a dashed random-baseline curve
    in its model colour. If `prevalence` is provided ({tol: rate}), a
    horizontal grey reference line ("random scan (full pool)") is drawn.
    """
    written = []
    # Oracle (theoretical-limit) cells are folded into per-model plots only;
    # exclude them from per-strategy comparisons so the model-vs-model panel
    # isn't mixing realistic and oracle curves with the same colour.
    strategies = sorted({s for (m, s, _) in traj if not m.endswith("_oracle")})
    for strat in strategies:
        cfgs = [(m, s, w) for (m, s, w) in traj
                if s == strat and not m.endswith("_oracle")]
        if not cfgs:
            continue
        fig, axes = plt.subplots(1, len(tols), figsize=(6 * len(tols), 5),
                                 sharey=False, squeeze=False)
        axes = list(axes.flat)
        _setup_axes(axes, tols, true_val, title_word, ylabel)
        fig.suptitle(f"Strategy: {strat} — model & warm-start comparison",
                     fontsize=16)

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
                    label=None,
                    uncertainty=uncertainty,
                )
            if prevalence is not None and tol in prevalence:
                ax.axhline(prevalence[tol], color="black", linestyle=":", linewidth=1.4,
                           label=None)
                ax.text(0.99, prevalence[tol], f" {prevalence[tol]:.4f}",
                        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                        fontsize=10, color="black")

        # Model legend (colour key) + Warm-start legend (linestyle/marker key).
        models_present = list(dict.fromkeys(m for (m, _, _) in sorted(cfgs)))
        color_handles = [
            Line2D([0], [0], color=MODEL_COLORS.get(m, "gray"), lw=2.4,
                   label=MODEL_DISPLAY.get(m, m))
            for m in models_present
        ]
        warms_present = list(dict.fromkeys(w for (_, _, w) in sorted(cfgs)))
        style_handles = [
            Line2D([0], [0], color="black",
                   linestyle=WARM_LS.get(w, "-"),
                   marker=WARM_MARKER.get(w, "x"),
                   markersize=5, label=w)
            for w in warms_present
        ]
        if prevalence:
            style_handles.append(Line2D(
                [0], [0], color="black", linestyle=":", lw=1.4,
                label=prevalence_label,
            ))

        out_path = out_dir / f"{file_prefix}_strategy_{strat}.png"
        _finalize_split_legend(
            fig, axes,
            color_handles=color_handles, style_handles=style_handles,
            out_path=out_path,
            color_title="Model", style_title="Warm-start",
        )
        written.append(out_path)
    return written


def plot_strategies_per_model(traj, tols, uncertainty, true_val, out_dir,
                              file_prefix, title_word, ylabel,
                              baseline_traj=None, prevalence=None,
                              prevalence_label="random scan (full pool)"):
    """One figure per model; lines are (strategy, warm) combos for that model.

    Mirror of `plot_models_per_strategy` with model and strategy roles swapped.
    Color encodes the acquisition strategy, linestyle encodes the warm-start
    variant. Useful for comparing e.g. top_k vs top_k_tol_only vs entropy_batch
    on the same model.

    If `baseline_traj` is provided (same {(model, strat, warm): {tol: ...}}
    structure), each plotted config also gets a dashed random-baseline curve
    in its strategy colour. If `prevalence` is provided ({tol: rate}), a
    horizontal grey "random scan (full pool)" reference line is drawn.
    """
    written = []
    # Oracle (theoretical-limit) cells are NOT shown on per-model strategy
    # plots — they go into the standalone `*_oracle_comparison.png` plot
    # alongside the regular best-pick curves for transformer and deep_gp.
    models = sorted({m for (m, _, _) in traj if not m.endswith("_oracle")})
    for model in models:
        cfgs = [(m, s, w) for (m, s, w) in traj if m == model]
        if not cfgs:
            continue
        fig, axes = plt.subplots(1, len(tols), figsize=(6 * len(tols), 5),
                                 sharey=False, squeeze=False)
        axes = list(axes.flat)
        _setup_axes(axes, tols, true_val, title_word, ylabel)
        fig.suptitle(f"Model: {model} — strategy & warm-start comparison",
                     fontsize=16)

        for ax, tol in zip(axes, tols):
            for (m, s, w) in sorted(cfgs):
                if tol not in traj[(m, s, w)]:
                    continue
                iters_ax, Y = traj[(m, s, w)][tol]
                _draw_curve(
                    ax, iters_ax, Y,
                    color=STRATEGY_COLORS.get(s, "gray"),
                    linestyle=WARM_LS.get(w, "-"),
                    marker=WARM_MARKER.get(w, "x"),
                    label=None,
                    uncertainty=uncertainty,
                )
            if prevalence is not None and tol in prevalence:
                ax.axhline(prevalence[tol], color="black", linestyle=":", linewidth=1.4,
                           label=None)
                ax.text(0.99, prevalence[tol], f" {prevalence[tol]:.4f}",
                        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                        fontsize=10, color="black")

        # Strategy legend (colour key) + Warm-start legend (linestyle key).
        strategies_present = list(dict.fromkeys(s for (_, s, _) in sorted(cfgs)))
        color_handles = [
            Line2D([0], [0], color=STRATEGY_COLORS.get(s, "gray"), lw=2.4, label=s)
            for s in strategies_present
        ]
        warms_present = list(dict.fromkeys(w for (_, _, w) in sorted(cfgs)))
        style_handles = [
            Line2D([0], [0], color="black",
                   linestyle=WARM_LS.get(w, "-"),
                   marker=WARM_MARKER.get(w, "x"),
                   markersize=5, label=w)
            for w in warms_present
        ]
        if prevalence:
            style_handles.append(Line2D(
                [0], [0], color="black", linestyle=":", lw=1.4,
                label=prevalence_label,
            ))

        out_path = out_dir / f"{file_prefix}_model_{model}.png"
        _finalize_split_legend(
            fig, axes,
            color_handles=color_handles, style_handles=style_handles,
            out_path=out_path,
            color_title="Strategy", style_title="Warm-start",
        )
        written.append(out_path)
    return written


def _best_setting_for_model(traj, model, tols, iter_completeness=0.9):
    """Pick the (strategy, warm) for `model` with highest mean final-iter hit rate.

    Tries the strictest tolerance first; if no config has data there, falls back
    to progressively looser tolerances. Returns (strategy, warm, tol_used, score)
    or None if no eligible config exists.

    `iter_completeness` (0..1) filters out cells whose padded trajectory length
    is shorter than `iter_completeness * max_iters` across the model's cells.
    This prevents a cell that timed out early (e.g. TabPFN entropy_batch at
    iter 11/40) from beating a properly-completed cell on its mean-final
    score, which would otherwise be measured at the early cell's iter and at
    the completed cell's iter 40 — not the same comparison. Set to 0.0 to
    disable the filter.
    """
    candidates = [(m, s, w) for (m, s, w) in traj if m == model]
    if not candidates:
        return None

    # Compute per-cell trajectory length (max iter reached, ignoring NaN seeds)
    # by inspecting the iter axis from the first available tol.
    iters_per_cand = {}
    for c in candidates:
        for tol in tols:
            if tol in traj[c]:
                iters_ax, _ = traj[c][tol]
                iters_per_cand[c] = int(iters_ax[-1]) if len(iters_ax) else 0
                break
    if iters_per_cand:
        max_iters = max(iters_per_cand.values())
        threshold = int(max_iters * float(iter_completeness))
        eligible = [c for c in candidates if iters_per_cand.get(c, 0) >= threshold]
        if eligible:
            candidates = eligible

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


def plot_oracle_comparison(traj, tols, uncertainty, true_val, out_dir,
                           file_prefix, title_word, ylabel,
                           prevalence=None,
                           prevalence_label="random scan (full pool)"):
    """Render the theoretical-limit oracle comparison plot.

    Shows, side by side on the same axes per tolerance panel:
      - For each base model that has an `<model>_oracle` counterpart in
        `traj` (typically `transformer` and `deep_gp`):
          • the regular-pipeline best (strategy, warm) trajectory (solid)
          • the corresponding oracle trajectory (dotted, "(oracle)" label)

    This is the talk's headline "what AL would achieve given a perfect
    candidate pre-filter" comparison. Returns the written paths.
    """
    written = []
    oracle_models = sorted({m for (m, _, _) in traj if m.endswith("_oracle")})
    if not oracle_models:
        return written

    fig, axes = plt.subplots(1, len(tols), figsize=(6 * len(tols), 5),
                             sharey=False, squeeze=False)
    axes = list(axes.flat)
    _setup_axes(axes, tols, true_val, title_word, ylabel)
    fig.suptitle(
        "Oracle comparison: regular AL vs theoretical limit "
        "(candidates restricted to MCMC pool)",
        fontsize=16,
    )

    # Capture per-base-model picks + seed counts so we can build the legend
    # once after drawing on every panel.
    base_picks: dict[str, tuple[str, str, int]] = {}  # base_model -> (s, w, n)
    oracle_picks: dict[str, tuple[str, str, int]] = {}  # base_model -> (s, w, n)
    for ax, tol in zip(axes, tols):
        for oracle_model in oracle_models:
            base_model = oracle_model[: -len("_oracle")]

            # 1) Regular pipeline: pick the model's best (strategy, warm) by hit rate
            chosen = _best_setting_for_model(traj, base_model, tols)
            if chosen is not None:
                s, w, _tu, _sc = chosen
                cfg = (base_model, s, w)
                if cfg in traj and tol in traj[cfg]:
                    iters_ax, Y = traj[cfg][tol]
                    base_picks.setdefault(base_model, (s, w, len(Y)))
                    _draw_curve(
                        ax, iters_ax, Y,
                        color=MODEL_COLORS.get(base_model, "gray"),
                        linestyle="-",
                        marker="o",
                        label=None,
                        uncertainty=uncertainty,
                    )

            # 2) Oracle counterpart — there should be exactly one (s, w) cell
            ocfgs = [(m, s, w) for (m, s, w) in traj if m == oracle_model]
            for (m, s, w) in sorted(ocfgs):
                if tol not in traj[(m, s, w)]:
                    continue
                iters_ax, Y = traj[(m, s, w)][tol]
                oracle_picks.setdefault(base_model, (s, w, len(Y)))
                _draw_curve(
                    ax, iters_ax, Y,
                    color=MODEL_COLORS.get(base_model, "gray"),
                    linestyle=":",
                    marker="*",
                    label=None,
                    uncertainty=uncertainty,
                    linewidth=2.0,
                )

        if prevalence is not None and tol in prevalence:
            ax.axhline(prevalence[tol], color="black", linestyle=":", linewidth=1.4,
                       label=None)
            ax.text(0.99, prevalence[tol], f" {prevalence[tol]:.4f}",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=8, color="black")

    # Model legend (colour key) + Curve legend (regular vs oracle).
    color_handles = []
    for base_model in sorted({m[: -len("_oracle")] for m in oracle_models}):
        # Show "(n=A; oracle n=B)" if both available; degrade otherwise.
        parts = []
        if base_model in base_picks:
            s, w, n = base_picks[base_model]
            parts.append(f"{s}-{w} (n={n})")
        if base_model in oracle_picks:
            s, w, n = oracle_picks[base_model]
            parts.append(f"oracle n={n}")
        label = MODEL_DISPLAY.get(base_model, base_model)
        if parts:
            label += " — " + "; ".join(parts)
        color_handles.append(Line2D(
            [0], [0], color=MODEL_COLORS.get(base_model, "gray"), lw=2.4,
            label=label,
        ))
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  marker="o",
               markersize=5, label="regular"),
        Line2D([0], [0], color="black", linestyle=":",  marker="*",
               markersize=6, lw=2.0, label="oracle"),
    ]
    if prevalence:
        style_handles.append(Line2D(
            [0], [0], color="black", linestyle=":", lw=1.4,
            label=prevalence_label,
        ))

    out_path = out_dir / f"{file_prefix}_oracle_comparison.png"
    _finalize_split_legend(
        fig, axes,
        color_handles=color_handles, style_handles=style_handles,
        out_path=out_path,
        color_title="Model", style_title="Curve",
    )
    written.append(out_path)
    return written


def plot_best_per_model(traj, tols, uncertainty, true_val, out_dir,
                        file_prefix, title_word, ylabel,
                        baseline_traj=None, prevalence=None,
                        prevalence_label="random scan (full pool)"):
    """Single figure: one curve per model using its best (strategy, warm) setting.

    If `baseline_traj` is provided (same {(model, strat, warm): {tol: ...}}
    structure), each picked config also gets a dashed random-baseline curve in
    its model colour. If `prevalence` is provided ({tol: rate}), a horizontal
    grey reference line is drawn on each panel.
    """
    # Oracle (theoretical-limit) variants are folded into per-model plots only;
    # exclude them from the best-per-model headline so the comparison stays
    # apples-to-apples across regular surrogates.
    models = sorted({m for (m, _, _) in traj if not m.endswith("_oracle")})
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
        fontsize=16,
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
                label=None,
                uncertainty=uncertainty,
            )
        if prevalence is not None and tol in prevalence:
            ax.axhline(prevalence[tol], color="black", linestyle=":", linewidth=1.4,
                       label=None)

    # Build split legends: Model (colour key) + Curve (linestyle key).
    color_handles = []
    for (m, s, w, _tu, _sc) in picks:
        n = 0
        for tol in tols:
            if tol in traj[(m, s, w)]:
                n = len(traj[(m, s, w)][tol][1])
                break
        color_handles.append(Line2D(
            [0], [0], color=MODEL_COLORS.get(m, "gray"), lw=2.4, marker="o",
            markersize=5,
            label=f"{MODEL_DISPLAY.get(m, m)} (n={n})",
        ))
    style_handles = []
    if prevalence:
        style_handles.append(Line2D(
            [0], [0], color="black", linestyle=":", lw=1.4,
            label=prevalence_label,
        ))

    out_path = out_dir / f"{file_prefix}_best_per_model.png"
    _finalize_split_legend(
        fig, axes,
        color_handles=color_handles, style_handles=style_handles,
        out_path=out_path,
        color_title=None, style_title="Reference",
    )

    click.echo(f"[best-per-model picks: {title_word}]")
    for (m, s, w, tu, sc) in picks:
        click.echo(f"  {m:12s} -> {s}-{w}  (final mean@{int(tu*100)}% = {sc:.4f})")

    return [out_path], picks


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
@click.option("--compute-accuracy/--no-compute-accuracy", default=False,
              show_default=True,
              help="Compute classification-accuracy trajectories for the "
                   "best-per-model picks on four eval datasets (static random, "
                   "MCMC, per-model train, per-model val). Recomputes from "
                   "saved per-iteration checkpoints; per-run JSON cache makes "
                   "re-renders cheap.")
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True,
              help="ROOT directory holding the MCMC eval set used by "
                   "--compute-accuracy.")
@click.option("--mcmc-max-samples", default=500_000, type=int, show_default=True,
              help="Seeded uniform subsample cap on the MCMC eval set (emcee "
                   "chains store ~96% repeated rows; the subsample preserves "
                   "multiplicity weighting). 0 disables.")
@click.option("--accuracy-device", default=None,
              help="Torch device for accuracy recompute (e.g. cuda:0). "
                   "Default: cuda if available, else cpu.")
@click.option("--accuracy-cache-refresh/--no-accuracy-cache-refresh",
              default=False, show_default=True,
              help="Force re-evaluation of cached iters (overwrites "
                   "<run_dir>/accuracy_trajectory.json entries).")
@click.option("--accuracy-static-eval-size", default=100_000, type=int,
              show_default=True,
              help="Static random eval set size used for --compute-accuracy. "
                   "Must match the AL-driver default (100_000).")
@click.option("--accuracy-dropout", default=0.1, type=float, show_default=True,
              help="Dropout rate used to instantiate the transformer for "
                   "checkpoint loading (matches the AL training default).")
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True,
              help="Post-hoc veto of non-neutralino LSPs (sneutrinos): drop "
                   "training rows whose SP_LSP_type is not in {1,2,3} (i.e. "
                   "F rows with NaN) before computing any metric. n_train_per_iter "
                   "is rebased so per-iteration slicing stays consistent.")
def main(manifest, sweep_id, output_dir, uncertainty, target, tolerances,
         min_seeds, include_status, baseline_data_dir,
         compute_accuracy, mcmc_data_dir, mcmc_max_samples, accuracy_device,
         accuracy_cache_refresh,
         accuracy_static_eval_size, accuracy_dropout,
         require_neutralino_lsp):
    global _REQUIRE_NEUTRALINO_LSP
    _REQUIRE_NEUTRALINO_LSP = bool(require_neutralino_lsp)
    if _REQUIRE_NEUTRALINO_LSP:
        click.echo("[filter] neutralino-LSP veto ENABLED — sneutrino rows will "
                   "be dropped from every run's training set before metric "
                   "computation.", err=True)
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
    prevalence_per_attempt = None
    p_valid = None
    p_valid_info = None
    if baseline_data_dir:
        try:
            Y_full = _load_y_full(baseline_data_dir, target, out_dir)
            prevalence = _pool_prevalence(Y_full, true_val, tols)
            p_valid, n_valid, n_total, log_src = _extract_validity_rate(
                df["expected_run_dir"].dropna().tolist()
            )
            if p_valid is not None:
                prevalence_per_attempt = {t: v * p_valid for t, v in prevalence.items()}
                p_valid_info = {"p_valid": p_valid, "n_valid": n_valid,
                                "n_total": n_total, "source_log": log_src}
                global _DESIRED_P_VALID
                _DESIRED_P_VALID = p_valid
                click.echo(f"[baseline] Run3ModelGen validity (from {log_src}): "
                           f"p_valid={p_valid:.4f} ({n_valid}/{n_total})")
                click.echo(f"[baseline] AL hits/desired initial-chunk denominator "
                           f"now divided by p_valid (was in valid units, now in "
                           f"attempt units to match acquisition increments).")
            else:
                click.echo("[warn] could not parse Run3ModelGen validity rate from any "
                           "active_learning.log; hits_per_desired baseline will fall "
                           "back to raw pool prevalence (over-optimistic for random scan).",
                           err=True)
            with open(out_dir / "random_baseline_prevalence.json", "w") as fh:
                json.dump({"data_dir": baseline_data_dir, "target": target,
                           "true_value": true_val, "n_pool": int(len(Y_full)),
                           "prevalence": {f"{t:.4f}": v for t, v in prevalence.items()},
                           "p_valid": p_valid_info,
                           "prevalence_per_attempt": (
                               {f"{t:.4f}": v for t, v in prevalence_per_attempt.items()}
                               if prevalence_per_attempt else None
                           )},
                          fh, indent=2)
            click.echo(f"[baseline] pool prevalence (n={len(Y_full)}): "
                       + ", ".join(f"tol={int(t*100)}%→{r:.4f}" for t, r in prevalence.items()))
            if prevalence_per_attempt is not None:
                click.echo(f"[baseline] per-attempt rate (× p_valid={p_valid:.4f}): "
                           + ", ".join(f"tol={int(t*100)}%→{r:.4f}"
                                       for t, r in prevalence_per_attempt.items()))
        except Exception as exc:
            click.echo(f"[warn] could not load random baseline pool from {baseline_data_dir}: {exc}",
                       err=True)
            Y_full = None
            prevalence = None
            prevalence_per_attempt = None

    written = []
    picks_by_metric: dict[str, list] = {}
    traj_by_metric: dict[str, dict] = {}
    for metric_name, (traj_fn, file_prefix, ylabel, title_word, traj_fn_baseline) in METRICS.items():
        traj = _collect_trajectories(df, true_val, tols, min_seeds, traj_fn)
        if not traj:
            click.echo(f"[warn] metric '{metric_name}': no groups passed min-seeds filter; skipping",
                       err=True)
            continue
        traj_by_metric[metric_name] = traj
        baseline_traj = None
        if Y_full is not None:
            baseline_traj = _collect_baseline_trajectories(
                df, true_val, tols, min_seeds, traj_fn_baseline, Y_full,
            )
        # For the per-attempt metric, the baseline trajectory comes back in
        # per-valid-sample units (same as the hit_rate panel). Multiply by
        # p_valid to match the AL hits/desired denominator, which already
        # pays for Run3ModelGen failures (~55% of attempts at p_valid=0.445).
        if (metric_name == "hits_per_desired" and baseline_traj is not None
                and p_valid is not None):
            for cfg, per_tol in baseline_traj.items():
                for tol, (it_ax, Y) in per_tol.items():
                    per_tol[tol] = (it_ax, Y * p_valid)
        if metric_name == "hits_per_desired" and prevalence_per_attempt is not None:
            prev_for_metric = prevalence_per_attempt
            prev_label = "random scan (per-attempt)"
        else:
            prev_for_metric = prevalence
            prev_label = "random scan (full pool)"
        written += plot_models_per_strategy(
            traj, tols, uncertainty, true_val, out_dir,
            file_prefix=file_prefix, title_word=title_word, ylabel=ylabel,
            baseline_traj=baseline_traj, prevalence=prev_for_metric,
            prevalence_label=prev_label,
        )
        written += plot_strategies_per_model(
            traj, tols, uncertainty, true_val, out_dir,
            file_prefix=file_prefix, title_word=title_word, ylabel=ylabel,
            baseline_traj=baseline_traj, prevalence=prev_for_metric,
            prevalence_label=prev_label,
        )
        paths, picks = plot_best_per_model(
            traj, tols, uncertainty, true_val, out_dir,
            file_prefix=file_prefix, title_word=title_word, ylabel=ylabel,
            baseline_traj=baseline_traj, prevalence=prev_for_metric,
            prevalence_label=prev_label,
        )
        written += paths
        picks_by_metric[metric_name] = picks
        # Theoretical-limit oracle plot — regular best + matching oracle for
        # each model that has an oracle counterpart in the traj.
        written += plot_oracle_comparison(
            traj, tols, uncertainty, true_val, out_dir,
            file_prefix=file_prefix, title_word=title_word, ylabel=ylabel,
            prevalence=prev_for_metric,
            prevalence_label=prev_label,
        )

    # ── Scan-efficiency improvement: AL → random ratio (per-attempt basis) ───
    if (prevalence_per_attempt is not None
            and "hits_per_desired" in picks_by_metric
            and "hits_per_desired" in traj_by_metric):
        traj_hd = traj_by_metric["hits_per_desired"]
        rows = []
        for (m, s, w, tu, _sc) in picks_by_metric["hits_per_desired"]:
            cfg = (m, s, w)
            for tol in tols:
                if cfg not in traj_hd or tol not in traj_hd[cfg]:
                    continue
                _, Y = traj_hd[cfg][tol]
                al_final = float(np.nanmean(Y, axis=0)[-1])
                rand_rate = prevalence_per_attempt.get(tol)
                if rand_rate is None or rand_rate <= 0:
                    continue
                rows.append({
                    "model": m, "strategy": s, "warm_start": w,
                    "tol": tol,
                    "al_rate_per_attempt": al_final,
                    "random_rate_per_attempt": rand_rate,
                    "speedup": al_final / rand_rate,
                    "n_seeds": int(Y.shape[0]),
                })
        with open(out_dir / "scan_efficiency_improvement.json", "w") as fh:
            json.dump({
                "p_valid": p_valid_info,
                "prevalence_per_attempt": {f"{t:.4f}": v
                                           for t, v in prevalence_per_attempt.items()},
                "rows": rows,
            }, fh, indent=2)
        click.echo("[scan-efficiency] AL (per-attempt) ÷ random (per-attempt) "
                   "at final iteration, best setting per model:")
        for tol in tols:
            tol_rows = [r for r in rows if r["tol"] == tol]
            if not tol_rows:
                continue
            tol_rows.sort(key=lambda r: -r["speedup"])
            line = "  tol={:>3d}%  ".format(int(tol * 100))
            line += " | ".join(f"{r['model']:>12s} {r['speedup']:5.1f}×"
                               for r in tol_rows)
            click.echo(line)

    # ── Classification-accuracy trajectories (opt-in, expensive on first run) ─
    if compute_accuracy:
        picks = picks_by_metric.get("hits_per_desired") or picks_by_metric.get("hit_rate") or []
        if not picks:
            click.echo("[warn] --compute-accuracy: no best-per-model picks "
                       "available; skipping accuracy plots", err=True)
        elif not baseline_data_dir:
            click.echo("[warn] --compute-accuracy needs --baseline-data-dir to "
                       "load X_full; skipping", err=True)
        else:
            try:
                X_full, Y_full_local = _load_xy_full(baseline_data_dir, target, out_dir)
            except Exception as exc:
                click.echo(f"[warn] --compute-accuracy: could not load X_full: {exc}",
                           err=True)
                X_full = None
            if X_full is not None:
                # Static random eval set (seed-123 perm, matches active_learning.py:474)
                # The AL driver reserves the first n_samples=initial_reserved entries
                # of the unshuffled pool for the initial AL data. With n_samples
                # unset the entire pool is reserved -> static set is empty; in
                # practice the sweep always sets n_samples (default 2000), but we
                # don't have it on hand. Fall back to using initial_reserved=0
                # (matches the static-set carving when --n-samples is unset
                # AND the pool exceeds initial_reserved): the seed-123 perm is
                # applied to the full pool, and the first static_eval_size are
                # used. This recovers the same indices the AL driver chose iff
                # initial_reserved was 0; otherwise it disagrees by an offset.
                # To stay correct we mirror the AL default: initial_reserved =
                # n_samples = len(X_full) when --n-samples was not set; but
                # since most sweep runs use n_samples=2000 we pick that.
                initial_reserved = 2000  # AL's default --n-samples
                static_idx = _static_random_indices(
                    len(Y_full_local), initial_reserved,
                    static_eval_size=accuracy_static_eval_size,
                )
                if len(static_idx) == 0:
                    click.echo("[warn] --compute-accuracy: static random eval "
                               "set is empty; skipping", err=True)
                    X_static = Y_static = None
                else:
                    X_static = torch.from_numpy(np.asarray(X_full[static_idx], dtype=np.float32))
                    Y_static = torch.from_numpy(np.asarray(Y_full_local[static_idx], dtype=np.float32))

                # MCMC eval set
                X_mcmc = Y_mcmc = None
                if mcmc_data_dir:
                    try:
                        from pmssm.data import load_mcmc_data  # noqa: PLC0415
                        Xm, Ym = load_mcmc_data(
                            data_dir=mcmc_data_dir,
                            max_samples=mcmc_max_samples or None,
                        )
                        X_mcmc = Xm.float() if hasattr(Xm, "float") else torch.from_numpy(np.asarray(Xm, dtype=np.float32))
                        Y_mcmc = Ym.float().view(-1) if hasattr(Ym, "float") else torch.from_numpy(np.asarray(Ym, dtype=np.float32)).view(-1)
                        click.echo(f"[accuracy] MCMC eval set: n={len(X_mcmc)}")
                        sys.stdout.flush()
                    except Exception as exc:
                        click.echo(f"[warn] --compute-accuracy: MCMC load failed ({exc}); skipping mcmc panel",
                                   err=True)
                        sys.stdout.flush()

                if X_static is None and X_mcmc is None:
                    click.echo("[warn] --compute-accuracy: neither static nor MCMC "
                               "eval set is available; skipping", err=True)
                else:
                    if accuracy_device is None:
                        accuracy_device = "cuda" if torch.cuda.is_available() else "cpu"
                    threshold_t = float(TARGET_CONFIG[target]["threshold"])
                    click.echo(f"[accuracy] device={accuracy_device}, "
                               f"threshold(transformed)={threshold_t}, "
                               f"static_n={0 if X_static is None else len(X_static)}, "
                               f"mcmc_n={0 if X_mcmc is None else len(X_mcmc)}")
                    sys.stdout.flush()
                    # Pad missing eval sets with zero-length tensors so the
                    # worker can iterate uniformly.
                    if X_static is None:
                        X_static = torch.empty(0, X_full.shape[1], dtype=torch.float32)
                        Y_static = torch.empty(0, dtype=torch.float32)
                    if X_mcmc is None:
                        X_mcmc = torch.empty(0, X_full.shape[1], dtype=torch.float32)
                        Y_mcmc = torch.empty(0, dtype=torch.float32)
                    Y_full_arr = np.asarray(Y_full_local)
                    X_full_arr = np.asarray(X_full)
                    # Build synthetic "picks" for oracle cells so the same
                    # _collect_accuracy_trajectories pass also fills traj_acc
                    # entries keyed on (transformer_oracle, …) etc. There's
                    # typically one (strategy, warm) cell per oracle model.
                    oracle_rows = df[df["model"].str.endswith("_oracle")]
                    oracle_picks: list = []
                    for om in sorted(oracle_rows["model"].unique()):
                        om_df = oracle_rows[oracle_rows["model"] == om]
                        # Take the (strategy, warm) cell with the most seed runs.
                        cell_counts = (om_df.groupby(["strategy", "warm_start"])
                                       .size().sort_values(ascending=False))
                        if cell_counts.empty:
                            continue
                        (s, w), _n = list(cell_counts.items())[0]
                        oracle_picks.append((om, s, w, min(tols), float("nan")))
                    if oracle_picks:
                        click.echo(f"[accuracy] including {len(oracle_picks)} oracle "
                                   f"pick(s): "
                                   + ", ".join(f"{m}-{s}-{w}"
                                                for (m, s, w, _, _) in oracle_picks))
                    traj_acc = _collect_accuracy_trajectories(
                        df, picks + oracle_picks, target, X_full_arr, Y_full_arr,
                        X_static, Y_static, X_mcmc, Y_mcmc, threshold_t,
                        device=accuracy_device, min_seeds=min_seeds,
                        refresh=accuracy_cache_refresh,
                        dropout=accuracy_dropout,
                    )
                    if traj_acc:
                        written += plot_classification_accuracy_best_per_model(
                            traj_acc, picks, out_dir, uncertainty,
                        )
                        # Oracle comparison plots — only produced if any
                        # *_oracle entries are present in traj_acc.
                        written += plot_classification_accuracy_oracle_comparison(
                            traj_acc, picks, out_dir, uncertainty,
                        )
                    else:
                        click.echo("[warn] --compute-accuracy: no accuracy "
                                   "trajectories produced", err=True)

    if not written:
        raise click.ClickException("no plots produced — every metric had too few seeds")

    click.echo(f"[plot] wrote {len(written)} file(s) to {out_dir}")
    for p in written:
        click.echo(f"  {p}")


if __name__ == "__main__":
    main()
