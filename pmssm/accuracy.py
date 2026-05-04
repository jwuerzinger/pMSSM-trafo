"""In-training classification-accuracy capture.

Each AL iteration computes binary classification accuracy at the constraint
threshold for the AL and Baseline models on every available eval dataset
(``static_random``, ``mcmc``, ``train``, ``val``) and writes the results to
``<run_dir>/accuracy_trajectory.json``. The schema matches the cache produced
post-hoc by ``scripts/plot_hit_rate_trajectories_multiseed.py``, so when that
script later runs to render the multi-seed accuracy plots, every iteration
already has a cache entry and no checkpoint reload / re-inference is needed.

Schema (matches `_load_accuracy_cache` in the offline script)::

    {
      "1": {
        "al":       {"static_random": 0.93, "mcmc": 0.91, "train": 0.95, "val": 0.92},
        "baseline": {"static_random": 0.78, ...}
      },
      "2": { ... }
    }
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import torch


CACHE_FILENAME = "accuracy_trajectory.json"


def binary_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor,
                    threshold: float) -> float:
    """Binary classification accuracy at ``threshold``.

    Comparison is invariant under any monotonic transform applied to both
    tensors, so caller may pass physical or transformed quantities as long as
    ``threshold`` is in the same space.
    """
    y_true = y_true.detach().reshape(-1).cpu()
    y_pred = y_pred.detach().reshape(-1).cpu()
    if len(y_true) == 0 or len(y_pred) == 0:
        return float("nan")
    return float(((y_pred >= threshold) == (y_true >= threshold)).float().mean().item())


def _load(run_dir: Path) -> dict:
    p = run_dir / CACHE_FILENAME
    if not p.exists():
        return {}
    try:
        with open(p) as fh:
            d = json.load(fh)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _atomic_save(run_dir: Path, data: dict) -> None:
    p = run_dir / CACHE_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, p)


def update_accuracy_trajectory(run_dir: str | os.PathLike, iteration: int,
                               role: str,
                               dataset_accs: Mapping[str, float]) -> None:
    """Merge ``dataset_accs`` into ``accuracy_trajectory.json`` for one (iter, role).

    Read-modify-write: only the specific ``(iteration, role, dataset)`` keys
    passed in are touched — every other entry already in the file is preserved
    through the merge. Atomic against partial writes via a temp file + rename.

    Idempotent: re-running with overlapping keys overwrites with the new value
    (correct on resume / re-run; same iteration with the same model state should
    yield the same accuracy).

    The cache file is per-run-directory: ``<run_dir>/accuracy_trajectory.json``.
    Distinct AL runs always have distinct ``run_dir`` values, so two sweeps
    cannot collide on the same file.
    """
    run_dir_p = Path(run_dir)
    cache = _load(run_dir_p)
    key = str(int(iteration))
    iter_entry = cache.get(key) or {}
    if not isinstance(iter_entry, dict):
        iter_entry = {}
    role_entry = iter_entry.get(role) or {}
    if not isinstance(role_entry, dict):
        role_entry = {}
    for ds, acc in dataset_accs.items():
        if acc is None:
            continue
        try:
            f = float(acc)
        except (TypeError, ValueError):
            continue
        if f != f:  # NaN check without numpy
            continue
        role_entry[ds] = f
    iter_entry[role] = role_entry
    cache[key] = iter_entry
    _atomic_save(run_dir_p, cache)


def write_iter_accuracies(run_dir: str | os.PathLike, iteration: int,
                          al_accs: Mapping[str, float] | None = None,
                          baseline_accs: Mapping[str, float] | None = None) -> None:
    """Convenience: write both roles in one call (skipping any None entry)."""
    if al_accs:
        update_accuracy_trajectory(run_dir, iteration, "al", al_accs)
    if baseline_accs:
        update_accuracy_trajectory(run_dir, iteration, "baseline", baseline_accs)
