"""Recompute verdict accuracy from checkpoints for the classification GP arms.

Why this exists
---------------
The GP driver recorded accuracy at run time by thresholding the model's
prediction against the target in PHYSICAL space. For a Bernoulli deep GP,
``gp_predict`` returned a probability in [0, 1], which inverse-transforms to
exp(p) in [1, e] and so was "excluded" at every point: the cached number was the
positive-class fraction (0.2907 on ExpR), not an accuracy. ``gp_predict`` now
returns the latent for Bernoulli models, whose sign at zero IS the verdict, but
the jobs already running loaded the old code and keep writing the broken value
for the rest of their run.

This closes that gap without restarting them: the per-iteration checkpoints are
on disk, so the accuracy can simply be recomputed and written back into
``accuracy_trajectory.json``, where every existing plotter reads it.

Nothing is reimplemented. The eval set is carved by the paper's own
``_static_random_indices`` (a seed-123 permutation of the post-reserved pool
tail, first ``static_eval_size`` rows), the model is rebuilt by its
``_load_iter_model``, and the metric is its ``_classification_accuracy`` at the
target's transformed threshold. Those helpers had no notion of the acquisition
head until now; they do, so a classification checkpoint rebuilds correctly.

Only the ``al`` role and the ``static_random`` set are recomputed: it is the
only eval set that is common across arms, and the random-additions baseline is
a regression model whose cached accuracy was never affected.

Usage (GPU node)
----------------
    python scripts/deepgp_posthoc_accuracy.py \
        --runs '/ptmp/jwuerzin/output/headtest_deepgp_*_seed1_20260821_*' \
        --data-dir /ptmp/jwuerzin/data/260804 --target ExpR
"""
from __future__ import annotations

import glob as globmod
import json
import sys
import time
from pathlib import Path

import click
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from plot_hit_rate_trajectories_multiseed import (      # noqa: E402
    _classification_accuracy, _load_iter_model, _load_xy_full,
    _parse_run_kwargs_from_log, _static_random_indices,
)
from pmssm.data import TARGET_CONFIG, transform_y       # noqa: E402
from pmssm.visualization import gp_predict              # noqa: E402


def _al_train_val(state: dict, k: int):
    """The (X, Y) train/val slices the run held after iteration k+1."""
    ntr = list(state["al_n_train"])
    nva = list(state["al_n_val"])
    t, v = int(ntr[k]), int(nva[k])
    return (state["X"][:t], state["Y"][:t].view(-1),
            state["X_val"][:v], state["Y_val"][:v].view(-1))


@click.command()
@click.option("--runs", required=True, help="Glob of run directories.")
@click.option("--data-dir", default="/ptmp/jwuerzin/data/260804", show_default=True)
@click.option("--target", default="ExpR", show_default=True)
@click.option("--pool-cache-dir", default="/ptmp/jwuerzin/analysis/pool_cache",
              show_default=True)
@click.option("--static-eval-size", default=100_000, show_default=True)
@click.option("--initial-reserved", default=2000, show_default=True,
              help="The run's --n-samples; the eval set is carved after it.")
@click.option("--device", default="cuda", show_default=True)
@click.option("--max-eval", default=0, show_default=True,
              help="Cap the eval set for a smoke run; 0 uses all of it.")
@click.option("--dry-run/--write", default=False,
              help="With --dry-run nothing is written back to the run's cache.")
def main(runs, data_dir, target, pool_cache_dir, static_eval_size,
         initial_reserved, device, max_eval, dry_run):
    cache_dir = Path(pool_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"[pool] loading {data_dir} ({target})")
    X_full, Y_full = _load_xy_full(data_dir, target, cache_dir)
    idx = _static_random_indices(len(Y_full), initial_reserved,
                                 static_eval_size=static_eval_size)
    if max_eval:
        idx = idx[:max_eval]
    X_static = torch.from_numpy(np.asarray(X_full[idx], dtype=np.float32))
    Y_static = torch.from_numpy(np.asarray(Y_full[idx], dtype=np.float32))
    thr = float(TARGET_CONFIG[target]["threshold"])
    y_true_t = transform_y(Y_static, target=target).view(-1).numpy()
    click.echo(f"[eval] static_random: {len(idx)} rows, threshold {thr:g}, "
               f"positive fraction {float((y_true_t >= thr).mean()):.4f}")

    from pmssm.data import build_norm_tensors, normalize_x
    data_min, data_max = build_norm_tensors()
    X_static_norm = normalize_x(X_static, data_min, data_max)

    for run_dir in sorted(globmod.glob(runs)):
        run_dir = Path(run_dir)
        state_path = run_dir / "state.pt"
        if not state_path.exists():
            click.echo(f"[skip] {run_dir.name}: no state.pt")
            continue
        state = torch.load(state_path, weights_only=False, map_location="cpu")
        run_kwargs = _parse_run_kwargs_from_log(run_dir)
        model_type = str(run_kwargs.get("model_type", "deep_gp"))
        head = str(run_kwargs.get("head", "regression"))
        n_iters = len(list(state["al_n_train"]))
        click.echo(f"\n[run] {run_dir.name}\n       model_type={model_type} "
                   f"head={head} iters={n_iters}")

        results: dict[int, float] = {}
        for k in range(n_iters):
            iter_dir = run_dir / f"iteration_{k + 1:03d}"
            if not (iter_dir / "al_model_checkpoint.pt").exists():
                continue
            Xtr, Ytr, Xva, Yva = _al_train_val(state, k)
            t0 = time.time()
            try:
                model = _load_iter_model(model_type, "al", iter_dir,
                                         Xtr, Ytr, Xva, Yva, run_kwargs, device)
            except Exception as exc:
                click.echo(f"  iter {k+1:>3}: rebuild failed: "
                           f"{type(exc).__name__}: {str(exc)[:110]}")
                continue
            if model is None:
                continue
            try:
                pred = gp_predict(model, X_static_norm, model_type,
                                  num_samples=int(run_kwargs.get("gp_num_samples", 8) or 8))
                acc = _classification_accuracy(
                    np.asarray(pred, dtype=np.float64), y_true_t, thr)
            except Exception as exc:
                click.echo(f"  iter {k+1:>3}: predict failed: "
                           f"{type(exc).__name__}: {str(exc)[:110]}")
                continue
            finally:
                del model
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            results[k + 1] = acc
            click.echo(f"  iter {k+1:>3}: accuracy {acc:.4f}   "
                       f"({time.time() - t0:.1f}s)")

        if not results:
            click.echo("  nothing recomputed")
            continue
        # Authoritative output: a file the RUNNING job never writes.
        #
        # Writing only into accuracy_trajectory.json cannot work while the job
        # is alive. It appends a fresh run-time value on every iteration, so
        # each pass is overtaken within minutes and the artefact creeps back
        # (observed: clsent iterations 14 and 15 back at 0.291 shortly after a
        # clean pass). Consumers that know about this file get a series which
        # only ever contains checkpoint-derived numbers.
        posthoc_path = run_dir / "accuracy_posthoc.json"
        posthoc = {}
        if posthoc_path.exists():
            try:
                posthoc = json.loads(posthoc_path.read_text())
            except Exception:
                posthoc = {}
        series = posthoc.setdefault("static_random", {})
        series.update({str(it): float(acc) for it, acc in results.items()})
        posthoc["meta"] = {
            "source": "scripts/deepgp_posthoc_accuracy.py",
            "why": "run-time accuracy for a verdict head thresholded a "
                   "probability against the physical target, giving the "
                   "positive-class fraction instead of an accuracy",
            "model_type": model_type, "head": head, "target": target,
            "eval": f"static_random, {len(idx)} rows",
        }
        if not dry_run:
            posthoc_path.write_text(json.dumps(posthoc, indent=1))
            click.echo(f"  [write] {posthoc_path} "
                       f"({len(series)} iterations, artefact-proof)")

        cache_path = run_dir / "accuracy_trajectory.json"
        cache = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except Exception:
                cache = {}
        # Every run-time value in these runs is the positive-class fraction, so
        # any iteration this pass could NOT recompute must be removed rather
        # than left in place. Otherwise the plot shows a recomputed curve that
        # falls off a cliff the moment it meets an iteration the still-running
        # job appended after this pass -- which reads as a result rather than as
        # a stale cache entry. Re-running this script later simply extends the
        # trustworthy range.
        # ...but ONLY for a verdict head. A regression arm's run-time accuracy
        # was always correct, so deleting its newest entry would discard good
        # data for no reason; recomputing the ones we can is still fine, since
        # the two agree.
        drop_stale = head != "regression"
        dropped = 0
        for key, entry in cache.items():
            if not drop_stale:
                break
            if not str(key).isdigit():
                continue
            it = int(key)
            role = entry.get("al")
            if not isinstance(role, dict) or "static_random" not in role:
                continue
            if it in results:
                role["static_random"] = float(results[it])
            else:
                del role["static_random"]
                dropped += 1
        for it, acc in results.items():
            entry = cache.setdefault(str(it), {})
            role = entry.setdefault("al", {})
            role["static_random"] = float(acc)
        if dropped:
            click.echo(f"  dropped {dropped} un-recomputed iteration(s) "
                       "whose cached value is the run-time artefact")
        if dry_run:
            click.echo(f"  [dry-run] {len(results)} iterations NOT written")
        else:
            cache_path.write_text(json.dumps(cache, indent=1))
            click.echo(f"  [write] {cache_path} ({len(results)} iterations "
                       "updated in place)")


if __name__ == "__main__":
    main()
