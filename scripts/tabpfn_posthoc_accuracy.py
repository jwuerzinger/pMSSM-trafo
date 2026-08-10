"""Post-hoc classification accuracy for TabPFN against the CURRENT references.

Why this exists
---------------
TabPFN v2 is frozen and in-context: the AL driver saves no per-iteration weight
file, so ``plot_hit_rate_trajectories_multiseed.py`` cannot recompute its
accuracy from checkpoints and instead harvests the run-time
``accuracy_trajectory.json`` verbatim (its ``cache_only`` branch). Those cached
MCMC numbers were scored against whatever snapshot of the emcee reference the
RUN happened to load, so as soon as the reference grows they stop being
comparable with the five checkpointed surrogates.

This script closes that gap without re-running AL: it re-fits the frozen TabPFN
in-context on each seed's stored AL (or random-baseline) training set and scores
it on eval sets built exactly the way the accuracy plots build them.

Reuse (nothing is reimplemented)
--------------------------------
* ``evaluate_uq._al_train_val``   — the iteration's stored (train, val) slices
* ``evaluate_uq._predict_tabpfn`` — the in-context refit + batched prediction
  path already used by the UQ evaluation; ``pred["mean"]`` is TabPFN's
  ``predict(X)`` point prediction in transformed space
* ``plot_hit_rate_trajectories_multiseed._classification_accuracy`` — the
  sign-of-constraint accuracy, thresholded at
  ``TARGET_CONFIG[target]["threshold"]`` (= 0 in log space, i.e. Omega > 0.12);
  this is NOT the +/-10% tolerance band
* ``plot_hit_rate_trajectories_multiseed._static_random_indices`` /
  ``_load_xy_full`` / ``_seed_perm`` / ``_baseline_iter_xy`` — eval-set carving
  and random-baseline training-set reconstruction
* ``pmssm.data.load_mcmc_data`` — seeded uniform subsample of the reference
  (default 500k rows, ``subsample_seed=42``), same call the plot script makes

Results are cached per run dir in ``tabpfn_posthoc_accuracy.json`` under a key
that carries the reference's current row count, so a grown reference can never
hit a stale entry.

Usage (GPU node; TabPFN needs a GPU and TABPFN_TOKEN):
    python scripts/tabpfn_posthoc_accuracy.py --output /ptmp/.../tabpfn_acc.json
Smoke test:
    python scripts/tabpfn_posthoc_accuracy.py --seeds 1 --mcmc-max-samples 20000 \
        --datasets mcmc --roles al
"""
from __future__ import annotations

import csv
import glob
import json
import sys
import time
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE_NAME = "tabpfn_posthoc_accuracy.json"
DATASETS = ("mcmc", "static_random")
ROLES = ("al", "baseline")


# ──────────────────────────────────────────────────────────────────────────────
# Eval sets
# ──────────────────────────────────────────────────────────────────────────────

def _mcmc_source_entries(data_dir: str) -> int:
    """Total rows in the reference's ROOT files (header read only, ~1 s).

    Used as a provenance tag: it changes whenever the emcee run is resumed, so
    both the .npy eval-set cache and the per-run accuracy cache key on it and a
    longer reference can never be scored with a stale cached number.
    """
    import uproot
    n = 0
    for f in sorted(glob.glob(f"{data_dir}/*.root")):
        n += int(uproot.open(f)["susy"].num_entries)
    return n


def _load_mcmc_eval(data_dir: str, target: str, max_samples: int, veto: bool,
                    cache_dir: Path, refresh: bool = False):
    """Seeded subsample of the reference, matching the accuracy plots' call.

    Returns (X float32 (N,19), Y float32 (N,), n_source_rows, from_cache).
    """
    n_src = _mcmc_source_entries(data_dir)
    safe = str(data_dir).replace("/", "_").strip("_")
    tag = f"{safe}_{target}_veto{int(veto)}_n{int(max_samples)}_src{n_src}"
    xp = cache_dir / f"mcmc_eval_x_{tag}.npy"
    yp = cache_dir / f"mcmc_eval_y_{tag}.npy"
    if xp.exists() and yp.exists() and not refresh:
        return np.load(xp), np.load(yp), n_src, True

    from pmssm.data import load_mcmc_data
    Xm, Ym = load_mcmc_data(data_dir=data_dir, target=target,
                            require_neutralino_lsp=veto,
                            max_samples=int(max_samples) or None)
    X = np.asarray(Xm.numpy(), dtype=np.float32)
    Y = np.asarray(Ym.numpy(), dtype=np.float32).ravel()
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(xp, X)
    np.save(yp, Y)
    return X, Y, n_src, False


# ──────────────────────────────────────────────────────────────────────────────
# Per-run evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _iter_list(spec: str, n_iters: int) -> list[int]:
    if spec.strip() in ("last", "final"):
        return [n_iters]
    return sorted({int(t) for t in spec.split(",") if t.strip()})


def _eval_run(run_dir: str, seed: int, eval_sets: dict, roles: tuple,
              iters_spec: str, device: str, target: str, threshold_t: float,
              X_full, Y_full, refresh: bool, cache_tag: str) -> dict:
    """{iter_no: {role: {ds: accuracy}}} for one seed run."""
    import torch
    import evaluate_uq as euq
    import plot_hit_rate_trajectories_multiseed as phr

    run_dir_p = Path(run_dir)
    state_path = run_dir_p / "state.pt"
    if not state_path.exists():
        click.echo(f"[tabpfn-acc]   skip (no state.pt): {run_dir}", err=True)
        return {}
    state = torch.load(state_path, weights_only=False, map_location="cpu")

    cache_path = run_dir_p / CACHE_NAME
    cache = {}
    if cache_path.exists() and not refresh:
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    n_iters = len(list(state.get("al_n_train") or []))
    X_full_shuf = Y_full_shuf = None
    out: dict = {}
    for iter_no in _iter_list(iters_spec, n_iters):
        if not (1 <= iter_no <= n_iters):
            click.echo(f"[tabpfn-acc]   iter {iter_no} out of range (n={n_iters})",
                       err=True)
            continue
        idx = iter_no - 1
        X_tr, Y_tr, _X_va, _Y_va = euq._al_train_val(state, idx)

        for role in roles:
            if role == "al":
                Xt, Yt = X_tr, Y_tr
            else:
                # Random-baseline training set at this iteration: the shared
                # initial split plus the seed-shuffled baseline_add_indices.
                if X_full_shuf is None:
                    perm = phr._seed_perm(len(Y_full), seed)
                    X_full_shuf = np.asarray(X_full)[perm]
                    Y_full_shuf = np.asarray(Y_full)[perm]
                base = phr._baseline_iter_xy(state, X_full_shuf, Y_full_shuf,
                                            idx, role="train")
                if base is None:
                    click.echo(f"[tabpfn-acc]   iter {iter_no}/{role}: could not "
                               "reconstruct baseline train set", err=True)
                    continue
                bX, bY = base
                Xt = torch.from_numpy(np.asarray(bX, dtype=np.float32))
                Yt = torch.from_numpy(np.asarray(bY, dtype=np.float32)).view(-1)
            if len(Xt) == 0:
                continue

            for ds in eval_sets:
                key = f"v1|{cache_tag}|iter{iter_no}|{role}|{ds}"
                if key in cache and not refresh:
                    out.setdefault(iter_no, {}).setdefault(role, {})[ds] = cache[key]
                    continue
                X_ev, y_true_t = eval_sets[ds]
                t0 = time.time()
                try:
                    pred = euq._predict_tabpfn(Xt, Yt, X_ev, device, target)
                except Exception as exc:
                    click.echo(f"[tabpfn-acc]   iter {iter_no}/{role}/{ds}: refit "
                               f"failed: {exc}", err=True)
                    continue
                mean_t = np.asarray(pred["mean"], dtype=np.float64).ravel()
                acc = phr._classification_accuracy(mean_t, y_true_t, threshold_t)
                rec = {
                    "accuracy": float(acc),
                    "n_eval": int(len(y_true_t)),
                    "n_train": int(len(Xt)),
                    "rmse": float(np.sqrt(((y_true_t - mean_t) ** 2).mean())),
                    "family": pred.get("family"),
                    "seconds": float(time.time() - t0),
                }
                cache[key] = rec
                out.setdefault(iter_no, {}).setdefault(role, {})[ds] = rec
                click.echo(f"[tabpfn-acc]   seed {seed} iter {iter_no:>3} {role:<8} "
                           f"{ds:<13} n_train={rec['n_train']:>6} "
                           f"n_eval={rec['n_eval']:>7} acc={acc:.4f} "
                           f"rmse={rec['rmse']:.4f} ({rec['seconds']:5.1f}s)")
                sys.stdout.flush()
                tmp = cache_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(cache, indent=1, sort_keys=True))
                tmp.replace(cache_path)
    return out


def _runtime_cached(run_dir: str, iter_no: int) -> dict:
    """The run-time accuracy_trajectory.json entry (the stale reference), for
    side-by-side reporting only."""
    p = Path(run_dir) / "accuracy_trajectory.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    return d.get(str(iter_no)) or {}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--model", default="tabpfn", show_default=True)
@click.option("--strategy", default="top_k", show_default=True)
@click.option("--warm", "warm_start", default="tabpfn", show_default=True,
              help="warm_start tag of the cell (TabPFN's cell is tagged 'tabpfn').")
@click.option("--include-status", default="completed", show_default=True)
@click.option("--seeds", default=None, help="Comma list (default: all in the cell).")
@click.option("--iters", default="last", show_default=True,
              help="'last' or a comma list of 1-based iteration numbers.")
@click.option("--roles", default="al,baseline", show_default=True,
              help="Comma list from al,baseline.")
@click.option("--datasets", default="mcmc,static_random", show_default=True,
              help="Comma list from mcmc,static_random.")
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True)
@click.option("--mcmc-max-samples", default=500_000, type=int, show_default=True,
              help="Seeded uniform subsample cap on the MCMC eval set.")
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358",
              show_default=True)
@click.option("--static-eval-size", default=100_000, type=int, show_default=True,
              help="Static-random eval size (matches --accuracy-static-eval-size).")
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True)
@click.option("--target", default="DMRD", show_default=True)
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp", default=False,
              show_default=True)
@click.option("--device", default=None, help="cuda/cpu (default: auto).")
@click.option("--output", default="/ptmp/jwuerzin/analysis/all_runs/tabpfn_posthoc_accuracy.json",
              show_default=True)
@click.option("--refresh", is_flag=True, default=False)
@click.option("--build-eval-only", is_flag=True, default=False,
              help="Build/refresh the MCMC eval-set .npy cache and exit (no GPU).")
def main(manifest, model, strategy, warm_start, include_status, seeds, iters,
         roles, datasets, mcmc_data_dir, mcmc_max_samples, baseline_data_dir,
         static_eval_size, cache_dir, target, require_neutralino_lsp, device,
         output, refresh, build_eval_only):
    import torch
    import plot_hit_rate_trajectories_multiseed as phr
    from pmssm import TARGET_CONFIG
    from pmssm.data import transform_y

    cache_dir_p = Path(cache_dir)
    ds_wanted = tuple(d.strip() for d in datasets.split(",") if d.strip())
    role_wanted = tuple(r.strip() for r in roles.split(",") if r.strip())
    for d in ds_wanted:
        if d not in DATASETS:
            raise click.ClickException(f"unknown dataset: {d}")
    for r in role_wanted:
        if r not in ROLES:
            raise click.ClickException(f"unknown role: {r}")

    threshold_t = float(TARGET_CONFIG[target]["threshold"])

    # ── eval sets ────────────────────────────────────────────────────────────
    eval_sets: dict = {}
    n_src = None
    if "mcmc" in ds_wanted or build_eval_only:
        Xm, Ym, n_src, cached = _load_mcmc_eval(
            mcmc_data_dir, target, mcmc_max_samples, require_neutralino_lsp,
            cache_dir_p, refresh=refresh)
        click.echo(f"[tabpfn-acc] MCMC eval: n={len(Ym)} from {mcmc_data_dir} "
                   f"(source rows={n_src}, "
                   f"{'npy cache' if cached else 'freshly loaded'})")
        if "mcmc" in ds_wanted:
            eval_sets["mcmc"] = (
                torch.from_numpy(Xm),
                transform_y(torch.from_numpy(Ym), target=target)
                .numpy().ravel().astype(np.float64),
            )
    if build_eval_only:
        click.echo("[tabpfn-acc] --build-eval-only: done")
        return

    X_full = Y_full = None
    if "static_random" in ds_wanted or "baseline" in role_wanted:
        X_full, Y_full = phr._load_xy_full(baseline_data_dir, target, cache_dir_p)
    if "static_random" in ds_wanted:
        # Same carving as the accuracy plots: seed-123 perm of the pool tail
        # after the AL driver's reserved head (--n-samples default 2000).
        static_idx = phr._static_random_indices(len(Y_full), 2000,
                                                static_eval_size=static_eval_size)
        Xs = np.asarray(np.asarray(X_full)[static_idx], dtype=np.float32)
        Ys = np.asarray(np.asarray(Y_full)[static_idx], dtype=np.float32).ravel()
        eval_sets["static_random"] = (
            torch.from_numpy(Xs),
            transform_y(torch.from_numpy(Ys), target=target)
            .numpy().ravel().astype(np.float64),
        )
        click.echo(f"[tabpfn-acc] static_random eval: n={len(Ys)} from "
                   f"{baseline_data_dir}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"[tabpfn-acc] device={device}, threshold(transformed)={threshold_t}, "
               f"roles={role_wanted}, datasets={tuple(eval_sets)}")

    # ── manifest rows ────────────────────────────────────────────────────────
    statuses = {s.strip() for s in include_status.split(",")}
    seed_filter = {int(s) for s in seeds.split(",")} if seeds else None
    rows = [r for r in csv.DictReader(open(manifest))
            if r["status"] in statuses
            and (r["model"], r["strategy"], r["warm_start"]) == (model, strategy, warm_start)
            and (seed_filter is None or int(r["seed"]) in seed_filter)]
    if not rows:
        raise click.ClickException(
            f"no manifest rows for {model}/{strategy}/{warm_start} "
            f"(status in {sorted(statuses)})")
    click.echo(f"[tabpfn-acc] {model}/{strategy}/{warm_start}: {len(rows)} seed run(s)")

    cache_tag = (f"n_mcmc{mcmc_max_samples}|n_static{static_eval_size}|"
                 f"veto{int(require_neutralino_lsp)}|src{n_src}")

    per_seed: dict = {}
    runtime_ref: dict = {}
    for r in rows:
        seed = int(r["seed"])
        rd = r["expected_run_dir"]
        click.echo(f"[tabpfn-acc]  seed {seed}: {rd}")
        sys.stdout.flush()
        res = _eval_run(rd, seed, eval_sets, role_wanted, iters, device, target,
                        threshold_t, X_full, Y_full, refresh, cache_tag)
        if res:
            per_seed[seed] = res
            for it in res:
                runtime_ref.setdefault(seed, {})[it] = _runtime_cached(rd, it)

    if not per_seed:
        raise click.ClickException("no seed produced a result")

    # ── aggregate: mean +/- SEM over seeds, per (iteration, role, dataset) ───
    agg: dict = {}
    all_iters = sorted({it for res in per_seed.values() for it in res})
    for it in all_iters:
        for role in role_wanted:
            for ds in eval_sets:
                vals = [res[it][role][ds]["accuracy"]
                        for res in per_seed.values()
                        if it in res and role in res[it] and ds in res[it][role]]
                if not vals:
                    continue
                v = np.asarray(vals, dtype=np.float64)
                stale = [runtime_ref.get(s, {}).get(it, {}).get(role, {}).get(ds)
                         for s, res in per_seed.items()
                         if it in res and role in res[it] and ds in res[it][role]]
                stale = [x for x in stale if isinstance(x, (int, float))]
                entry = {
                    "n_seeds": int(len(v)),
                    "mean": float(v.mean()),
                    "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                    "per_seed": {str(s): float(res[it][role][ds]["accuracy"])
                                 for s, res in per_seed.items()
                                 if it in res and role in res[it] and ds in res[it][role]},
                }
                if stale:
                    sv = np.asarray(stale, dtype=np.float64)
                    entry["runtime_cache_mean"] = float(sv.mean())
                    entry["runtime_cache_sem"] = (
                        float(sv.std(ddof=1) / np.sqrt(len(sv))) if len(sv) > 1 else 0.0)
                agg.setdefault(str(it), {}).setdefault(role, {})[ds] = entry

    payload = {
        "config": {
            "manifest": manifest, "cell": [model, strategy, warm_start],
            "include_status": sorted(statuses), "iters": iters,
            "roles": list(role_wanted), "datasets": list(eval_sets),
            "mcmc_data_dir": mcmc_data_dir,
            "mcmc_max_samples": mcmc_max_samples,
            "mcmc_source_rows": n_src,
            "baseline_data_dir": baseline_data_dir,
            "static_eval_size": static_eval_size,
            "target": target, "threshold_transformed": threshold_t,
            "require_neutralino_lsp": require_neutralino_lsp,
            "device": device,
            "method": "post-hoc in-context refit (evaluate_uq._predict_tabpfn)",
        },
        "aggregate": agg,
        "per_seed": {str(s): {str(it): v for it, v in res.items()}
                     for s, res in per_seed.items()},
    }
    outp = Path(output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=1))
    click.echo(f"[tabpfn-acc] wrote {outp}")

    click.echo("\n[tabpfn-acc] ── refit accuracy (mean +/- SEM over seeds) ──")
    click.echo(f"{'iter':>5} {'role':<9} {'dataset':<14} {'n':>3} "
               f"{'refit':>16}   {'run-time cache':>16}")
    for it in sorted(agg, key=int):
        for role in agg[it]:
            for ds, e in agg[it][role].items():
                stale = (f"{e['runtime_cache_mean']:.4f} +/- {e['runtime_cache_sem']:.4f}"
                         if "runtime_cache_mean" in e else "n/a")
                click.echo(f"{it:>5} {role:<9} {ds:<14} {e['n_seeds']:>3} "
                           f"{e['mean']:.4f} +/- {e['sem']:.4f}   {stale:>16}")


if __name__ == "__main__":
    main()
