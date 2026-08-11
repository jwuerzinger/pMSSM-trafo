"""Does ensemble disagreement acquire better than MC-dropout spread?

Table~\\ref{tab:uq} leaves the dropout surrogates with a weak acquisition
uncertainty, and the obvious suspicion is that dropout is the wrong estimator:
its variance comes from Bernoulli masks applied to one trained optimum, with a
scale set by a regularisation hyperparameter. A deep ensemble replaces that with
the disagreement of K independently initialised optima, which answers the
epistemic question directly.

This scores that substitution on the metric that pays for itself, hits per
simulator ATTEMPT, at matched iteration and matched strategy. Both arms use
top_k, because an ensemble's across-member covariance has rank at most K-1 and
entropy_batch's batch score is degenerate below the batch size.

The compute accounting needs care. Ensemble members write their training logs as
al_m<k>_training.log, so a parser that reads only al_training.log sees one
member and undercounts the arm by nearly K. Both are summed here.

Usage:
    python scripts/ensemble_comparison.py --iteration 10
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Run3ModelGen's validity rate: hits/desired is per ATTEMPT, so the random
# baseline must be deflated by it rather than using raw pool prevalence.
P_VALID = 0.445414
_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_SEL_START = re.compile(r"Running \d+ MC Dropout forward passes"
                        r"|Ensemble uncertainty over"
                        r"|Generating \d+ candidate points"
                        r"|Select(?:ing|ed) .*top[_-]?k", re.IGNORECASE)
_SEL_END = re.compile(r"Saved selected points to")

MODELS = ["transformer", "dnn", "dnn_match_trafo"]
DISPLAY = {"transformer": "Transformer", "dnn": "DNN",
           "dnn_match_trafo": "DNN (matched)"}


def _random_baseline(prevalence_json: Path, tolerance: float) -> float:
    """Random-scan hits per ATTEMPT at this tolerance, as measured on the pool.

    Must be read rather than assumed. The per-attempt rate is the in-band
    prevalence among VALID models times p_valid (0.009237 x 0.445414 = 0.004114
    at tolerance 0.1); taking prevalence to be a round 1% instead understates
    every reported multiple by about 9%.
    """
    d = json.loads(prevalence_json.read_text())
    per_attempt = d.get("prevalence_per_attempt", {})
    key = f"{tolerance:.4f}"
    if key in per_attempt:
        return float(per_attempt[key])
    prev = float(d["prevalence"][key])
    return prev * float(d["p_valid"]["p_valid"])


def _log_span(path: Path) -> float:
    """Seconds between the first and last timestamped line."""
    if not path.exists():
        return 0.0
    first = last = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _TS.match(line)
            if m:
                t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                first = first or t
                last = t
    return (last - first).total_seconds() if first and last else 0.0


def _train_seconds(iter_dir: Path) -> float:
    """Training seconds for an iteration, counting EVERY ensemble member.

    Members log to al_m<k>_training.log; reading only al_training.log
    undercounts a K-member arm by close to K and would make the ensemble look
    cheaper than the dropout it is being compared against.
    """
    total = _log_span(iter_dir / "al_training.log")
    for m in sorted(iter_dir.glob("al_m*_training.log")):
        total += _log_span(m)
    return total


def _selection_seconds(main_log: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not main_log.exists():
        return out
    it = None
    start = None
    with open(main_log, errors="replace") as fh:
        for line in fh:
            m = re.search(r"Iteration (\d+) ===", line)
            if m:
                it, start = int(m.group(1)), None
                continue
            ts = _TS.match(line)
            t = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S") if ts else None
            if t is None or it is None:
                continue
            if start is None and _SEL_START.search(line):
                start = t
            elif start is not None and _SEL_END.search(line):
                out[it] = out.get(it, 0.0) + (t - start).total_seconds()
                start = None
    return out


def _arm_stats(run_dir: Path, iteration: int, tolerance: float, target: float,
               mu_window: tuple[float, float]):
    """Yield, cost and composition for one run at a given iteration."""
    import torch
    from analyse_runs import load_run

    state = torch.load(run_dir / "state.pt", weights_only=False, map_location="cpu")
    n_tr = list(state.get("al_n_train") or [])
    n_va = list(state.get("al_n_val") or [])
    if len(n_tr) < iteration:
        return None
    L = int(n_tr[iteration - 1]) + int(n_va[iteration - 1])

    # ---- yield: hits per ATTEMPT and per valid model ------------------------
    Y = np.asarray(load_run(str(run_dir)).Y).ravel()[:L]
    hits = int((np.abs(Y - target) / target < tolerance).sum())
    hit_rate = hits / max(1, len(Y))                    # per valid model
    attempts = len(Y) / P_VALID                          # per simulator call
    hits_desired = hits / attempts

    # ---- cost: training over all members, plus selection -------------------
    sel = _selection_seconds(run_dir / "active_learning.log")
    secs = 0.0
    for i in range(1, iteration + 1):
        d = run_dir / f"iteration_{i:03d}"
        if d.exists():
            secs += _train_seconds(d) + sel.get(i, 0.0)

    return {"L": L, "hits": hits, "hit_rate": hit_rate,
            "hits_desired": hits_desired, "gpu_hours": secs / 3600.0}


def _composition(run_dirs, iteration, tolerance, target):
    """Pooled SPheno composition of in-band points, same definition as tab:composition.

    Reuses composition_fractions' reader and pmssm.visualization's classifier
    rather than re-deriving either, so these rows are directly comparable with
    the main table instead of merely looking like it.
    """
    from composition_fractions import _al_cell_from_ntuples, sanitize_spheno_fracs
    from pmssm.visualization import LSP_TYPE_NAMES, classify_lsp_type

    om, fr, _n_seeds, n_files = _al_cell_from_ntuples(
        [str(d) for d in run_dirs], max_iter=iteration)
    if om is None:
        return None
    fr = sanitize_spheno_fracs(np.asarray(fr, dtype=np.float64))
    keep = np.abs(np.asarray(om, dtype=np.float64) - target) / target < tolerance
    if keep.sum() == 0:
        return None
    # classify_lsp_type returns integer CODES (0 bino, 1 wino, 2 higgsino,
    # 3 mixed, -1 non-neutralino/missing), not names. Comparing against strings
    # silently yields all-zero fractions, so map through LSP_TYPE_NAMES and
    # normalise over the classified rows only, matching tab:composition's
    # convention of excluding non-neutralino LSPs from the four columns.
    codes = np.asarray(classify_lsp_type(fr[keep]))
    n_in = int(keep.sum())
    n_cls = int((codes >= 0).sum())
    out = {"n": n_in, "n_classified": n_cls, "n_ntuple_files": int(n_files),
           "no_neutralino_lsp": float(n_in - n_cls) / max(1, n_in)}
    for code, name in LSP_TYPE_NAMES.items():
        out[name] = float((codes == code).sum()) / max(1, n_cls)
    return out


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True)
@click.option("--iteration", default=10, show_default=True,
              help="Compare at this iteration (the ensemble arms' budget).")
@click.option("--tolerance", default=0.1, show_default=True)
@click.option("--target", default=0.12, show_default=True)
@click.option("--mu-window", default="1000,1250", show_default=True)
@click.option("--ens-glob", default="*ens5*", show_default=True)
@click.option("--skip-composition", is_flag=True, default=False)
def main(manifest, output_dir, iteration, tolerance, target, mu_window,
         ens_glob, skip_composition):
    lo, hi = (float(v) for v in mu_window.split(","))
    out_root = Path("/ptmp/jwuerzin/output")
    rand = _random_baseline(Path(output_dir) / "random_baseline_prevalence.json",
                            tolerance)
    click.echo(f"[ens] random baseline at tol={tolerance:g}: "
               f"{rand:.6f} hits/attempt")
    rows = list(csv.DictReader(open(manifest)))
    results: dict = {}

    for model in MODELS:
        rec: dict = {"model": model, "display": DISPLAY[model]}

        # ---- ensemble arm: one run per model -------------------------------
        ens = [d for d in sorted(out_root.glob(ens_glob))
               if d.is_dir() and (d / "state.pt").exists()
               and d.name.startswith(f"active_learning_{model}_")]
        # dnn must not swallow dnn_match_trafo
        if model == "dnn":
            ens = [d for d in ens if "match_trafo" not in d.name]
        if not ens:
            click.echo(f"[ens] {model}: no ensemble run matching {ens_glob}", err=True)
            continue
        e = _arm_stats(ens[0], iteration, tolerance, target, (lo, hi))
        if e is None:
            click.echo(f"[ens] {model}: ensemble run has < {iteration} iterations",
                       err=True)
            continue
        rec["ensemble"] = e
        rec["ensemble_dir"] = str(ens[0])

        # ---- dropout arm: the matched top_k / cold cell, all seeds ---------
        drop_dirs = [r["expected_run_dir"] for r in rows
                     if r["model"] == model and r["strategy"] == "top_k"
                     and r["warm_start"] == "cold"
                     and r["status"] in ("completed", "timeout")
                     and Path(r["expected_run_dir"], "state.pt").exists()]
        d_stats = [s for s in (_arm_stats(Path(d), iteration, tolerance, target,
                                          (lo, hi)) for d in drop_dirs)
                   if s is not None]
        if not d_stats:
            click.echo(f"[ens] {model}: no dropout arm at iteration {iteration}",
                       err=True)
            continue
        agg = {}
        for k in ("L", "hits", "hit_rate", "hits_desired", "gpu_hours"):
            v = np.array([s[k] for s in d_stats], dtype=float)
            agg[k] = float(v.mean())
            agg[k + "_sem"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        agg["n_seeds"] = len(d_stats)
        rec["dropout"] = agg

        if not skip_composition:
            rec["ensemble_composition"] = _composition([ens[0]], iteration,
                                                       tolerance, target)
            rec["dropout_composition"] = _composition(drop_dirs, iteration,
                                                      tolerance, target)

        results[model] = rec
        e_hd, d_hd = e["hits_desired"], agg["hits_desired"]
        click.echo(
            f"\n[ens] {DISPLAY[model]}  (iteration {iteration})\n"
            f"        {'':<14}{'ensemble':>12}{'dropout':>16}\n"
            f"        {'hits/desired':<14}{e_hd:>12.4f}"
            f"{d_hd:>10.4f}+-{agg['hits_desired_sem']:.4f}\n"
            f"        {'vs random':<14}{e_hd / rand:>11.2f}x"
            f"{d_hd / rand:>15.2f}x\n"
            f"        {'hit rate':<14}{e['hit_rate']:>12.4f}"
            f"{agg['hit_rate']:>10.4f}+-{agg['hit_rate_sem']:.4f}\n"
            f"        {'GPU-hours':<14}{e['gpu_hours']:>12.4f}"
            f"{agg['gpu_hours']:>10.4f}+-{agg['gpu_hours_sem']:.4f}"
            f"   ({e['gpu_hours'] / max(agg['gpu_hours'], 1e-9):.2f}x)\n"
            f"        {'|L|':<14}{e['L']:>12}{agg['L']:>16.0f}")
        for tag in ("ensemble_composition", "dropout_composition"):
            c = rec.get(tag)
            if c:
                click.echo(f"        {tag.split('_')[0]:<14}n={c['n']:<7} "
                           f"bino={c['bino']:.3f} wino={c['wino']:.3f} "
                           f"higgsino={c['higgsino']:.3f} mixed={c['mixed']:.3f}")

    p = Path(output_dir) / "ensemble_comparison.json"
    p.write_text(json.dumps({"config": {"iteration": iteration,
                                        "p_valid": P_VALID,
                                        "random_hits_per_attempt": rand,
                                        "tolerance": tolerance},
                             "results": results}, indent=1))
    click.echo(f"\n[ens] wrote {p}")


if __name__ == "__main__":
    main()
