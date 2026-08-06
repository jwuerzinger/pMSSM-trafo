"""Labeled-dataset size versus cumulative surrogate compute, per model.

For each best-per-model pick this parses the run logs of every seed:

  * training time  — timestamp span of ``iteration_NNN/al_training.log``
    (the AL surrogate alone; the baseline model trains on the other GPU
    and is not part of the method's cost)
  * selection time — from the first selection marker in
    ``active_learning.log`` (MC-dropout passes / entropy evaluation /
    top-k) to the ``Saved selected points`` line

and accumulates their sum over iterations. Plotted: labeled-set size
|L| = n_train + n_val (from ``state.pt``) versus cumulative GPU-hours,
seed mean with SEM bands in both coordinates, one curve per model in the
hit-rate colours. Simulator (Run3ModelGen) time and evaluation/plotting
instrumentation are excluded by construction.

Usage:
    python scripts/plot_compute_vs_dataset.py \\
        --sweep-id 20260803_18 --include-status completed,running,timeout,submitted
"""
from __future__ import annotations

import csv
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

from mcmc_diagnostics import DEFAULT_AL_PICKS, MODEL_DISPLAY  # noqa: E402

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_ITER_RE = re.compile(r"Iteration (\d+) ===")
_SEL_START_RE = re.compile(
    r"Running \d+ (?:MC Dropout|TabPFN ensemble) forward passes"
    r"|Using entropy-based batch selection"
    r"|Entropy selection: evaluating"
    r"|Running iterative batch selector"
    r"|Select(?:ing|ed) .*top[_-]?k", re.IGNORECASE)
_SEL_END_RE = re.compile(r"Saved selected points to")


def _ts(line: str) -> datetime | None:
    m = _TS_RE.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


def _train_seconds(iter_dir: Path) -> float | None:
    log = iter_dir / "al_training.log"
    if not log.exists():
        return None
    first = last = None
    with open(log, errors="replace") as fh:
        for line in fh:
            t = _ts(line)
            if t is not None:
                if first is None:
                    first = t
                last = t
    if first is None or last is None:
        return None
    return (last - first).total_seconds()


def _selection_seconds(main_log: Path) -> dict[int, float]:
    """{iteration: selection seconds} parsed from active_learning.log."""
    out: dict[int, float] = {}
    cur_iter = None
    sel_start = None
    with open(main_log, errors="replace") as fh:
        for line in fh:
            m = _ITER_RE.search(line)
            if m:
                cur_iter = int(m.group(1))
                sel_start = None
                continue
            if cur_iter is None:
                continue
            if sel_start is None and _SEL_START_RE.search(line):
                sel_start = _ts(line)
                continue
            if sel_start is not None and _SEL_END_RE.search(line):
                t = _ts(line)
                if t is not None:
                    out[cur_iter] = (t - sel_start).total_seconds()
                sel_start = None
    return out


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--sweep-id", default=None,
              help="Only manifest rows whose sweep_id starts with this prefix.")
@click.option("--include-status", default="completed,running,timeout", show_default=True)
@click.option("--models", default=None,
              help="Comma list of picks (default: all DEFAULT_AL_PICKS).")
@click.option("--min-seeds", default=2, show_default=True)
def main(manifest, output_dir, sweep_id, include_status, models, min_seeds):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_hit_rate_trajectories_multiseed as phr

    statuses = {s.strip() for s in include_status.split(",")}
    picks = dict(DEFAULT_AL_PICKS)
    if models:
        wanted = {m.strip() for m in models.split(",")}
        picks = {m: sw for m, sw in picks.items() if m in wanted}
    rows = [r for r in csv.DictReader(open(manifest))
            if r["status"] in statuses
            and (sweep_id is None or r.get("sweep_id", "").startswith(sweep_id))]

    def _collect_cell(model: str, strat: str, warm: str):
        """(Cm, Lm) seed matrices for one cell, or None if too few seeds."""
        sel = [r for r in rows
               if (r["model"], r["strategy"], r["warm_start"]) == (model, strat, warm)]
        per_seed_C, per_seed_L = [], []
        seen = set()
        for r in sel:
            d = Path(r["expected_run_dir"])
            if str(d) in seen or not (d / "state.pt").exists():
                continue
            seen.add(str(d))
            state = torch.load(d / "state.pt", weights_only=False, map_location="cpu")
            n_tr = list(state.get("al_n_train") or [])
            n_va = list(state.get("al_n_val") or [])
            sel_secs = _selection_seconds(d / "active_learning.log")
            C, L = [], []
            cum = 0.0
            for i in range(len(n_tr)):
                tr = _train_seconds(d / f"iteration_{i + 1:03d}")
                if tr is None:
                    break
                cum += tr + sel_secs.get(i + 1, 0.0)
                C.append(cum / 3600.0)
                L.append(int(n_tr[i]) + int(n_va[i]))
            if len(C) >= 5:
                per_seed_C.append(np.asarray(C))
                per_seed_L.append(np.asarray(L, dtype=float))
        if len(per_seed_C) < min_seeds:
            return None
        n_it = min(len(c) for c in per_seed_C)
        return (np.stack([c[:n_it] for c in per_seed_C]),
                np.stack([l[:n_it] for l in per_seed_L]))

    STRAT_SHORT = {"top_k": "top-k", "top_k_tol_only": "tol-only",
                   "entropy_batch": "entropy"}
    STRAT_LS = {"top_k": "--", "top_k_tol_only": ":", "entropy_batch": "-"}

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    fig2, (ax_l, ax_c) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    summary = {}
    for model, (strat, warm) in picks.items():
        cell = _collect_cell(model, strat, warm)
        if cell is None:
            click.echo(f"[compute] {model}: too few usable seeds — skipped")
            continue
        Cm, Lm = cell
        n_it = Cm.shape[1]
        pick_label = MODEL_DISPLAY.get(model, model)
        if not model.startswith("tabpfn"):
            pick_label += f" ({STRAT_SHORT.get(strat, strat)}, {warm})"
        else:
            pick_label += f" ({STRAT_SHORT.get(strat, strat)})"
        c_mu, c_sem = Cm.mean(0), Cm.std(0, ddof=1) / np.sqrt(len(Cm))
        l_mu, l_sem = Lm.mean(0), Lm.std(0, ddof=1) / np.sqrt(len(Lm))
        col = phr.MODEL_COLORS.get(model, "gray")
        ax.plot(c_mu, l_mu, "o-", ms=2.5, lw=1.6, color=col,
                label=pick_label)
        ax.fill_betweenx(l_mu, c_mu - c_sem, c_mu + c_sem, color=col, alpha=0.2, lw=0)

        # companion figure: |L| vs iteration, per-iteration compute vs iteration
        it_ax = np.arange(1, n_it + 1)
        ax_l.plot(it_ax, l_mu, lw=1.6, color=col,
                  label=pick_label)
        ax_l.fill_between(it_ax, l_mu - l_sem, l_mu + l_sem, color=col, alpha=0.2, lw=0)
        Dm = np.diff(Cm, axis=1, prepend=0.0) * 3600.0  # seconds per iteration
        d_mu = Dm.mean(0)
        d_sem = Dm.std(0, ddof=1) / np.sqrt(len(Dm))
        ax_c.plot(it_ax, d_mu, lw=1.6, color=col)
        ax_c.fill_between(it_ax, d_mu - d_sem, d_mu + d_sem, color=col, alpha=0.2, lw=0)
        summary[model] = {"n_seeds": len(Cm), "gpu_hours_final": float(c_mu[-1]),
                          "L_final": float(l_mu[-1]),
                          "sec_per_iter_final": float((Cm[:, -1] - Cm[:, -2]).mean() * 3600)}
        click.echo(f"[compute] {model}: {len(Cm)} seeds, {n_it} iters, "
                   f"final |L|={l_mu[-1]:.0f} at {c_mu[-1]:.2f} GPU h "
                   f"(last iter {summary[model]['sec_per_iter_final']:.0f}s)")
    if not summary:
        raise click.ClickException("no usable runs")

    ax.set_xscale("log")
    ax.set_xlabel("cumulative surrogate compute [GPU h] (training + selection)")
    ax.set_ylabel(r"labeled-set size $|L|$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "compute_vs_dataset.png"
    fig.savefig(p, dpi=200)
    click.echo(f"[compute] wrote {p}")

    ax_l.set_xlabel("AL iteration")
    ax_l.set_ylabel(r"labeled-set size $|L|$")
    ax_l.grid(alpha=0.3)
    ax_l.legend(fontsize=9)
    ax_c.set_xlabel("AL iteration")
    ax_c.set_ylabel("surrogate compute per iteration [s]\n(training + selection)")
    ax_c.set_yscale("log")
    ax_c.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    p2 = out / "compute_per_iteration.png"
    fig2.savefig(p2, dpi=200)
    click.echo(f"[compute] wrote {p2}")

    # ── per warm-mode figures: every (model, strategy) cell ──────────────────
    from matplotlib.lines import Line2D
    all_models = [m for m in MODEL_DISPLAY if not m.startswith("tabpfn")]
    all_strats = ("entropy_batch", "top_k", "top_k_tol_only")
    for warm_mode in ("warm", "cold"):
        figw, axw = plt.subplots(figsize=(7.0, 5.2))
        models_present, strats_present = set(), set()
        for model in all_models:
            for strat in all_strats:
                cell = _collect_cell(model, strat, warm_mode)
                if cell is None:
                    continue
                Cm, Lm = cell
                axw.plot(Cm.mean(0), Lm.mean(0), lw=1.5,
                         ls=STRAT_LS[strat],
                         color=phr.MODEL_COLORS.get(model, "gray"))
                models_present.add(model)
                strats_present.add(strat)
        if not models_present:
            plt.close(figw)
            click.echo(f"[compute] {warm_mode}: no usable cells — skipped")
            continue
        axw.set_xscale("log")
        axw.set_xlabel("cumulative surrogate compute [GPU h] (training + selection)")
        axw.set_ylabel(r"labeled-set size $|L|$")
        axw.grid(alpha=0.3, which="both")
        mh = [Line2D([], [], color=phr.MODEL_COLORS.get(m, "gray"), lw=1.6,
                     label=MODEL_DISPLAY.get(m, m))
              for m in all_models if m in models_present]
        sh = [Line2D([], [], color="black", ls=STRAT_LS[s], lw=1.4,
                     label=STRAT_SHORT[s])
              for s in all_strats if s in strats_present]
        leg1 = axw.legend(handles=mh, fontsize=8, loc="upper left")
        axw.add_artist(leg1)
        axw.legend(handles=sh, fontsize=8, loc="lower right")
        pw = out / f"compute_vs_dataset_{warm_mode}.png"
        figw.savefig(pw, dpi=200)
        plt.close(figw)
        click.echo(f"[compute] wrote {pw}")


if __name__ == "__main__":
    main()
