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

from coverage_saturation import _support_axis  # noqa: E402
from pmssm.config import TARGET_CONFIG as TARGET_CONFIG_  # noqa: E402
from mcmc_diagnostics import (  # noqa: E402
    DEFAULT_AL_PICKS, MODEL_DISPLAY, PARAM_ORDER, picks_with_tag)
from coverage_saturation import (  # noqa: E402
    AXES as COV_AXES,
    _cells,
    build_target,
    coverage_of,
)

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_ITER_RE = re.compile(r"Iteration (\d+) ===")
_SEL_START_RE = re.compile(
    r"Running \d+ (?:MC Dropout|TabPFN ensemble) forward passes"
    r"|Using entropy-based batch selection"
    r"|Entropy selection: evaluating"
    r"|Running iterative batch selector"
    r"|Generating \d+ candidate points"
    r"|Select(?:ing|ed) .*top[_-]?k", re.IGNORECASE)
_SEL_END_RE = re.compile(r"Saved selected points to")


def _ts(line: str) -> datetime | None:
    m = _TS_RE.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


# Each training attempt opens with this line. The driver APPENDS to the log on a
# resume rather than truncating, so a retried iteration accumulates several
# attempts in one file and a naive first-to-last span measures the wall-clock
# between the earliest failure and the eventual success. For the Deep GP, which
# was retried three times after GPU faults, that inflated its total from
# ~2.6 to 37 GPU-hours. Only the final attempt produced the checkpoint the
# results rest on, so only that attempt is the cost of the result.
_ATTEMPT_START = re.compile(r"Training set size:")


def _train_seconds(iter_dir: Path) -> float | None:
    log = iter_dir / "al_training.log"
    if not log.exists():
        return None
    with open(log, errors="replace") as fh:
        lines = fh.readlines()
    starts = [i for i, ln in enumerate(lines) if _ATTEMPT_START.search(ln)]
    tail = lines[starts[-1]:] if starts else lines
    stamps = [t for t in (_ts(ln) for ln in tail) if t is not None]
    if len(stamps) < 2:
        # A single-timestamp attempt carries no measurable span; fall back to the
        # whole file rather than reporting zero, and only if that has two.
        stamps = [t for t in (_ts(ln) for ln in lines) if t is not None]
        if len(stamps) < 2:
            return None
    return (stamps[-1] - stamps[0]).total_seconds()


_TABPFN_FITEVAL_RE = re.compile(
    r"TabPFN fit\+eval wall-clock:\s*([0-9.]+)s")


def _tabpfn_fit_eval_seconds(main_log: Path) -> dict[int, float]:
    """{iteration: seconds} for TabPFN's fit+eval phase.

    TabPFN takes no gradient step, so it writes no ``al_training.log``; the
    work equivalent to another model's training is the in-context fit (0.7 s,
    it only memorises the context) plus the forward passes that ingest that
    context to score the train/val/eval sets. The driver times that phase on
    one line, and because the AL and baseline threads run concurrently on two
    GPUs the reported wall-clock is comparable to the span an
    ``al_training.log`` covers for the other models.
    """
    out: dict[int, float] = {}
    cur = None
    with open(main_log, errors="replace") as fh:
        for line in fh:
            m = _ITER_RE.search(line)
            if m:
                cur = int(m.group(1))
                continue
            m = _TABPFN_FITEVAL_RE.search(line)
            if m and cur is not None:
                out[cur] = float(m.group(1))
    return out


def _iter_dir_chain(d: Path) -> dict[int, Path]:
    """{iteration number: directory} for a run, following resume chains.

    A run resumed with --resume-from writes only the iterations IT ran, so a
    continuation directory starts at e.g. iteration_041 while its state.pt
    carries the whole history from iteration 1. The earlier iterations live in
    the base directory, whose name is a prefix of the continuation's. Walking
    both is what makes a resumed run's cumulative compute complete; reading the
    continuation alone would silently drop every iteration before the resume
    point, and the resulting trajectory would look plausible.
    """
    out: dict[int, Path] = {}
    candidates = [d]
    parent, name = d.parent, d.name
    if parent.is_dir():
        # Longest-prefix siblings first, so the nearest ancestor in a chain of
        # resumes wins over an earlier one for any iteration they share.
        candidates += sorted(
            (sib for sib in parent.iterdir()
             if sib.is_dir() and sib != d and name.startswith(sib.name)),
            key=lambda sib: len(sib.name), reverse=True)
    for cand in candidates:
        for it_dir in cand.glob("iteration_[0-9][0-9][0-9]"):
            try:
                n = int(it_dir.name.split("_")[1])
            except ValueError:
                continue
            out.setdefault(n, it_dir)      # first candidate wins
    return out


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
@click.option("--picks", "picks_override", default=None,
              help="Override the (strategy, warm) cell per model as "
                   "model:strategy:warm[,...]. Needed for off-sweep families "
                   "whose cell is not the benchmark's best, e.g. the "
                   "large-batch runs, which use top_k where DEFAULT_AL_PICKS "
                   "names entropy_batch and would therefore match no row.")
@click.option("--models", default=None,
              help="Comma list of picks (default: all DEFAULT_AL_PICKS).")
@click.option("--min-seeds", default=2, show_default=True)
@click.option("--min-iters", default=5, show_default=True,
              help="Iterations a seed needs before its compute series is used. "
                   "Lower it to keep a cell that barely started: the "
                   "large-batch Deep GP run reached three iterations, and "
                   "dropping it hid the compute it had already spent.")
@click.option("--coverage/--no-coverage", default=True, show_default=True,
              help="Also plot covered in-band support versus cumulative compute.")
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True, help="Emcee reference defining one support.")
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358",
              show_default=True,
              help="Random-scan pool defining the second support. Same axes, "
                   "bins, tolerance and min_cell as the emcee one; only the "
                   "reference population differs. Pass '' to omit that panel.")
@click.option("--model-tag", default="", show_default=True,
              help="OUTPUT_TAG of a variant sweep (e.g. 'expr'); re-keys the\n                   default per-model picks so tagged manifest rows resolve.")
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--target", default="DMRD", show_default=True,
              help="TARGET_CONFIG key; sets the value the in-band test is taken around.")
@click.option("--n-bins", default=12, show_default=True)
@click.option("--min-cell", default=20, show_default=True)
@click.option("--mcmc-max-samples", default=500_000, show_default=True)
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True)
def main(manifest, output_dir, sweep_id, include_status, picks_override, models, min_seeds, min_iters,
         coverage, mcmc_data_dir, baseline_data_dir, tolerance, target, model_tag, n_bins, min_cell, mcmc_max_samples,
         require_neutralino_lsp):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_hit_rate_trajectories_multiseed as phr

    statuses = {s.strip() for s in include_status.split(",")}
    picks = picks_with_tag(model_tag)
    if picks_override:
        parsed = {}
        for tok in picks_override.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                m, st, wm = tok.split(":")
            except ValueError:
                raise click.ClickException(
                    f"--picks entry {tok!r} is not model:strategy:warm") from None
            parsed[m] = (st, wm)
        picks = parsed
    if models:
        wanted = {m.strip() for m in models.split(",")}
        picks = {m: sw for m, sw in picks.items() if m in wanted}
    rows = [r for r in csv.DictReader(open(manifest))
            if r["status"] in statuses
            and (sweep_id is None or r.get("sweep_id", "").startswith(sweep_id))]

    cov_ctx = None
    # One coverage panel per support source. The emcee posterior exists only for
    # the relic density, whereas the random-scan pool exists for every target, so
    # Omega gets both panels and any other target gets the pool one. Binning,
    # tolerance, half-split and min_cell are shared, so the panels differ only in
    # which population defines the target region — but note their support sizes
    # differ a lot (a flat scan barely samples the Omega band), so a coverage
    # value is comparable ACROSS MODELS within a panel, not across panels.
    cov_sources: list[tuple[str, str]] = []
    if coverage:
        if TARGET_CONFIG_[target].get("has_mcmc_reference", False) and mcmc_data_dir:
            cov_sources.append(("mcmc", "emcee reference"))
        if baseline_data_dir:
            cov_sources.append(("pool", "random scan"))
        if not cov_sources:
            click.echo(f"[compute] no support source available for target "
                       f"{target!r}; coverage panels disabled.", err=True)
            coverage = False
    cov_ctxs: list[tuple] = []
    if coverage:
        from analyse_runs import filter_run_neutralino_lsp, load_run
        ax_idx = [PARAM_ORDER.index(a) for a in COV_AXES]
        for src, src_label in cov_sources:
            rng = np.random.default_rng(20260807)
            edges, tmap, n_target, _held = build_target(
                mcmc_data_dir, ax_idx, n_bins, min_cell, tolerance,
                mcmc_max_samples, require_neutralino_lsp, rng, true_val=None,
                target=target, support_source=src,
                pool_data_dir=baseline_data_dir)
            cov_ctxs.append((src, src_label, ax_idx, edges, tmap, n_target,
                             load_run, filter_run_neutralino_lsp))
            click.echo(f"[compute] {src_label} support: {n_target} cells "
                       f"({n_bins}^{len(COV_AXES)} grid, min_cell={min_cell})")
        cov_ctx = cov_ctxs[0]

    def _cov_trajectory(run_dir: str, ctx=None) -> list[float] | None:
        """Covered fraction of one support after each iteration."""
        _src, _lbl, ax_idx, edges, tmap, n_target, load_run, veto_fn = ctx or cov_ctx
        try:
            run = load_run(run_dir)
            if require_neutralino_lsp:
                run = veto_fn(run)
        except Exception:
            return None
        Y = np.asarray(run.Y).ravel()
        from pmssm.config import TARGET_CONFIG  # noqa: PLC0415
        _tv = TARGET_CONFIG[target]["true_value"]
        inb = np.abs(Y - _tv) / _tv < tolerance
        cells = np.where(inb, _cells(np.asarray(run.X)[:, ax_idx], edges), -1)
        nt = list(run.n_train_per_iter)
        return [coverage_of(cells[:min(int(n), len(cells))], tmap, n_target)
                for n in nt]

    def _collect_cell(model: str, strat: str, warm: str):
        """(Cm, Lm) seed matrices for one cell, or None if too few seeds."""
        sel = [r for r in rows
               if (r["model"], r["strategy"], r["warm_start"]) == (model, strat, warm)]
        per_seed_C, per_seed_L, per_seed_T, per_seed_S = [], [], [], []
        per_seed_V: list[list[np.ndarray]] = [[] for _ in cov_ctxs]
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
            fit_secs = _tabpfn_fit_eval_seconds(d / "active_learning.log")
            C, L, T, S = [], [], [], []
            cum = 0.0
            chain = _iter_dir_chain(d)
            for i in range(len(n_tr)):
                iter_dir = chain.get(i + 1)
                if iter_dir is None:
                    # Stop, but say so. Truncating in silence yields a curve
                    # that ends early for no visible reason, which is how a
                    # resumed run's extension can vanish from a figure.
                    if i < len(n_tr) - 1:
                        click.echo(
                            f"[compute] {d.name}: no directory for iteration "
                            f"{i + 1} of {len(n_tr)} (searched the resume "
                            f"chain); truncating this seed there", err=True)
                    break
                tr = _train_seconds(iter_dir)
                if tr is None:
                    # No al_training.log: TabPFN. Substitute its fit+eval
                    # phase, which is the training-equivalent work.
                    tr = fit_secs.get(i + 1)
                    if tr is None:
                        break
                se = sel_secs.get(i + 1, 0.0)
                cum += tr + se
                C.append(cum / 3600.0)
                L.append(int(n_tr[i]) + int(n_va[i]))
                T.append(tr)
                S.append(se)
            if len(C) >= min_iters:
                per_seed_C.append(np.asarray(C))
                per_seed_L.append(np.asarray(L, dtype=float))
                per_seed_T.append(np.asarray(T))
                per_seed_S.append(np.asarray(S))
                if cov_ctx is not None:
                    for _k, _ctx in enumerate(cov_ctxs):
                        _cv = _cov_trajectory(str(d), _ctx)
                        if _cv is not None:
                            per_seed_V[_k].append(np.asarray(_cv, dtype=float))

        if len(per_seed_C) < min_seeds:
            return None
        n_it = min(len(c) for c in per_seed_C)
        # One coverage matrix per support source; None where a source produced
        # nothing for every seed of this cell.
        V = [np.stack([v[:n_it] for v in per_src]) if len(per_src) == len(per_seed_C)
             and per_src else None for per_src in per_seed_V] or None
        return (np.stack([c[:n_it] for c in per_seed_C]),
                np.stack([l[:n_it] for l in per_seed_L]),
                np.stack([t[:n_it] for t in per_seed_T]),
                np.stack([s[:n_it] for s in per_seed_S]), V)

    STRAT_SHORT = {"top_k": "top-k", "top_k_tol_only": "tol-only",
                   "entropy_batch": "entropy"}
    STRAT_LS = {"top_k": "--", "top_k_tol_only": ":", "entropy_batch": "-"}

    cell_summary: dict = {}
    # One coverage axis per support source, and none at all when coverage is not
    # computable: an empty panel still labelled with a quantity we cannot define
    # is worse than no panel.
    n_cov = len(cov_ctxs) if coverage else 0
    fig, _axs = plt.subplots(1, 1 + n_cov,
                             figsize=(7.0 + 5.5 * n_cov, 5.2), squeeze=False)
    ax = _axs[0, 0]
    cov_axes = list(_axs[0, 1:])
    ax3 = cov_axes[0] if cov_axes else None
    fig2, axes2 = plt.subplots(2, 2, figsize=(11.5, 8.6))
    (ax_l, ax_c), (ax_t, ax_s) = axes2
    summary = {}
    for model, (strat, warm) in picks.items():
        cell = _collect_cell(model, strat, warm)
        if cell is None:
            click.echo(f"[compute] {model}: too few usable seeds — skipped")
            continue
        Cm, Lm, Tm, Sm, Vm = cell
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

        # Vm is one coverage matrix per support source, aligned with cov_axes.
        for _ax, _Vm in zip(cov_axes, Vm if isinstance(Vm, list) else [Vm]):
            if _Vm is None or not np.isfinite(_Vm).any():
                continue
            v_mu = np.nanmean(_Vm, axis=0)
            v_sem = np.nanstd(_Vm, axis=0, ddof=1) / np.sqrt(len(_Vm))
            _ax.plot(c_mu, v_mu, "o-", ms=2.5, lw=1.6, color=col, label=pick_label)
            _ax.fill_between(c_mu, v_mu - v_sem, v_mu + v_sem, color=col,
                             alpha=0.2, lw=0)

        # companion figure: |L| vs iteration, per-iteration compute vs iteration
        it_ax = np.arange(1, n_it + 1)
        ax_l.plot(it_ax, l_mu, lw=1.6, color=col,
                  label=pick_label)
        ax_l.fill_between(it_ax, l_mu - l_sem, l_mu + l_sem, color=col, alpha=0.2, lw=0)
        for ax_i, M in ((ax_c, Tm + Sm), (ax_t, Tm), (ax_s, Sm)):
            m_mu = M.mean(0)
            m_sem = M.std(0, ddof=1) / np.sqrt(len(M))
            ax_i.plot(it_ax, m_mu, lw=1.6, color=col)
            ax_i.fill_between(it_ax, m_mu - m_sem, m_mu + m_sem,
                              color=col, alpha=0.2, lw=0)
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
    ax.legend(fontsize=8)
    # Name the population that defined each support, and its cell count: the
    # supports differ a lot in resolution (a flat scan barely samples the relic
    # band), so a coverage value compares models WITHIN a panel, not across them.
    for _ax, _ctx in zip(cov_axes, cov_ctxs):
        _lbl, _n = _ctx[1], _ctx[5]
        _ax.set_xscale("log")
        _ax.set_xlabel("cumulative surrogate compute [GPU h] (training + selection)")
        _ax.set_ylabel(f"fraction of {_lbl} in-band support covered\n({_n} cells)")
        _support_axis(_ax, full_range=False)
        _ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "compute_vs_dataset.png"
    fig.savefig(p, dpi=200)
    click.echo(f"[compute] wrote {p}")

    ax_l.set_ylabel(r"labeled-set size $|L|$")
    ax_l.grid(alpha=0.3)
    ax_l.legend(fontsize=8)
    for ax_i, lab in ((ax_c, "surrogate compute per iteration [s]\n(training + selection)"),
                      (ax_t, "training time per iteration [s]"),
                      (ax_s, "selection time per iteration [s]")):
        ax_i.set_ylabel(lab)
        ax_i.set_yscale("log")
        ax_i.grid(alpha=0.3, which="both")
    for ax_i in (ax_l, ax_c, ax_t, ax_s):
        ax_i.set_xlabel("AL iteration")
    fig2.tight_layout()
    p2 = out / "compute_per_iteration.png"
    fig2.savefig(p2, dpi=200)
    click.echo(f"[compute] wrote {p2}")

    # ── per warm-mode figures: every (model, strategy) cell ──────────────────
    from matplotlib.lines import Line2D
    # Acquisition-variant cells (*_laplace) are registered in MODEL_DISPLAY so
    # that --picks can name them, but they must not leak into the main-body
    # per-warm-mode figures just by being registered: those compare
    # architectures under one acquisition rule. Select them explicitly instead.
    all_models = [m for m in MODEL_DISPLAY
                  if not m.endswith(("_laplace", "_oracle"))]
    all_strats = ("entropy_batch", "top_k", "top_k_tol_only")
    for warm_mode in ("warm", "cold"):
        # Same two-panel layout as the best-per-model figure: what the compute
        # buys in labels (left) and in covered support (right), for every
        # (model, strategy) cell of this warm mode.
        figw, (axw, axw2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
        models_present, strats_present = set(), set()
        cov_present = False
        for model in all_models:
            # TabPFN has no warm/cold distinction to make: its weights never
            # change and the labelled set enters only as context, so every
            # iteration is a fresh in-context fit that carries nothing forward.
            # Its runs are tagged warm_start="tabpfn", which matches neither
            # panel, so it used to drop out of this figure entirely. Show it in
            # the cold panel, which is the honest analogue of never reusing
            # state, and only there so it is not drawn twice.
            if model.startswith("tabpfn"):
                if warm_mode != "cold":
                    continue
                cell_warm = "tabpfn"
            else:
                cell_warm = warm_mode
            for strat in all_strats:
                cell = _collect_cell(model, strat, cell_warm)
                if cell is None:
                    continue
                Cm, Lm, _Tm, _Sm, Vm = cell
                # This companion figure has a single coverage axis, so it shows
                # the FIRST support source only (emcee where one exists, the
                # random pool otherwise); the axis label below records which.
                Vm = (Vm[0] if isinstance(Vm, list) and Vm
                      else (None if isinstance(Vm, list) else Vm))
                col = phr.MODEL_COLORS.get(model, "gray")
                axw.plot(Cm.mean(0), Lm.mean(0), lw=1.5, ls=STRAT_LS[strat],
                         color=col)
                if Vm is not None and np.isfinite(Vm).any():
                    axw2.plot(Cm.mean(0), np.nanmean(Vm, axis=0), lw=1.5,
                              ls=STRAT_LS[strat], color=col)
                    cov_present = True
                cell_summary.setdefault(warm_mode, {}).setdefault(model, {})[strat] = {
                    "n_seeds": int(Cm.shape[0]),
                    "gpu_hours_final": float(Cm.mean(0)[-1]),
                    "L_final": float(Lm.mean(0)[-1]),
                    "coverage_final": (float(np.nanmean(Vm, axis=0)[-1])
                                       if Vm is not None and np.isfinite(Vm).any()
                                       else None),
                    "coverage_sem": (float(np.nanstd(Vm, axis=0, ddof=1)[-1]
                                           / np.sqrt(len(Vm)))
                                     if Vm is not None and len(Vm) > 1
                                     and np.isfinite(Vm).any() else None),
                }
                models_present.add(model)
                strats_present.add(strat)
        if not models_present:
            plt.close(figw)
            click.echo(f"[compute] {warm_mode}: no usable cells — skipped")
            continue
        for ax_i, lab in ((axw, r"labeled-set size $|L|$"),
                          (axw2, f"fraction of {cov_ctxs[0][1]} in-band support "
                                 f"covered ({cov_ctxs[0][5]} cells)"
                                 if cov_ctxs else
                                 "fraction of in-band support covered")):
            ax_i.set_xscale("log")
            ax_i.set_xlabel("cumulative surrogate compute [GPU h] "
                            "(training + selection)")
            ax_i.set_ylabel(lab)
            ax_i.grid(alpha=0.3, which="both")
            if "support covered" in lab:
                _support_axis(ax_i, full_range=False)
        mh = [Line2D([], [], color=phr.MODEL_COLORS.get(m, "gray"), lw=1.6,
                     label=MODEL_DISPLAY.get(m, m))
              for m in all_models if m in models_present]
        sh = [Line2D([], [], color="black", ls=STRAT_LS[s], lw=1.4,
                     label=STRAT_SHORT[s])
              for s in all_strats if s in strats_present]
        leg1 = axw.legend(handles=mh, fontsize=8, loc="upper left")
        axw.add_artist(leg1)
        axw.legend(handles=sh, fontsize=8, loc="lower right")
        if not cov_present:
            axw2.text(0.5, 0.5, "coverage disabled (--no-coverage)",
                      transform=axw2.transAxes, ha="center", va="center",
                      fontsize=9, color="0.45")
        figw.tight_layout()
        pw = out / f"compute_vs_dataset_{warm_mode}.png"
        figw.savefig(pw, dpi=200)
        plt.close(figw)
        click.echo(f"[compute] wrote {pw}")

    import json
    (out / "compute_vs_dataset.json").write_text(json.dumps(
        {"best_per_model": summary, "per_cell": cell_summary}, indent=1))
    click.echo(f"[compute] wrote {out / 'compute_vs_dataset.json'}")


if __name__ == "__main__":
    main()
