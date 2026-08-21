"""Preliminary hit-rate trajectories for the head/strategy arms, per valid point
and per attempt.

Why two panels
--------------
The in-band rate *per valid acquired point* is what ``headtest_progress.py``
reports and what the acquisition machinery controls. It is not the cost-relevant
number: an invalid model still consumes a full prep_input -> SPheno ->
micrOMEGAs -> SModelS evaluation, so the quantity that converts to a budget is
the rate *per attempt*, i.e. hits/desired. The two differ by the arm's own
validity rate, and that rate is emphatically not the pool's 0.5839 -- an arm
acquires where it chose to, and the classification arms choose regions that fail
validity far more often than a uniform draw does. Reporting only the per-valid
panel would rank the arms in the opposite order from their cost.

Both panels are therefore drawn, with the per-attempt one first, and the ratio
between them is the validity rate printed in the summary table.

Where the numbers come from
---------------------------
in-band      ``state.pt``: the transformed targets of the points each iteration
             actually acquired, counted against |Y/true - 1| <= tau. The seed
             set is excluded, since the loop did not choose it.
attempts     the run's slurm log, from the ``Loaded N valid models from ntuple
             (filtered from M total)`` lines, summed per iteration between
             ``Active Learning Iteration`` markers. This counter is used rather
             than the ``Generation attempt k/n (M points)`` lines because the GP
             driver omits the point count from those, so a regex on them
             silently returns zero attempts for every GP arm.

Runs whose log cannot be located appear in the per-valid panel only, and are
listed as such rather than being silently dropped.

Usage
-----
    python scripts/plot_hit_rate_preliminary.py \
        --headtest-glob '/ptmp/jwuerzin/output/headtest_*_20260821_*' \
        --output-dir /ptmp/jwuerzin/analysis/joint/prelim_20260821
"""
from __future__ import annotations

import csv
import glob as globmod
import re
import subprocess
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

# Pool band prevalence per valid point, and pool validity per attempt. The
# per-attempt random baseline is the product: a uniform draw must also pay for
# the models it generates that are not valid.
REF_VALID = 0.0336
POOL_PVALID = 0.5839
REF_ATTEMPT = REF_VALID * POOL_PVALID

# Style is imported, not re-declared: this figure sits beside
# accuracy_arms_static_random.png in the same document, and a second copy of the
# palette would drift from it the first time either is touched. Colour therefore
# encodes the ARM here too, and the panel fixes the model.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_headtest_accuracy import (          # noqa: E402
    ARM_STYLE, BASELINE_STYLE, MODEL_DISPLAY, MODEL_ORDER,
    _HEADTEST_ARM, _HEADTEST_MODEL, _draw,
)

_PROD_MODEL = {
    "transformer_expr": "transformer", "dnn_expr": "dnn",
    "dnn_match_trafo_expr": "dnn_match_trafo", "exact_gp_expr": "exact_gp",
    "deep_gp_expr": "deep_gp", "tabpfn_expr": "tabpfn",
}


_LOG_INDEX: dict[str, list[Path]] | None = None


def _build_log_index() -> dict[str, list[Path]]:
    """Map run-directory name -> slurm log, in ONE pass over logs/.

    Grepping the whole of logs/ once per run does not scale: the directory holds
    several GB and a recursive grep per run silently hit its timeout, which then
    looked exactly like "this run has no log". ``grep -o -m`` stops after the
    first few matches in each file, and every driver prints its output directory
    in the header, so one pass is enough.
    """
    index: dict[str, list[Path]] = {}
    logs = sorted(globmod.glob("logs/*.out"))
    for lg in logs:
        try:
            out = subprocess.run(
                ["grep", "-o", "-m", "40", "-a", "-E",
                 r"/ptmp/jwuerzin/output/(headtest|active_learning)[A-Za-z0-9_]+", lg],
                capture_output=True, text=True, timeout=60).stdout.split()
        except Exception:
            continue
        for hit in set(out):
            index.setdefault(Path(hit).name, []).append(Path(lg))
    return index


def _find_log(run_dir: Path) -> list[Path]:
    """All logs that mention this run, largest first.

    More than one can match: a bundled sweep writes a per-seed log AND a
    combined dispatcher log, and only the former carries the generation lines.
    Picking by size alone chose the wrong one, so every candidate is returned
    and the caller keeps whichever actually parses.
    """
    global _LOG_INDEX
    if _LOG_INDEX is None:
        _LOG_INDEX = _build_log_index()
        click.echo(f"[index] {len(_LOG_INDEX)} run directories mapped to logs")
    got = _LOG_INDEX.get(run_dir.name, [])
    return sorted(got, key=lambda p: -p.stat().st_size)


def _attempts_per_iteration(log: Path) -> list[int]:
    """Models generated per iteration, from the ntuple filter lines."""
    per_iter: list[int] = []
    cur = 0
    started = False
    # The drivers do not agree on the banner: the GP and DNN scripts print
    # "=== GP/DNN Active Learning Iteration N ===" while the transformer prints
    # "=== Global Iteration N ===". Matching only the first silently yields zero
    # iterations for every transformer arm.
    pat_iter = re.compile(r"(?:Active Learning|Global) Iteration\s+(\d+)")
    pat_tot = re.compile(r"valid models from ntuple \(filtered from (\d+) total\)")
    with open(log, errors="ignore") as fh:
        for line in fh:
            if pat_iter.search(line):
                if started:
                    per_iter.append(cur)
                cur, started = 0, True
                continue
            m = pat_tot.search(line)
            if m and started:
                cur += int(m.group(1))
    if started:
        per_iter.append(cur)
    return per_iter


def _inband_per_iteration(state_path: Path, true_value: float, tau: float):
    """(in-band, acquired) per iteration, excluding the seed set."""
    st = torch.load(state_path, weights_only=False, map_location="cpu")
    ntr, nva = st["al_n_train"], st["al_n_val"]
    Y, Yv = st["Y"].view(-1), st["Y_val"].view(-1)
    rows = []
    pt, pv = int(ntr[0]), int(nva[0])          # skip the seed set
    for k in range(1, len(ntr)):
        t, v = int(ntr[k]), int(nva[k])
        new = torch.cat([Y[pt:t], Yv[pv:v]])
        nin = int(((new / true_value - 1.0).abs() <= tau).sum()) if len(new) else 0
        rows.append((nin, len(new)))
        pt, pv = t, v
    return rows


def _cumulative(nums, dens):
    """Cumulative ratio, as a fraction of the corresponding random baseline."""
    n = np.cumsum(np.asarray(nums, dtype=float))
    d = np.cumsum(np.asarray(dens, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)


@click.command()
@click.option("--headtest-glob", default="/ptmp/jwuerzin/output/headtest_*",
              show_default=True)
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/joint/manifest_expr.csv",
              show_default=True, help="Supplies the production reference arms.")
@click.option("--reference-seed", default=1, show_default=True,
              help="Which production seed to draw as the reference line.")
@click.option("--true-value", default=1.0, show_default=True)
@click.option("--tau", default=0.1, show_default=True)
@click.option("--max-iter", default=0, show_default=True,
              help="Truncate the x axis; 0 keeps everything.")
@click.option("--output-dir", required=True)
def main(headtest_glob, manifest, reference_seed, true_value, tau, max_iter,
         output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = []   # (model, arm, run_dir, is_reference)
    for d in sorted(globmod.glob(headtest_glob)):
        d = Path(d)
        m = re.match(r"headtest_([a-z]+)_([a-z]+)_seed(\d+)_", d.name)
        if not m or not (d / "state.pt").exists():
            continue
        model = _HEADTEST_MODEL.get(m.group(1), m.group(1))
        arm = _HEADTEST_ARM.get(m.group(2), m.group(2))
        runs.append((model, arm, d, False))

    if manifest and Path(manifest).exists():
        for r in csv.DictReader(open(manifest)):
            if int(r.get("seed", 0) or 0) != reference_seed:
                continue
            model = _PROD_MODEL.get(r["model"])
            d = Path(r["expected_run_dir"])
            if model and (d / "state.pt").exists():
                runs.append((model, r["strategy"], d, True))

    click.echo(f"collected {len(runs)} runs")
    series, missing = [], []
    for model, arm, d, is_ref in runs:
        rows = _inband_per_iteration(d / "state.pt", true_value, tau)
        if not rows:
            continue
        nin = [r[0] for r in rows]
        nvalid = [r[1] for r in rows]
        att = []
        for cand in _find_log(d):
            got = _attempts_per_iteration(cand)
            if len(got) >= len(nin):
                att = got[:len(nin)]
                break
        if not att:
            missing.append(f"{model}/{arm}")
        series.append(dict(model=model, arm=arm, ref=is_ref, nin=nin,
                           nvalid=nvalid, att=att, run=d.name))
        click.echo(f"  {model:<16} {arm:<16} iters={len(nin):>3} "
                   f"log={'yes' if att else 'NO '} {d.name}")
    if missing:
        click.echo(f"[note] no per-attempt data for: {', '.join(sorted(set(missing)))}")

    # ---- one figure per metric, panels per model, exactly as the accuracy
    # ---- figure is laid out, so the two can sit side by side in the document.
    models = [m for m in MODEL_ORDER if any(s_["model"] == m for s_ in series)]
    for mode, fname, ylab in (
            ("attempt", "hit_rate_per_attempt",
             "In-band rate per attempt (× random)"),
            ("valid", "hit_rate_per_valid",
             "In-band rate per valid point (× random)")):
        ncol = min(3, len(models)) or 1
        nrow = int(np.ceil(len(models) / ncol))
        # Shared axes, because the layout labels only the left column and the
        # bottom row: with independent scales those labels would describe one
        # panel and silently mislabel the other five, and a hit rate spans a
        # wide enough range here (0.2x to 4x) for that to matter.
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.3 * nrow),
                                 squeeze=False, sharex=True, sharey=True)
        flat = [a for row in axes for a in row]
        arms_seen, drew_baseline = [], False
        for ax, model in zip(flat, models):
            mine = [s_ for s_ in series if s_["model"] == model]
            # Reference arms first so the benchmark curve sits under the new ones.
            mine.sort(key=lambda s_: (not s_["ref"], s_["arm"]))
            n_ref_drawn = 0
            for s_ in mine:
                if mode == "attempt":
                    if not s_["att"]:
                        continue
                    y = _cumulative(s_["nin"], s_["att"]) / REF_ATTEMPT
                else:
                    y = _cumulative(s_["nin"], s_["nvalid"]) / REF_VALID
                x = np.arange(1, len(y) + 1, dtype=float)
                if max_iter:
                    keep = x <= max_iter
                    x, y = x[keep], y[keep]
                color, ls, marker, _lbl = ARM_STYLE.get(
                    s_["arm"], ("0.2", "-", "x", s_["arm"]))
                _draw(ax, x, y[None, :], color=color, linestyle=ls,
                      marker=marker, label=None)
                if s_["arm"] not in arms_seen and len(x):
                    arms_seen.append(s_["arm"])
                n_ref_drawn += int(s_["ref"] and len(x) > 0)
            # The random draw is the unit of this axis, so it is the flat line
            # at one, drawn in the same style the accuracy figure uses for it.
            ax.axhline(1.0, **BASELINE_STYLE)
            drew_baseline = True
            tag = MODEL_DISPLAY.get(model, model)
            ref_tag = "  (reference drawn)" if n_ref_drawn else ""
            if mode == "attempt" and not n_ref_drawn and any(
                    s_["ref"] for s_ in mine):
                ref_tag = "  (reference has no attempt counter)"
            ax.annotate(f"{tag}{ref_tag}", xy=(0.03, 0.96),
                        xycoords="axes fraction", va="top", ha="left",
                        fontsize=10)
            ax.grid(alpha=0.3)
        for ax in flat[len(models):]:
            ax.axis("off")
        for r in range(nrow):
            axes[r][0].set_ylabel(ylab)
        for c in range(ncol):
            axes[nrow - 1][c].set_xlabel("AL iteration")

        handles = [Line2D([0], [0], color=ARM_STYLE[a][0],
                          linestyle=ARM_STYLE[a][1], marker=ARM_STYLE[a][2],
                          markersize=4.5, lw=1.8, label=ARM_STYLE[a][3])
                   for a in arms_seen if a in ARM_STYLE]
        handles += [Line2D([0], [0], color="0.2", lw=1.8, label=a)
                    for a in arms_seen if a not in ARM_STYLE]
        if drew_baseline:
            handles.append(Line2D([0], [0], label="random draw",
                                  **BASELINE_STYLE))
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 3), frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.06 if nrow > 1 else 0.10, 1, 1))
        png = out / f"{fname}.png"
        fig.savefig(png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        click.echo(f"[write] {png}")

    csv_path = out / "hit_rate_preliminary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "arm", "iters", "attempts", "valid", "in_band",
                    "p_valid", "per_valid", "per_attempt",
                    "x_random_valid", "x_random_attempt", "run"])
        table = []
        for s in series:
            nv, nb = sum(s["nvalid"]), sum(s["nin"])
            at = sum(s["att"]) if s["att"] else 0
            pv_rate = nb / nv if nv else float("nan")
            pa_rate = nb / at if at else float("nan")
            w.writerow([s["model"], s["arm"], len(s["nin"]), at, nv, nb,
                        f"{nv/at:.4f}" if at else "",
                        f"{pv_rate:.5f}", f"{pa_rate:.5f}" if at else "",
                        f"{pv_rate/REF_VALID:.3f}",
                        f"{pa_rate/REF_ATTEMPT:.3f}" if at else "", s["run"]])
            table.append((s["model"], s["arm"], len(s["nin"]), at, nv, nb,
                          pv_rate, pa_rate))
    click.echo(f"[write] {csv_path}\n")

    click.echo(f"{'model':<12} {'arm':<14} {'it':>3} {'attempts':>9} {'valid':>7} "
               f"{'p_valid':>8} {'/valid':>8} {'xr':>6} {'/attempt':>9} {'xr':>6}")
    click.echo("-" * 92)
    for model, arm, it, at, nv, nb, pvr, par in sorted(
            table, key=lambda r: -(r[7] if r[7] == r[7] else -1)):
        click.echo(f"{model:<12} {arm:<14} {it:>3} {at or '-':>9} {nv:>7} "
                   f"{(nv/at if at else float('nan')):>8.3f} {pvr:>8.4f} "
                   f"{pvr/REF_VALID:>5.2f}x "
                   f"{par:>9.4f} {par/REF_ATTEMPT:>5.2f}x")


if __name__ == "__main__":
    main()
