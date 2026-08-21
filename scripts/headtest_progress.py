"""Read the head/strategy test runs at whatever iteration they have reached.

The runs launched by ``slurm/submit_headtest_expr.sh`` are single-seed, are not
registered in a sweep manifest, and will stop at different iterations when they
hit their wall clock. The standard analysis path assumes complete runs listed in
a manifest, so this reads the run directories directly and reports the headline
quantity per iteration, tolerating any truncation point.

What it measures, straight from ``state.pt``:

  per-valid in-band rate   of the points each iteration actually acquired, i.e.
                           the fraction of newly labelled models landing inside
                           the +-10% band around the target. This is directly
                           comparable to the paper's "per valid" column (DNN
                           0.033, Deep GP 0.106 on ExpR) and to the pool's own
                           band prevalence, which is what "random" means here.
  cumulative rate          the same over all points acquired so far, which is
                           the quantity that converts to a per-attempt yield by
                           multiplying with p_valid.
  labelled-set size        |L| per iteration, so runs that differ in how many
                           candidates survived validity are compared fairly.

Acquisition-time metrics only: verdict accuracy needs the per-iteration
checkpoints reloaded against a held-out split, which the accuracy plotter does
and which is worth running once the runs stop moving.

Usage
-----
    python scripts/headtest_progress.py
    python scripts/headtest_progress.py --glob '/ptmp/jwuerzin/output/headtest_*' \
        --target ExpR --tau 0.1
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import click
import torch

# Band prevalence of the valid random-scan pool, i.e. what a uniform draw gets,
# and the per-attempt validity rate that converts per-valid to per-attempt.
POOL_PREVALENCE = {"ExpR": 0.0336, "DMRD": 0.0092}
P_VALID = {"ExpR": 0.5839, "DMRD": 0.4454}
TRUE_VALUE = {"ExpR": 1.0, "DMRD": 0.12}


def _cfg(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("config", {}) or {}
        except Exception:
            pass
    return {}


def _label(run_dir: Path, cfg: dict) -> str:
    """Model/arm from the directory name, with the config as the authority."""
    m = re.match(r"headtest_([a-z]+)_([a-z]+)_seed(\d+)", run_dir.name)
    model, arm, seed = (m.group(1), m.group(2), m.group(3)) if m else ("?", "?", "?")
    head = cfg.get("head", "regression")
    strat = cfg.get("selection_strategy", "?")
    return f"{model:<12} {arm:<8} seed{seed} [{head[:3]}/{strat}]"


def _per_iteration(state: dict, true_value: float, tau: float):
    """(iteration, |L|, new points, in-band among new, in-band cumulative)."""
    n_tr, n_va = state["al_n_train"], state["al_n_val"]
    Y = state["Y"].view(-1)
    Yv = state["Y_val"].view(-1)
    rows = []
    prev_t = prev_v = 0
    for k in range(len(n_tr)):
        t, v = int(n_tr[k]), int(n_va[k])
        new = torch.cat([Y[prev_t:t], Yv[prev_v:v]])
        tot = torch.cat([Y[:t], Yv[:v]])
        in_new = int(((new / true_value - 1.0).abs() <= tau).sum()) if len(new) else 0
        in_tot = int(((tot / true_value - 1.0).abs() <= tau).sum())
        rows.append((k + 1, t + v, len(new), in_new, in_tot))
        prev_t, prev_v = t, v
    return rows


@click.command()
@click.option("--glob", "pattern", default="/ptmp/jwuerzin/output/headtest_*",
              help="Glob of run directories.")
@click.option("--target", default="ExpR", type=click.Choice(sorted(TRUE_VALUE)))
@click.option("--tau", default=0.1, show_default=True, help="Band half-width.")
@click.option("--tail", default=10, show_default=True,
              help="Also report the rate over the last N iterations, which is "
                   "where a loop that improves with data shows it.")
@click.option("--per-iteration/--no-per-iteration", default=False,
              help="Print the full per-iteration table for each run.")
def main(pattern, target, tau, tail, per_iteration):
    dirs = sorted(Path(p) for p in glob.glob(pattern) if (Path(p) / "state.pt").exists())
    if not dirs:
        raise click.UsageError(f"no run directories with a state.pt match {pattern}")

    ref = POOL_PREVALENCE[target]
    pv = P_VALID[target]
    tv = TRUE_VALUE[target]
    click.echo(f"target {target}: pool band prevalence {ref:.4f} per valid point "
               f"({ref * pv:.4f} per attempt at p_valid {pv})")
    click.echo(f"\n{'run':<46} {'iters':>5} {'|L|':>7} {'cum rate':>9} "
               f"{'vs rand':>8} {'last' + str(tail):>9} {'vs rand':>8}")
    click.echo("-" * 100)

    for d in dirs:
        cfg = _cfg(d)
        try:
            state = torch.load(d / "state.pt", weights_only=False, map_location="cpu")
            rows = _per_iteration(state, tv, tau)
        except Exception as exc:                      # a run mid-write, say
            click.echo(f"{_label(d, cfg):<46} unreadable: {exc}")
            continue
        if not rows:
            click.echo(f"{_label(d, cfg):<46} no completed iterations yet")
            continue

        it, size, _new, _inb, in_tot = rows[-1]
        cum = in_tot / size if size else 0.0
        recent = rows[-tail:] if len(rows) > tail else rows
        r_new = sum(r[2] for r in recent)
        r_in = sum(r[3] for r in recent)
        rate_tail = r_in / r_new if r_new else float("nan")
        click.echo(f"{_label(d, cfg):<46} {it:>5} {size:>7} {cum:>9.4f} "
                   f"{cum / ref:>7.2f}x {rate_tail:>9.4f} {rate_tail / ref:>7.2f}x")

        if per_iteration:
            for k, size_k, new_k, in_k, tot_k in rows:
                rate = in_k / new_k if new_k else float("nan")
                click.echo(f"      iter {k:>3}  |L|={size_k:>6}  new={new_k:>4}  "
                           f"in-band={in_k:>3}  rate={rate:.4f} "
                           f"({rate / ref:.2f}x)  cum={tot_k / size_k:.4f}")
    click.echo("\nRates are per valid acquired point; multiply by p_valid "
               f"({pv}) for per-attempt.")


if __name__ == "__main__":
    main()
