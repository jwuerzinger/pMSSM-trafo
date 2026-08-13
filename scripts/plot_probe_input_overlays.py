"""Probe-vs-benchmark input marginals: does a bigger budget move the search?

The extended-budget (160-iteration) and large-batch (20k) probes let each
surrogate acquire far more than the 40-iteration benchmark does. This asks what
it did with the extra budget, by overlaying each probe cell's marginals on the
*same cell* from the benchmark sweep and taking the ratio.

Pairing is on the full cell key (model, strategy, warm start), so the only thing
differing between the two arms is the budget. That is what makes the ratio
interpretable: it is not a model comparison, it is the same configuration given
more room. Ratio panels read probe / benchmark, so above one means the probe
sampled that region more.

Only the August sweeps are eligible on the benchmark side. The manifest also
carries rows from earlier sweeps whose run directories are partly archived, and
pooling those would mix acquisition histories from different code versions.

Usage:
    python scripts/plot_probe_input_overlays.py            # both probes
    python scripts/plot_probe_input_overlays.py --probe extended
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyse_runs import (  # noqa: E402
    FREE_PARAM_INDICES,
    FREE_PARAM_NAMES,
)
from plot_hit_rate_trajectories_multiseed import (            # noqa: E402
    MODEL_COLORS,
    MODEL_DISPLAY,
)
from plot_laplace_input_overlays import (                     # noqa: E402
    OMEGA_LABEL,
    TRUE_VALUE,
    _pool_cell,
    plot_overlay_with_ratio,
)

MAIN_MANIFEST = "/ptmp/jwuerzin/analysis/all_runs/manifest_mainbody.csv"
PROBES = {
    "extended": {
        "manifest": "/ptmp/jwuerzin/analysis/probe_extended/manifest.csv",
        "out": "/ptmp/jwuerzin/analysis/probe_extended",
        "tag": "160 it",
    },
    "20k": {
        "manifest": "/ptmp/jwuerzin/analysis/probe_20k/manifest.csv",
        "out": "/ptmp/jwuerzin/analysis/probe_20k",
        "tag": "20k batch",
    },
}
# Benchmark-side sweeps to trust. Earlier sweeps predate the current generation
# config and are partly archived without their training logs.
BENCH_SWEEP_PREFIX = "202608"
BENCH_KEY = "{model}__bench"
PROBE_KEY = "{model}__probe"


def _cells(manifest, sweep_prefix=None):
    """{(model, strategy, warm): [run_dir, ...]} from a sweep manifest."""
    out = defaultdict(list)
    with open(manifest) as fh:
        for r in csv.DictReader(fh):
            if sweep_prefix and not r.get("sweep_id", "").startswith(sweep_prefix):
                continue
            out[(r["model"], r["strategy"], r["warm_start"])].append(
                r["expected_run_dir"])
    return out


REF_KEY = "__reference"


def _load_reference(mcmc_data_dir, require_neutralino_lsp, max_samples):
    """The emcee reference's free parameters and Omega, or (None, None).

    Ratios against this answer "how far is each arm from the posterior we are
    trying to reproduce?", which is the question the composition table raises
    and which a probe/benchmark ratio cannot address.
    """
    from pmssm.data import load_mcmc_data                      # noqa: PLC0415
    Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir,
                            require_neutralino_lsp=require_neutralino_lsp,
                            max_samples=max_samples or None)
    Xm = Xm.numpy() if hasattr(Xm, "numpy") else np.asarray(Xm)
    Ym = (Ym.numpy() if hasattr(Ym, "numpy") else np.asarray(Ym)).ravel()
    return Xm[:, FREE_PARAM_INDICES], Ym


@click.command()
@click.option("--probe", type=click.Choice(["extended", "20k", "both"]),
              default="both", show_default=True)
@click.option("--main-manifest", default=MAIN_MANIFEST, show_default=True)
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=True, show_default=True)
@click.option("--nbins", default=25, show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True,
              help="emcee reference; the ratio panels divide by it. Empty "
                   "string falls back to probe/benchmark ratios.")
@click.option("--mcmc-max-samples", default=500_000, show_default=True)
def main(probe, main_manifest, tolerance, require_neutralino_lsp, nbins,
         mcmc_data_dir, mcmc_max_samples):
    bench = _cells(main_manifest, sweep_prefix=BENCH_SWEEP_PREFIX)
    names = list(FREE_PARAM_NAMES) + [OMEGA_LABEL]
    which = list(PROBES) if probe == "both" else [probe]

    ref_X = ref_om = None
    if mcmc_data_dir:
        click.echo(f"[probe-overlay] loading reference from {mcmc_data_dir} ...")
        ref_X, ref_om = _load_reference(mcmc_data_dir, require_neutralino_lsp,
                                        mcmc_max_samples)
        click.echo(f"[probe-overlay] reference: {len(ref_X)} samples")

    for pname in which:
        spec = PROBES[pname]
        probe_cells = _cells(spec["manifest"])
        pooled, colors, labels, pairs = {}, {}, {}, []

        for (model, strat, warm), dirs in sorted(probe_cells.items()):
            bdirs = bench.get((model, strat, warm))
            if not bdirs:
                click.echo(f"[probe-overlay] {pname}: no benchmark cell for "
                           f"{model}/{strat}/{warm}; skipped", err=True)
                continue
            bk = BENCH_KEY.format(model=model)
            pk = PROBE_KEY.format(model=model)
            for key, dd in ((bk, bdirs), (pk, dirs)):
                X, om, n = _pool_cell(dd, require_neutralino_lsp)
                if X is None:
                    click.echo(f"[probe-overlay] {pname}: {key} empty", err=True)
                    break
                pooled[key] = (X, om)
                click.echo(f"[probe-overlay] {pname}: {key:<28} "
                           f"{n} seeds, {len(X)} points")
            else:
                disp = MODEL_DISPLAY.get(model, model)
                # Same colour for both arms, since it is one configuration at two
                # budgets; the linestyle carries the budget (see the plotter).
                colors[bk] = colors[pk] = MODEL_COLORS.get(model)
                labels[bk] = f"{disp} (40 it)"
                labels[pk] = f"{disp} ({spec['tag']})"
                pairs.append((bk, pk))

        if not pairs:
            click.echo(f"[probe-overlay] {pname}: nothing to plot", err=True)
            continue

        out = Path(spec["out"])
        out.mkdir(parents=True, exist_ok=True)
        for tag, restrict in (("inband", True), ("all", False)):
            data = {}
            if ref_X is not None:
                if restrict:
                    k = np.abs(ref_om - TRUE_VALUE) / TRUE_VALUE < tolerance
                    data[REF_KEY] = np.column_stack([ref_X[k], ref_om[k]])
                else:
                    data[REF_KEY] = np.column_stack([ref_X, ref_om])
                labels[REF_KEY] = "emcee reference"

            for key, (X, om) in pooled.items():
                if restrict:
                    keep = np.abs(om - TRUE_VALUE) / TRUE_VALUE < tolerance
                    if keep.sum() < 50:
                        click.echo(f"[probe-overlay] {key}: only "
                                   f"{int(keep.sum())} in-band; excluded", err=True)
                        continue
                    data[key] = np.column_stack([X[keep], om[keep]])
                else:
                    data[key] = np.column_stack([X, om])
            usable = [(a, b) for a, b in pairs if a in data and b in data]
            if not usable:
                continue
            path = out / f"probe_input_overlay_{tag}.png"
            plot_overlay_with_ratio(
                data, names, str(path), nbins=nbins, pairs=usable,
                colors=colors, labels=labels,
                ratio_label=("arm/ref" if ref_X is not None else "probe/bench"),
                ratio_ref=(REF_KEY if ref_X is not None else None))
            click.echo(f"[probe-overlay] wrote {path}")


if __name__ == "__main__":
    main()
