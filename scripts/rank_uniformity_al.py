"""Rank-statistic uniformity across AL seed replicas, via Run3ModelGen.

This is a thin driver: the statistics and the figures come from
``Run3ModelGen/source/Run3ModelGen/scripts/emcee_diagnostics.py`` itself
(``autocorr_time``, ``rank_uniformity``, ``plot_rank_hist``,
``plot_rank_ecdf``), so the AL cells and the MCMC reference are diagnosed by
one implementation under one set of conventions. Nothing statistical is
reimplemented here; this module only reshapes AL runs into the ensemble
layout upstream expects and loops over the sweep's best-per-model picks.

Mapping onto the upstream data model:

  * one seed replica  -> one "ensemble" holding a single walker, i.e. an array
    of shape ``(n_acquired, 1, n_params)`` in acquisition order
  * ``max_tau``       -> ``max`` over replicas and parameters of
    ``emcee.autocorr.integrated_time`` (upstream's ``all_tau_max`` rule)
  * ``discard``       -> acquisitions dropped from the head of each replica;
    the default 0 keeps the shared initial random block, ``--discard 2000``
    drops it so only genuinely acquired points are compared

Interpretation for AL (NOT the MCMC reading): the replicas are repetitions of
a stochastic acquisition process, not Markov chains targeting a common
stationary law, so uniformity is a reproducibility statement (independently
seeded runs acquire statistically indistinguishable point sets), never a
statement about calibration of the underlying distribution.

Run with the Run3ModelGen environment, which carries emcee and arviz:
    Run3ModelGen/.pixi/envs/default/bin/python scripts/rank_uniformity_al.py \\
        --require-neutralino-lsp --no-fold-signs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts"),
           str(_REPO_ROOT / "Run3ModelGen" / "source" / "Run3ModelGen" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:                       # needs the Run3ModelGen env (emcee, arviz)
    import emcee_diagnostics as ed  # noqa: E402  (single source of truth)
except ImportError:        # --export-only runs in the torch env instead
    ed = None

from mcmc_diagnostics import (  # noqa: E402
    DEFAULT_AL_PICKS,
    MODEL_DISPLAY,
    PARAM_ORDER,
)

try:                                    # only needed when building the cache
    from mcmc_diagnostics import _load_al_chains  # noqa: E402
except Exception:                       # pragma: no cover
    _load_al_chains = None

# Upstream orders parameters as the sampler backend stores them (sorted), and
# its fold logic keys off these exact names.
FREE_PARAMS_UP = ["AT", "Ab", "Atau", "M_1", "M_2", "meL", "meR", "mu", "tanb"]
_UP_TO_NTUPLE = {"AT": "IN_At", "Ab": "IN_Ab", "Atau": "IN_Atau", "M_1": "IN_M_1",
                 "M_2": "IN_M_2", "meL": "IN_meL", "meR": "IN_meR", "mu": "IN_mu",
                 "tanb": "IN_tanb"}
ALPHA = 0.01


def _pick_run_dirs(manifest: str, picks: dict, sweep_id: str | None,
                   statuses: set[str]) -> dict[str, list[str]]:
    """Run dirs per pick, restricted to one sweep.

    ``mcmc_diagnostics._picks_from_manifest`` filters on status alone, which
    pools generations (the Deep GP cell then contributes ten replicas from two
    sweeps). Diagnosing replica agreement across generations is meaningless,
    since the generations differ by an ingest-time veto, so the sweep is
    selected explicitly here.
    """
    import csv
    out: dict[str, list[str]] = {m: [] for m in picks}
    seen: set[str] = set()
    for r in csv.DictReader(open(manifest)):
        if r.get("status") not in statuses:
            continue
        if sweep_id and not (r.get("sweep_id") or "").startswith(sweep_id):
            continue
        m = r.get("model")
        if m not in picks:
            continue
        s_, w_ = picks[m]
        if (r.get("strategy"), r.get("warm_start")) != (s_, w_):
            continue
        d = r["expected_run_dir"]
        if d not in seen:
            seen.add(d)
            out[m].append(d)
    return out


def _al_ensembles(run_dirs, veto: bool, cache: Path) -> list[np.ndarray]:
    """One (n_acquired, 1, n_params) array per seed replica, upstream order.

    Loading ``state.pt`` needs torch, while the upstream diagnostics need
    emcee/arviz, and the two live in different environments here. The chains
    are therefore cached as an .npz on first use (run once with the torch
    environment), and read back from it afterwards.
    """
    if cache.exists():
        with np.load(cache) as z:
            return [z[k][:, None, :] for k in sorted(z.files, key=lambda x: int(x[1:]))]
    idx = [PARAM_ORDER.index(_UP_TO_NTUPLE[p]) for p in FREE_PARAMS_UP]
    chains = _load_al_chains(run_dirs, idx, require_neutralino_lsp=veto)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f"c{i}": np.asarray(c, dtype=float)
                       for i, c in enumerate(chains)})
    return [np.asarray(c, dtype=float)[:, None, :] for c in chains]


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs/rank",
              show_default=True)
@click.option("--models", default=None,
              help="Comma list of picks (default: all DEFAULT_AL_PICKS).")
@click.option("--discard", default=0, show_default=True,
              help="Acquisitions dropped from each replica's head "
                   "(2000 drops the shared initial random block).")
@click.option("--fold-signs/--no-fold-signs", default=False, show_default=True,
              help="Upstream canon: fold the exact (M_1,M_2,mu) sign symmetry "
                   "to mu >= 0 before ranking.")
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True)
@click.option("--sweep-id", default="20260803_18", show_default=True,
              help="Restrict to manifest rows whose sweep_id starts with this; "
                   "empty string uses every generation.")
@click.option("--include-status", default="completed,running,timeout,submitted",
              show_default=True,
              help="Manifest statuses to accept (the current sweep's rows still "
                   "read 'submitted' until the manifest is refreshed).")
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs/rank/chains",
              show_default=True, help="Where the per-cell chain .npz caches live.")
@click.option("--export-only", is_flag=True, default=False,
              help="Only build the chain caches (run this with the torch env).")
def main(manifest, output_dir, models, discard, fold_signs, require_neutralino_lsp,
         sweep_id, include_status, cache_dir, export_only):
    picks = dict(DEFAULT_AL_PICKS)
    if models:
        wanted = {m.strip() for m in models.split(",")}
        picks = {m: sw for m, sw in picks.items() if m in wanted}
    statuses = {x.strip() for x in include_status.split(",") if x.strip()}
    run_dirs = _pick_run_dirs(manifest, picks, sweep_id or None, statuses)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ed is None and not export_only:
        raise click.ClickException(
            "emcee_diagnostics unavailable: run with "
            "Run3ModelGen/.pixi/envs/default/bin/python, or pass --export-only "
            "to build the chain caches with the torch environment first")
    payload = {"config": {"sweep_id": sweep_id, "discard": discard,
                          "fold_signs": fold_signs,
                          "alpha": ALPHA, "bins": None if ed is None else ed.RANK_BINS,
                          "require_neutralino_lsp": require_neutralino_lsp,
                          "source": "Run3ModelGen emcee_diagnostics"},
               "results": {}}

    for model, dirs in run_dirs.items():
        tag = (sweep_id or "all").replace("/", "_")
        cache = (Path(cache_dir)
                 / f"{model}_{tag}_veto{int(require_neutralino_lsp)}.npz")
        chains = _al_ensembles(dirs, require_neutralino_lsp, cache)
        if export_only:
            click.echo(f"[rank] cached {len(chains)} replicas -> {cache}")
            continue
        if len(chains) < 2:
            click.echo(f"[rank] {model}: {len(chains)} replicas — skipped")
            continue
        taus = []
        for ch in chains:
            tau, _ok = ed.autocorr_time(ch[discard:], tol=ed.TAU_FACTOR)
            taus.append(np.nanmax(tau))
        max_tau = float(np.nanmax(taus))

        label = (f"{MODEL_DISPLAY.get(model, model)}, {len(chains)} seed replicas"
                 + (", mu>=0 canonical" if fold_signs else ", RAW signs (no fold)"))
        try:
            res = ed.rank_uniformity(chains, FREE_PARAMS_UP, discard, max_tau,
                                     label=label, canon=fold_signs)
        except Exception as exc:
            click.echo(f"[rank] {model}: rank_uniformity failed: {exc}", err=True)
            continue

        for fn, base in ((ed.plot_rank_hist, f"rank_{model}.png"),
                         (ed.plot_rank_ecdf, f"rank_ecdf_{model}.png")):
            try:
                fn(res, FREE_PARAMS_UP, str(out_dir / base))
            except Exception as exc:
                click.echo(f"[rank] {model}: {base} skipped: {exc}", err=True)

        worst = res.get("worst_p", res.get("worst_p_holm"))
        estimable = bool(res.get("estimable", True))
        verdict = ("not_estimable" if not estimable else
                   "fail" if (worst is not None and np.isfinite(worst) and worst < ALPHA)
                   else "pass")
        payload["results"][model] = {
            "n_chains": res["n_chains"], "n_thinned": res["n_thinned"],
            "max_tau": max_tau, "estimable": estimable,
            "worst_param": res.get("worst_param"), "worst_p_holm": worst,
            "verdict": verdict,
            "per_param": {p: {k: v for k, v in (res["per_param"].get(p) or {}).items()
                              if k in ("chi2", "dof", "p_raw", "p_holm")}
                          for p in FREE_PARAMS_UP},
            "shapes": {p: (res.get("shapes") or {}).get(p) for p in FREE_PARAMS_UP},
        }
        wp = "n/a" if worst is None or not np.isfinite(worst) else f"{worst:.2e}"
        click.echo(f"[rank] {model:<16} [{verdict:>13}] C={res['n_chains']} "
                   f"max_tau={max_tau:8.1f} n_thinned={res['n_thinned']:>6} "
                   f"worst p_Holm={wp} ({res.get('worst_param')})")

    p = out_dir / "rank_uniformity_al.json"
    p.write_text(json.dumps(payload, indent=1))
    click.echo(f"[rank] wrote {p}")


if __name__ == "__main__":
    main()
