"""Distinct in-band models per simulator call: AL vs MCMC vs random scan.

Reproduces the paper's yield-comparison table (Table `tab:yield`): the number
of DISTINCT models inside the |Omega - 0.12|/0.12 < tol band produced per
simulator evaluation, with every method charged its full overheads:

  * random scan   — band prevalence in the valid pool × Run3ModelGen validity
                    rate p_valid (in-band valid models per attempt);
  * emcee MCMC    — distinct in-band (M_1, M_2, mu) states in the stored
                    chains ÷ total proposals (iterations × nwalkers, parsed
                    from diag/diagnostics.txt, i.e. INCLUDING burn-in); the
                    post-burn-in marginal variant divides by stored rows;
  * AL picks      — final cumulative hits/desired value in attempt units
                    (initial random block deflated by p_valid, validity
                    dropouts included), one value per best-per-model cell,
                    averaged over seeds.

Raw acceptance rate and hit rate are not commensurable across methods; this
per-call distinct yield is. Note it scores dataset-building efficiency only:
MCMC repeats carry posterior weight and are the correct output for Bayesian
inference, which the coverage-driven AL scan does not address.

Usage:
    python scripts/compute_yield_comparison.py \\
        --output-dir /ptmp/jwuerzin/analysis/all_runs/ \\
        --require-neutralino-lsp
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyse_runs import filter_run_neutralino_lsp, load_run  # noqa: E402
from mcmc_diagnostics import DEFAULT_AL_PICKS  # noqa: E402
import plot_hit_rate_trajectories_multiseed as phr  # noqa: E402

_DIAG_ITER_RE = re.compile(r"iterations=(\d+)\s+nwalkers=(\d+)")


def mcmc_yield(mcmc_data_dir: str, tol: float) -> dict:
    """Distinct in-band states per proposal for an emcee-era dataset.

    Distinctness is keyed on (IN_M_1, IN_M_2, IN_mu) — continuous parameters,
    so collisions between genuinely different models are negligible while
    walker-repeat rows collapse exactly.
    """
    import uproot

    true_value = 0.12
    tot_rows = 0
    distinct_inband = 0
    for fn in sorted(glob.glob(f"{mcmc_data_dir}/*.root")):
        t = uproot.open(fn)["susy"]
        om = t["MO_Omega"].array(library="np")
        key = np.stack([t["IN_M_1"].array(library="np"),
                        t["IN_M_2"].array(library="np"),
                        t["IN_mu"].array(library="np")], axis=1)
        inband = np.abs(om - true_value) / true_value < tol
        tot_rows += len(om)
        distinct_inband += len(np.unique(key[inband], axis=0))

    diag_txt = Path(mcmc_data_dir) / "diag" / "diagnostics.txt"
    proposals = None
    if diag_txt.exists():
        pairs = _DIAG_ITER_RE.findall(diag_txt.read_text(errors="ignore"))
        if pairs:
            proposals = sum(int(i) * int(w) for i, w in pairs)
    if proposals is None:
        click.echo(f"[yield] WARNING: {diag_txt} missing or unparseable — "
                   "cannot count burn-in proposals; whole-run yield "
                   "unavailable, reporting post-burn-in only", err=True)
    return {
        "stored_rows": tot_rows,
        "distinct_inband": distinct_inband,
        "total_proposals": proposals,
        "yield_whole_run": (distinct_inband / proposals) if proposals else None,
        "yield_post_burnin": distinct_inband / tot_rows,
    }


def al_yields(manifest: str, tol: float, require_neutralino_lsp: bool,
              p_valid: float) -> dict:
    """Final cumulative hits/desired (attempt units) per best-per-model cell."""
    phr._DESIRED_P_VALID = p_valid
    rows = [r for r in csv.DictReader(open(manifest))
            if r["status"] == "completed"]
    out = {}
    for model, (strat, warm) in DEFAULT_AL_PICKS.items():
        finals = []
        for r in rows:
            if (r["model"], r["strategy"], r["warm_start"]) != (model, strat, warm):
                continue
            try:
                run = load_run(r["expected_run_dir"])
            except Exception as e:
                click.echo(f"[yield]   skip {r['expected_run_dir']}: {e}", err=True)
                continue
            if require_neutralino_lsp:
                run = filter_run_neutralino_lsp(run)
            _, rates = phr._hits_per_desired_trajectory(run, 0.12, tol)
            if rates:
                finals.append(rates[-1])
        if finals:
            f = np.asarray(finals, dtype=float)
            sem = f.std(ddof=1) / np.sqrt(len(f)) if len(f) > 1 else 0.0
            out[model] = {"strategy": strat, "warm": warm, "n_seeds": len(f),
                          "yield": float(f.mean()), "sem": float(sem)}
    return out


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True)
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358",
              show_default=True,
              help="Random-scan pool; band prevalence × p_valid gives the "
                   "per-attempt baseline yield.")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs/",
              show_default=True,
              help="Where yield_comparison.json is written (also used as the "
                   "Y-pool .npy cache dir).")
@click.option("--tolerance", default=0.10, type=float, show_default=True)
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True)
def main(manifest, mcmc_data_dir, baseline_data_dir, output_dir, tolerance,
         require_neutralino_lsp):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    true_value = 0.12

    # ── random-scan baseline ────────────────────────────────────────────────
    Y_full = phr._load_y_full(baseline_data_dir, "DMRD", out_dir)
    prevalence = float(np.mean(np.abs(Y_full - true_value) / true_value < tolerance))
    run_dirs = [r["expected_run_dir"] for r in csv.DictReader(open(manifest))
                if r["status"] == "completed"]
    p_valid, n_valid, n_total, src = phr._extract_validity_rate(run_dirs)
    if p_valid is None:
        raise click.ClickException("could not extract p_valid from any run log")
    random_yield = prevalence * p_valid
    click.echo(f"[yield] p_valid={p_valid:.4f} ({n_valid}/{n_total}, {src})")
    click.echo(f"[yield] pool prevalence@{tolerance:.0%}={prevalence:.4f} "
               f"-> random per-attempt yield {random_yield:.4f}")

    # ── MCMC ────────────────────────────────────────────────────────────────
    m = mcmc_yield(mcmc_data_dir, tolerance)
    click.echo(f"[yield] MCMC: {m['distinct_inband']:,} distinct in-band states, "
               f"{m['stored_rows']:,} stored rows, "
               f"{m['total_proposals'] and format(m['total_proposals'], ',')} proposals")

    # ── AL picks ────────────────────────────────────────────────────────────
    al = al_yields(manifest, tolerance, require_neutralino_lsp, p_valid)

    # ── table ───────────────────────────────────────────────────────────────
    rows = [("Random scan", random_yield)]
    if m["yield_whole_run"]:
        rows.append(("emcee reference (incl. burn-in)", m["yield_whole_run"]))
    rows.append(("emcee reference (excl. burn-in)", m["yield_post_burnin"]))
    rows += [(f"{k} ({v['strategy']}/{v['warm']}, n={v['n_seeds']})", v["yield"])
             for k, v in sorted(al.items(), key=lambda kv: kv[1]["yield"])]
    click.echo(f"\n  {'method':<45s} {'yield/call':>10s} {'vs random':>10s}")
    for name, y in rows:
        click.echo(f"  {name:<45s} {y:10.4f} {y/random_yield:9.1f}x")

    result = {"tolerance": tolerance, "true_value": true_value,
              "require_neutralino_lsp": require_neutralino_lsp,
              "p_valid": p_valid, "pool_prevalence": prevalence,
              "random_yield_per_attempt": random_yield,
              "mcmc": m, "al": al}
    json_path = out_dir / "yield_comparison.json"
    json_path.write_text(json.dumps(result, indent=2))
    click.echo(f"\n[yield] wrote {json_path}")


if __name__ == "__main__":
    main()
