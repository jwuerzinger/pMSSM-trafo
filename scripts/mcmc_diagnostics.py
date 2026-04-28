"""MCMC convergence diagnostics for Run3ModelGen-style chain outputs.

Computes split-R̂ (Gelman-Rubin, with Vehtari et al. 2021 splitting) and
bulk-ESS for each free parameter, treating each ROOT file in the input
directory as a separate chain.

Usage:
    python mcmc_diagnostics.py --data-dir /ptmp/jwuerzin/data/19250082
    python mcmc_diagnostics.py --data-dir /path/to/output --no-file-cut
    python mcmc_diagnostics.py --data-dir /path/to/output --params M_1,M_2,mu,tanb

Designed to be self-contained: imports only `numpy`, `uproot`, and (optional)
`arviz`. Drop into Run3ModelGen/source/Run3ModelGen/scripts/ and call
post-generation, or import `compute_diagnostics()` and call from inside the
generation loop.

Pass/fail thresholds (from Vehtari et al. 2021, "Rank-normalization, folding,
and localization"):
    R̂ < 1.01    well-converged
    R̂ < 1.05    acceptable for posterior summaries
    R̂ < 1.1     sometimes used, but borderline
    R̂ ≥ 1.1     not converged
    ESS < 100    posterior summaries unreliable
    ESS < 400    quantiles especially unreliable
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

# Paramater names in the order they appear in the ROOT files.
# (Free parameters that vary in the user's pMSSM scan; fixed parameters skipped.)
DEFAULT_FREE_PARAMS = [
    "IN_M_1", "IN_M_2", "IN_mu", "IN_tanb",
    "IN_At",  "IN_Ab",  "IN_Atau",
    "IN_meL", "IN_meR",
]
# Y branch and per-sample sanity filter
TARGET_BRANCH = "MO_Omega"
TRUE_VAL = 0.120
OMEGA_MIN = 0.04


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_chains(data_dir: Path, params: list[str], skip_file_cut: bool):
    import uproot
    files = sorted(glob.glob(str(data_dir / "*.root")))
    if not files:
        raise FileNotFoundError(f"no ROOT files under {data_dir}")
    chains, dropped_no_straddle, dropped_below_min = [], 0, 0
    for f in files:
        t = uproot.open(f)["susy"]
        y = t[TARGET_BRANCH].array(library="np")
        if not skip_file_cut:
            if not (np.any(y < TRUE_VAL) and np.any(y > TRUE_VAL)):
                dropped_no_straddle += 1
                continue
            if np.any(y < OMEGA_MIN):
                dropped_below_min += 1
                continue
        sp_mh = t["SP_m_h"].array(library="np")
        mask = (y > 0) & (y < 1.0) & (sp_mh != -1)
        if not mask.any():
            continue
        cols = [t[p].array(library="np")[mask] for p in params]
        chains.append(np.column_stack(cols))
    return chains, len(files), dropped_no_straddle, dropped_below_min


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def split_rhat(chains: list[np.ndarray], min_len: int) -> np.ndarray:
    """Vehtari split-R̂ across M chains, each truncated to `min_len`."""
    half = min_len // 2
    if half < 5:
        raise ValueError(f"min chain length {min_len} too short for split-R̂")
    sub = []
    for c in chains:
        c = c[:min_len]
        sub.append(c[:half])
        sub.append(c[half:half * 2])
    sub = np.array(sub)                       # (2M, N, P)
    means = sub.mean(axis=1)                  # (2M, P)
    vars_ = sub.var(axis=1, ddof=1)           # (2M, P)
    N = half
    W = vars_.mean(axis=0)
    B = N * means.var(axis=0, ddof=1)
    V = ((N - 1) / N) * W + B / N
    return np.sqrt(V / W)


def ess_bulk(chains_arr: np.ndarray) -> np.ndarray:
    """Bulk effective sample size per parameter for an (M, N, P) array.

    Uses arviz if available (Vehtari rank-normalised). Falls back to a
    naive autocorrelation-truncation estimator otherwise.
    """
    try:
        import arviz as az
    except ImportError:
        return _ess_bulk_fallback(chains_arr)
    M, N, P = chains_arr.shape
    return np.array([float(az.ess(chains_arr[:, :, p], method="bulk"))
                     for p in range(P)])


def _ess_bulk_fallback(chains_arr: np.ndarray) -> np.ndarray:
    """Naive multi-chain ESS via autocorrelation; arviz-free fallback.

    Less rigorous than arviz's rank-normalised version but adequate for
    detecting catastrophic mixing.
    """
    M, N, P = chains_arr.shape
    out = np.empty(P)
    for p in range(P):
        # Per-chain autocorrelation at lag k, averaged over chains
        x = chains_arr[:, :, p].astype(np.float64)
        x_mean = x.mean(axis=1, keepdims=True)
        x_centered = x - x_mean
        var = (x_centered ** 2).sum(axis=1) / N
        var = np.where(var > 0, var, 1.0)
        # truncate sum at lag where avg autocorr drops below 0.05
        rho_sum = 0.0
        for k in range(1, N // 4):
            rho_k = ((x_centered[:, :-k] * x_centered[:, k:]).sum(axis=1)
                     / (N - k) / var).mean()
            if rho_k < 0.05:
                break
            rho_sum += rho_k
        tau = 1.0 + 2.0 * rho_sum
        out[p] = M * N / max(tau, 1.0)
    return out


def compute_diagnostics(
    data_dir: str | Path,
    params: list[str] = DEFAULT_FREE_PARAMS,
    skip_file_cut: bool = False,
    print_table: bool = True,
) -> dict:
    """Returns {param: {"rhat": float, "ess_bulk": float}} plus context."""
    data_dir = Path(data_dir)
    chains, n_files, dropped_ns, dropped_bm = _load_chains(
        data_dir, params, skip_file_cut
    )
    if not chains:
        raise RuntimeError(f"no usable chains under {data_dir}")

    long_chains = [c for c in chains if len(c) >= 100]
    min_len = min(len(c) for c in long_chains)
    arr = np.stack([c[:min_len] for c in long_chains], axis=0)
    M, N, P = arr.shape

    rhat = split_rhat(long_chains, min_len)
    ess = ess_bulk(arr)
    raw_total = M * N

    out = {
        "context": {
            "data_dir": str(data_dir),
            "n_files": n_files,
            "n_chains_used": M,
            "min_chain_len": int(min_len),
            "raw_samples": int(raw_total),
            "dropped_no_straddle": dropped_ns,
            "dropped_below_omega_min": dropped_bm,
            "skip_file_cut": skip_file_cut,
        },
        "per_param": {
            params[p]: {
                "rhat": float(rhat[p]),
                "ess_bulk": float(ess[p]),
                "ess_efficiency": float(ess[p]) / raw_total,
            }
            for p in range(P)
        },
    }

    if print_table:
        ctx = out["context"]
        print(f"\n[mcmc-diag] data_dir={ctx['data_dir']}")
        print(f"[mcmc-diag] using {ctx['n_chains_used']} of {ctx['n_files']} files "
              f"(skip_file_cut={ctx['skip_file_cut']})")
        if not ctx["skip_file_cut"]:
            print(f"[mcmc-diag] file-cut dropped {ctx['dropped_no_straddle']} "
                  f"(no Ω-straddle) + {ctx['dropped_below_omega_min']} (Ω<{OMEGA_MIN})")
        print(f"[mcmc-diag] truncated each chain to {ctx['min_chain_len']} samples; "
              f"raw total = {ctx['raw_samples']}")
        print()
        print(f"  {'param':10s} {'split-R̂':>9s} {'ESS_bulk':>10s} {'ESS/N':>8s}  flag")
        for name in params:
            r = out["per_param"][name]["rhat"]
            e = out["per_param"][name]["ess_bulk"]
            eff = out["per_param"][name]["ess_efficiency"]
            flag = ""
            if r > 1.1 or e < 100:
                flag = "BAD"
            elif r > 1.05 or e < 400:
                flag = "borderline"
            else:
                flag = "ok"
            print(f"  {name:10s} {r:9.3f} {e:10.0f} {eff:7.2%}  {flag}")

        worst_r = max(out["per_param"][p]["rhat"] for p in params)
        worst_ess = min(out["per_param"][p]["ess_bulk"] for p in params)
        print()
        print(f"[mcmc-diag] worst R̂ = {worst_r:.3f} "
              f"({'NOT CONVERGED' if worst_r > 1.1 else 'converged' if worst_r < 1.05 else 'borderline'})")
        print(f"[mcmc-diag] min ESS_bulk = {worst_ess:.0f} "
              f"({'unreliable' if worst_ess < 100 else 'borderline' if worst_ess < 400 else 'ok'})")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Directory containing per-chain ROOT files (one chain per file).")
    p.add_argument("--params", default=",".join(DEFAULT_FREE_PARAMS),
                   help="Comma-separated branch names to diagnose. "
                        f"Default: {','.join(DEFAULT_FREE_PARAMS)}")
    p.add_argument("--no-file-cut", action="store_true",
                   help="Skip the file-level cuts (chains-must-straddle-Ω, "
                        "no-values-below-0.04). Lets you diagnose the full sampler.")
    return p.parse_args()


def main():
    args = _parse_args()
    params = [s.strip() for s in args.params.split(",") if s.strip()]
    compute_diagnostics(
        data_dir=args.data_dir,
        params=params,
        skip_file_cut=args.no_file_cut,
        print_table=True,
    )


if __name__ == "__main__":
    main()
