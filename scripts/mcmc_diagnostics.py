"""MCMC convergence diagnostics for Run3ModelGen-style chain outputs.

Computes split-R̂ (Gelman-Rubin, with Vehtari et al. 2021 splitting), bulk-ESS
and autocorrelation length τ for each free parameter, treating each ROOT file
in the input directory as a separate chain.

Also supports treating each AL-run's cumulative training set (X in state.pt)
as a chain, letting you compare the AL model datasets against the MCMC
reference.

Usage:
    # MCMC diagnostics on the reference set:
    python mcmc_diagnostics.py --data-dir /ptmp/jwuerzin/data/19250082

    # AL vs MCMC comparison plot (best-per-model picks from a manifest):
    python mcmc_diagnostics.py \\
        --data-dir /ptmp/jwuerzin/data/19250082 \\
        --al-manifest /ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv \\
        --output-dir /ptmp/jwuerzin/analysis/all_runs/

    # Custom picks (skip auto-selection):
    python mcmc_diagnostics.py --data-dir ... --al-manifest ... \\
        --al-picks transformer:entropy_batch:cold,deep_gp:entropy_batch:warm

Designed to be self-contained: imports only `numpy`, `uproot`, and (optional)
`arviz`, plus `torch` + `matplotlib` when the AL comparison is requested.

Pass/fail thresholds (from Vehtari et al. 2021, "Rank-normalization, folding,
and localization"):
    R̂ < 1.01    well-converged
    R̂ < 1.05    acceptable for posterior summaries
    R̂ < 1.1     sometimes used, but borderline
    R̂ ≥ 1.1     not converged
    ESS < 100    posterior summaries unreliable
    ESS < 400    quantiles especially unreliable
    τ = M·N / ESS_bulk   (samples per effective sample; larger = more correlated)
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

# ── AL comparison: which (strategy, warm_start) to pick per model when the
# user asks for AL-vs-MCMC diagnostics but doesn't supply an override.
# Chosen from prior sweep analysis: highest mean-final hit-rate @10% tolerance.
DEFAULT_AL_PICKS = {
    "transformer":     ("entropy_batch",  "cold"),
    "deep_gp":         ("entropy_batch",  "warm"),
    "exact_gp":        ("entropy_batch",  "warm"),
    "dnn":             ("entropy_batch",  "cold"),
    "dnn_match_trafo": ("top_k",          "cold"),
    "tabpfn":          ("top_k_tol_only", "tabpfn"),
}

# Full 19-parameter ordering saved to state.pt's X tensor. Only 9 are free.
PARAM_ORDER = [
    "IN_meL", "IN_meR", "IN_mtauL", "IN_mtauR",
    "IN_mqL1", "IN_muR", "IN_mdR", "IN_mqL3",
    "IN_mtR", "IN_mbR", "IN_M_1", "IN_M_2",
    "IN_mu", "IN_M_3", "IN_At", "IN_Ab",
    "IN_Atau", "IN_mA", "IN_tanb",
]

# Friendly display labels for the comparison plot.
MODEL_DISPLAY = {
    "transformer":     "Transformer",
    "deep_gp":         "Deep GP",
    "exact_gp":        "Exact GP",
    "dnn":             "DNN",
    "dnn_match_trafo": "DNN (matched)",
    "tabpfn":          "TabPFN",
}


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_chains(data_dir: Path, params: list[str], skip_file_cut: bool,
                 require_neutralino_lsp: bool = False):
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
        if require_neutralino_lsp and "SP_LSP_type" in t.keys():
            lsp = t["SP_LSP_type"].array(library="np")
            mask = mask & ((lsp == 1) | (lsp == 2) | (lsp == 3))
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


def diagnostics_from_chains(
    chains: list[np.ndarray],
    params: list[str],
    min_len_threshold: int = 100,
) -> dict:
    """Core diagnostic: R̂, ESS_bulk, τ (autocorr length) per parameter.

    Args:
        chains: list of (n_samples_i, n_params) arrays — one entry per chain.
        params: parameter names, len == chains[0].shape[1].
        min_len_threshold: chains shorter than this are dropped (too little
            data for split-R̂). Set to 5 to allow short AL runs.

    Returns:
        dict with 'context' and 'per_param' keys (see compute_diagnostics).
        None if no chains survive the length filter.
    """
    long_chains = [c for c in chains if len(c) >= min_len_threshold]
    if not long_chains:
        return None
    min_len = min(len(c) for c in long_chains)
    arr = np.stack([c[:min_len] for c in long_chains], axis=0)
    M, N, P = arr.shape

    rhat = split_rhat(long_chains, min_len)
    ess = ess_bulk(arr)
    raw_total = M * N
    tau = raw_total / np.maximum(ess, 1.0)   # samples per effective sample

    return {
        "context": {
            "n_chains_used": M,
            "min_chain_len": int(min_len),
            "raw_samples": int(raw_total),
        },
        "per_param": {
            params[p]: {
                "rhat": float(rhat[p]),
                "ess_bulk": float(ess[p]),
                "ess_efficiency": float(ess[p]) / raw_total,
                "tau": float(tau[p]),
            }
            for p in range(P)
        },
    }


def compute_diagnostics(
    data_dir: str | Path,
    params: list[str] = DEFAULT_FREE_PARAMS,
    skip_file_cut: bool = False,
    print_table: bool = True,
    require_neutralino_lsp: bool = False,
) -> dict:
    """Returns {param: {"rhat": float, "ess_bulk": float}} plus context.

    When ``require_neutralino_lsp`` is True, additionally requires
    ``SP_LSP_type in {1, 2, 3}`` on every ROOT sample.
    """
    data_dir = Path(data_dir)
    chains, n_files, dropped_ns, dropped_bm = _load_chains(
        data_dir, params, skip_file_cut,
        require_neutralino_lsp=require_neutralino_lsp,
    )
    if not chains:
        raise RuntimeError(f"no usable chains under {data_dir}")

    out = diagnostics_from_chains(chains, params, min_len_threshold=100)
    if out is None:
        raise RuntimeError("no chains long enough for split-R̂ (< 100 samples)")
    # Enrich context with MCMC-specific fields (n_files, filter drops).
    out["context"].update({
        "data_dir": str(data_dir),
        "n_files": n_files,
        "dropped_no_straddle": dropped_ns,
        "dropped_below_omega_min": dropped_bm,
        "skip_file_cut": skip_file_cut,
    })

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
# AL comparison: treat each seed's cumulative training set as one chain
# ──────────────────────────────────────────────────────────────────────────────

def _load_al_chains(run_dirs: list[str | Path],
                    param_indices: list[int],
                    require_neutralino_lsp: bool = False) -> list[np.ndarray]:
    """Load `state.pt` from each run dir; return one (N, len(param_indices))
    array per dir. The cumulative training X is in physical units, in the
    order points were added (initial random block, then per-iteration picks).

    If ``require_neutralino_lsp`` is True, rows whose F (LSP composition)
    tensor contains any NaN are dropped — these correspond to non-neutralino
    LSPs (e.g. sneutrinos) as coerced by ``pmssm.data._load_lsp_fracs``.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("torch required to load AL state.pt files") from e
    chains: list[np.ndarray] = []
    for d in run_dirs:
        sp = Path(d) / "state.pt"
        if not sp.exists():
            continue
        state = torch.load(sp, map_location="cpu", weights_only=False)
        X = state.get("X")
        if X is None:
            continue
        X = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
        if require_neutralino_lsp:
            F = state.get("F")
            if F is None:
                # Older runs without F — leave as-is (no veto possible).
                pass
            else:
                F = F.detach().cpu().numpy() if hasattr(F, "detach") else np.asarray(F)
                keep = np.isfinite(F).all(axis=1)
                X = X[keep]
        chains.append(X[:, list(param_indices)])
    return chains


def _picks_from_manifest(manifest_csv: str | Path,
                         picks_spec: dict[str, tuple[str, str]]
                         ) -> dict[str, list[str]]:
    """Read sweep_manifest.csv and, for each (model, strategy, warm) in
    picks_spec, return a list of completed run_dirs (5 seeds typically).
    """
    import csv
    out: dict[str, list[str]] = {m: [] for m in picks_spec}
    with open(manifest_csv) as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "completed":
                continue
            m = r.get("model")
            if m not in picks_spec:
                continue
            s, w = picks_spec[m]
            if r.get("strategy") == s and r.get("warm_start") == w:
                out[m].append(r["expected_run_dir"])
    return out


def compare_and_plot(mcmc_dir: str | Path,
                     picks_spec: dict[str, tuple[str, str]],
                     manifest_csv: str | Path,
                     out_dir: str | Path,
                     params: list[str] = DEFAULT_FREE_PARAMS,
                     print_summary: bool = True,
                     require_neutralino_lsp: bool = False) -> Path:
    """Compute diagnostics for MCMC + each AL pick, render comparison plot.

    Returns the path to the written PNG. When ``require_neutralino_lsp`` is
    True, both the MCMC reference set and every AL training set have their
    non-neutralino rows (sneutrinos etc.) vetoed before chain statistics
    are computed.
    """
    # 1) MCMC diagnostics
    mcmc = compute_diagnostics(mcmc_dir, params=params, print_table=False,
                               require_neutralino_lsp=require_neutralino_lsp)

    # 2) AL diagnostics per model
    idx_map = {p: PARAM_ORDER.index(p) for p in params}
    param_indices = [idx_map[p] for p in params]

    run_dirs_per_model = _picks_from_manifest(manifest_csv, picks_spec)
    al_results: dict[str, dict] = {}
    al_meta: dict[str, dict] = {}
    for model, run_dirs in run_dirs_per_model.items():
        if not run_dirs:
            al_meta[model] = {"n_seeds": 0, "picked": picks_spec[model]}
            continue
        chains = _load_al_chains(run_dirs, param_indices,
                                 require_neutralino_lsp=require_neutralino_lsp)
        res = diagnostics_from_chains(chains, params, min_len_threshold=100)
        if res is None:
            al_meta[model] = {"n_seeds": len(chains), "picked": picks_spec[model],
                              "note": "no chain long enough"}
            continue
        al_results[model] = res
        al_meta[model] = {
            "n_seeds": res["context"]["n_chains_used"],
            "min_chain_len": res["context"]["min_chain_len"],
            "picked": picks_spec[model],
        }

    # 3) Assemble source table (MCMC first, then models by pick order)
    sources: list[tuple[str, dict]] = [("MCMC (ref)", mcmc)]
    for model in picks_spec:
        if model in al_results:
            s, w = picks_spec[model]
            n = al_meta[model]["n_seeds"]
            label = f"{MODEL_DISPLAY.get(model, model)} — {s}/{w} (n={n})"
            sources.append((label, al_results[model]))

    # 4) Render + write JSON
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "mcmc_diagnostics_comparison.png"
    _render_comparison_plot(sources, params, png_path)

    import json
    json_path = out_dir / "mcmc_diagnostics_comparison.json"
    dump = {
        "mcmc": mcmc,
        "al": {m: {"meta": al_meta.get(m, {}), "diagnostics": al_results.get(m)}
               for m in picks_spec},
        "params": params,
    }
    with open(json_path, "w") as fh:
        json.dump(dump, fh, indent=2, default=str)

    if print_summary:
        print(f"\n[compare] wrote {png_path}")
        print(f"[compare] wrote {json_path}")
        print(f"[compare] sources included: {len(sources)}")
        for label, _ in sources:
            print(f"           - {label}")
    return png_path


def _render_comparison_plot(sources: list[tuple[str, dict]],
                            params: list[str],
                            out_path: Path) -> None:
    """2-panel heatmap: R̂ (top) and autocorrelation length τ (bottom).
    Rows = data sources (MCMC + each model), cols = free parameters.
    """
    # Localised matplotlib import; preload pixi's libstdc++ if that's what we
    # need for the C++-extension modules to load (HPC login node quirk).
    try:
        import ctypes, os
        libpath = Path(sys.prefix) / "lib" / "libstdc++.so.6"
        if libpath.exists():
            ctypes.CDLL(str(libpath), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_src = len(sources); n_par = len(params)
    rhat = np.full((n_src, n_par), np.nan)
    tau  = np.full((n_src, n_par), np.nan)
    for i, (_, r) in enumerate(sources):
        for j, p in enumerate(params):
            if r is None or p not in r["per_param"]:
                continue
            rhat[i, j] = r["per_param"][p]["rhat"]
            tau[i, j]  = r["per_param"][p]["tau"]

    row_labels = [s for (s, _) in sources]
    fig, axes = plt.subplots(2, 1, figsize=(1.1 * n_par + 4, 0.55 * n_src + 3))

    # R̂ panel — colour scale centred on convergence thresholds
    im_r = axes[0].imshow(rhat, aspect="auto", cmap="RdYlGn_r",
                          vmin=1.0, vmax=1.10)
    axes[0].set_yticks(range(n_src)); axes[0].set_yticklabels(row_labels)
    axes[0].set_xticks(range(n_par))
    axes[0].set_xticklabels(params, rotation=30, ha="right")
    axes[0].set_title(r"split-$\hat{R}$ (Gelman-Rubin)  —  "
                      r"$<1.01$ converged · $<1.05$ acceptable · $>1.1$ not converged",
                      fontsize=11)
    for i in range(n_src):
        for j in range(n_par):
            if np.isfinite(rhat[i, j]):
                axes[0].text(j, i, f"{rhat[i, j]:.2f}",
                             ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im_r, ax=axes[0], label=r"$\hat{R}$", shrink=0.85, pad=0.02)

    # τ panel — log-normalised colours so both MCMC (~1-3) and AL (potentially
    # much larger) read well.
    from matplotlib.colors import LogNorm
    tau_pos = np.where(np.isfinite(tau) & (tau > 0), tau, np.nan)
    finite = tau_pos[np.isfinite(tau_pos)]
    vmin = max(1.0, np.nanmin(finite)) if finite.size else 1.0
    vmax = max(vmin + 1e-3, np.nanmax(finite)) if finite.size else 10.0
    im_t = axes[1].imshow(tau_pos, aspect="auto", cmap="RdYlGn_r",
                          norm=LogNorm(vmin=vmin, vmax=vmax))
    axes[1].set_yticks(range(n_src)); axes[1].set_yticklabels(row_labels)
    axes[1].set_xticks(range(n_par))
    axes[1].set_xticklabels(params, rotation=30, ha="right")
    axes[1].set_title(r"Autocorrelation length  $\tau = M \cdot N / \mathrm{ESS}_\mathrm{bulk}$  "
                      r"(samples per effective sample; lower = better)",
                      fontsize=11)
    for i in range(n_src):
        for j in range(n_par):
            v = tau[i, j]
            if np.isfinite(v):
                lbl = f"{v:.1f}" if v < 100 else f"{v:.0f}"
                axes[1].text(j, i, lbl,
                             ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im_t, ax=axes[1], label=r"$\tau$", shrink=0.85, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_picks_arg(spec: str) -> dict[str, tuple[str, str]]:
    """'transformer:entropy_batch:cold,deep_gp:entropy_batch:warm' → dict."""
    out: dict[str, tuple[str, str]] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad pick spec {chunk!r}; want model:strategy:warm")
        m, s, w = parts
        out[m] = (s, w)
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
    # AL comparison mode
    p.add_argument("--al-manifest", type=Path, default=None,
                   help="sweep_manifest.csv path. Enables AL-vs-MCMC comparison "
                        "plot (writes mcmc_diagnostics_comparison.png/json).")
    p.add_argument("--al-picks", default=None,
                   help="Override picks: 'model:strategy:warm,...'. "
                        "Defaults to DEFAULT_AL_PICKS (best-per-model from sweep).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write the comparison PNG/JSON. Required with "
                        "--al-manifest.")
    p.add_argument("--require-neutralino-lsp", action="store_true",
                   help="Post-hoc veto of non-neutralino LSPs: require "
                        "SP_LSP_type in {1,2,3} on MCMC samples and drop "
                        "state.pt rows whose F is NaN on the AL side.")
    return p.parse_args()


def main():
    args = _parse_args()
    params = [s.strip() for s in args.params.split(",") if s.strip()]
    compute_diagnostics(
        data_dir=args.data_dir,
        params=params,
        skip_file_cut=args.no_file_cut,
        print_table=True,
        require_neutralino_lsp=args.require_neutralino_lsp,
    )
    if args.al_manifest is not None:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required with --al-manifest")
        picks = (_parse_picks_arg(args.al_picks) if args.al_picks
                 else dict(DEFAULT_AL_PICKS))
        compare_and_plot(
            mcmc_dir=args.data_dir,
            picks_spec=picks,
            manifest_csv=args.al_manifest,
            out_dir=args.output_dir,
            params=params,
            require_neutralino_lsp=args.require_neutralino_lsp,
        )


if __name__ == "__main__":
    main()
