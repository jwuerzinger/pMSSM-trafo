"""
Run3ModelGen interface for generating new pMSSM model points.

This module provides functions to:
- Generate models using Run3ModelGen
- Load generated data from ROOT ntuples
- Save selected candidate points to CSV
"""

import subprocess
import shutil
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .config import PARAM_ORDER, CSV_TO_MODELGEN


def _run_modelgen(df, output_dir, logger, label=""):
    """
    Run genModels.py for a DataFrame of candidates in output_dir.

    Args:
        df: DataFrame with parameter columns
        output_dir: Directory to save generated models
        logger: Logger instance
        label: Optional label for logging (e.g., "worker 0")

    Returns:
        Path to the generated ROOT ntuple, or None on failure
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"[{label}] " if label else ""
    n_models = len(df)

    # Build Run3ModelGen config
    config = {
        "prior": "fixed",
        "num_models": n_models,
        "isGMSB": False,
        "parameters": {},
        "steps": [
            {"name": "prep_input", "output_dir": "input", "prefix": "IN"},
            {"name": "SPheno", "input_dir": "input", "output_dir": "SPheno", "log_dir": "SPheno_log", "prefix": "SP"},
            {"name": "micromegas", "input_dir": "SPheno", "output_dir": "micromegas", "prefix": "MO"},
        ],
    }

    for csv_col, modelgen_param in CSV_TO_MODELGEN.items():
        if csv_col in df.columns:
            config["parameters"][modelgen_param] = df[csv_col].tolist()
        else:
            logger.warning(f"{prefix}Column {csv_col} not found in DataFrame")

    config_path = output_dir / "modelgen_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=None)

    scan_dir = output_dir / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).parent.parent.resolve()
    run3modelgen_dir = project_root / "Run3ModelGen"
    setup_script = run3modelgen_dir / "build" / "setup.sh"

    if not setup_script.exists():
        logger.error(f"{prefix}Run3ModelGen setup script not found: {setup_script}")
        return None

    pixi_path = shutil.which("pixi")
    if pixi_path is None:
        for path in [Path.home() / ".pixi" / "bin" / "pixi", Path("/u/jwuerzin/.pixi/bin/pixi")]:
            if path.exists():
                pixi_path = str(path)
                break
    if pixi_path is None:
        logger.error(f"{prefix}Could not find pixi executable")
        return None

    # Use relative paths to avoid exceeding SPheno's Fortran CHARACTER buffer
    # limit (~120 chars). Absolute paths for retry directories can exceed this.
    cmd = f"source {setup_script} && cd {output_dir} && genModels.py --config_file modelgen_config.yaml --scan_dir scan"
    logger.info(f"{prefix}Starting model generation ({n_models} models) in {scan_dir}...")

    try:
        result = subprocess.run(
            [pixi_path, "run", "bash", "-c", cmd],
            cwd=str(run3modelgen_dir),
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            logger.error(f"{prefix}genModels.py failed with return code {result.returncode}")
            logger.error(f"{prefix}stderr: {result.stderr[-500:]}")
            return None

        logger.info(f"{prefix}Model generation complete")
        logger.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    except subprocess.TimeoutExpired:
        logger.error(f"{prefix}Model generation timed out after 1 hour")
        return None
    except Exception as e:
        logger.error(f"{prefix}Model generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    ntuple_files = list(scan_dir.glob("*.root"))
    if ntuple_files:
        logger.info(f"{prefix}Generated ntuple: {ntuple_files[0]}")
        return ntuple_files[0]
    else:
        logger.warning(f"{prefix}No ROOT ntuple found after generation")
        return None


def generate_models_from_csv(csv_path, output_dir, logger, n_workers=1):
    """
    Generate pMSSM models using Run3ModelGen from selected points CSV.

    When n_workers > 1, the candidate list is split evenly across workers
    and all genModels.py subprocesses are launched in parallel.

    Args:
        csv_path: Path to selected_points.csv
        output_dir: Directory to save generated models (iteration_XXX)
        logger: Logger instance
        n_workers: Number of parallel genModels.py processes (default: 1)

    Returns:
        List of Paths to generated ROOT ntuples (empty list on total failure)
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Read {len(df)} points from {csv_path}")

    output_dir = Path(output_dir).resolve()

    if n_workers <= 1:
        ntuple = _run_modelgen(df, output_dir, logger)
        return [ntuple] if ntuple is not None else []

    # Split candidates evenly across workers
    chunk_size = int(np.ceil(len(df) / n_workers))
    chunks = [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(n_workers)]
    chunks = [c for c in chunks if len(c) > 0]  # drop empty tail chunks if len(df) < n_workers
    n_actual = len(chunks)
    logger.info(f"Splitting {len(df)} models across {n_actual} parallel workers "
                f"(~{len(chunks[0])} models each)...")

    def run_worker(args):
        i, chunk = args
        worker_dir = output_dir / f"worker_{i:02d}"
        return _run_modelgen(chunk, worker_dir, logger, label=f"worker {i}")

    with ThreadPoolExecutor(max_workers=n_actual) as executor:
        results = list(executor.map(run_worker, enumerate(chunks)))

    ntuple_paths = [r for r in results if r is not None]
    n_failed = n_actual - len(ntuple_paths)
    if n_failed:
        logger.warning(f"{n_failed}/{n_actual} generation workers failed")
    logger.info(f"Parallel generation complete: {len(ntuple_paths)}/{n_actual} workers succeeded")
    return ntuple_paths


def load_generated_data(ntuple_path, logger, return_lsp_fracs=False):
    """
    Load newly generated data from ROOT ntuple.

    Args:
        ntuple_path: Path to generated ROOT file
        logger: Logger instance
        return_lsp_fracs: If True, also return (N, 3) tensor of neutralino
            [bino, wino, higgsino] fractions; NaN rows when LSP is not a
            neutralino or fraction branches are missing.

    Returns:
        (X, Y) tensors — or (X, Y, lsp_fracs) when return_lsp_fracs=True.
        Returns (None, None[, None]) if loading failed.
    """
    import uproot

    def _none_return():
        return (None, None, None) if return_lsp_fracs else (None, None)

    try:
        root_file = uproot.open(str(ntuple_path))
        # Run3ModelGen uses 'susy' as tree name
        tree = root_file["susy"]

        # Check if required branches exist (SPheno or micromegas may have failed for all models)
        for required in ("MO_Omega", "SP_m_h"):
            if required not in tree.keys():
                logger.warning(f"{required} not found in ntuple - SPheno or micromegas may have failed for all models")
                logger.warning("No new training data available from this generation")
                return _none_return()

        # Extract input parameters (same order as PARAM_ORDER)
        branches = PARAM_ORDER

        X = np.column_stack([tree[b].array(library="np") for b in branches])
        Y = tree["MO_Omega"].array(library="np")
        sp_mh = tree["SP_m_h"].array(library="np")

        # Base filter: valid relic density, sub-dominant DM, valid Higgs mass.
        mask = (Y > 0) & (Y < 1.0) & (sp_mh != -1)

        # Additional filter: require a neutralino LSP (bino/wino/higgsino;
        # SP_LSP_type in {1, 2, 3} per Run3ModelGen's ntupling convention;
        # values >=1e6 correspond to non-neutralino LSPs, e.g. sneutrinos
        # SP_LSP_type in {1000012, 1000014}). This vetoes phenomenologically
        # excluded LSP candidates before they enter the AL training set.
        # If the branch is missing we fall back to the base filter and warn
        # once so this remains a soft addition rather than a hard dependency.
        if "SP_LSP_type" in tree.keys():
            lsp_type = tree["SP_LSP_type"].array(library="np")
            neutralino = (lsp_type == 1) | (lsp_type == 2) | (lsp_type == 3)
            n_before_lsp = int(mask.sum())
            mask = mask & neutralino
            n_dropped = n_before_lsp - int(mask.sum())
            if n_dropped:
                logger.info(f"Neutralino-LSP filter dropped {n_dropped} non-neutralino points "
                            f"(sneutrino or other) from {n_before_lsp} otherwise-valid models")
        else:
            logger.warning("SP_LSP_type branch missing from ntuple; neutralino-LSP filter "
                           "skipped (falling back to base filter only)")

        X_t = torch.from_numpy(X[mask]).float()
        Y_t = torch.from_numpy(Y[mask]).float().unsqueeze(1)

        if len(X_t) == 0:
            logger.warning("No valid models found after filtering "
                           "(MO_Omega > 0 & < 1 & SP_m_h != -1 & SP_LSP_type in {1,2,3})")
            return _none_return()

        logger.info(f"Loaded {len(X_t)} valid models from ntuple (filtered from {len(mask)} total)")

        if return_lsp_fracs:
            cols = []
            for b in ("SP_LSP_Bino_frac", "SP_LSP_Wino_frac", "SP_LSP_Higgsino_frac"):
                if b in tree.keys():
                    cols.append(tree[b].array(library="np").astype(np.float32))
                else:
                    cols.append(np.full(mask.shape, np.nan, dtype=np.float32))
            fracs = np.stack(cols, axis=1)[mask]
            bad = (~np.isfinite(fracs).all(axis=1)) | (fracs < 0).any(axis=1)
            if bad.any():
                fracs[bad] = np.nan
            return X_t, Y_t, torch.from_numpy(fracs).float()
        return X_t, Y_t

    except Exception as e:
        logger.error(f"Failed to load generated data: {e}")
        return _none_return()


def save_selected_points(X_candidates, uncertainties, indices, output_dir, iteration):
    """
    Save selected points to CSV file.

    Args:
        X_candidates: Candidate pool tensor (N, D)
        uncertainties: Uncertainty values (N, 1) or (N,)
        indices: Indices of selected points
        output_dir: Output directory
        iteration: Current iteration number

    Returns:
        csv_path: Path to saved CSV file
    """
    iter_dir = output_dir / f"iteration_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame with parameter names
    param_names = [p.replace("IN_", "") for p in PARAM_ORDER]

    selected_X = X_candidates[indices].numpy() if torch.is_tensor(X_candidates[indices]) else X_candidates[indices]
    selected_unc = uncertainties[indices].squeeze().numpy() if torch.is_tensor(uncertainties) else uncertainties[indices].squeeze()

    df = pd.DataFrame(selected_X, columns=param_names)
    df["uncertainty"] = selected_unc

    csv_path = iter_dir / "selected_points.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def load_true_eval_dataset(eval_data_path, target="DMRD", logger=None):
    """
    Load a dedicated evaluation dataset (separate from training data).

    Supports ROOT files (via pmssm.load_pmssm_data logic) and CSV files.

    Args:
        eval_data_path: Path to ROOT file or CSV containing true evaluation data
        target: Target function name (determines branch name for ROOT files)
        logger: Logger instance

    Returns:
        X_eval: (N, 19) tensor in physical units
        Y_eval: (N,) tensor of target values (untransformed)
    """
    from .config import TARGET_CONFIG
    from .data import load_pmssm_data

    eval_path = Path(eval_data_path)

    if eval_path.suffix == ".csv":
        df = pd.read_csv(eval_path)
        branch = TARGET_CONFIG[target]["branch"]
        param_cols = [p.replace("IN_", "") for p in PARAM_ORDER]
        X_eval = torch.tensor(df[param_cols].values, dtype=torch.float32)
        Y_eval = torch.tensor(df[branch].values, dtype=torch.float32)
    else:
        # ROOT file — reuse pmssm loading
        X_eval, Y_eval = load_pmssm_data(
            data_dir=str(eval_path.parent),
            n_datasets=-1,
            logger=logger,
            target=target
        )

    if logger:
        logger.info(f"Loaded true eval dataset: {len(X_eval)} points from {eval_path}")

    return X_eval, Y_eval.view(-1)
