"""
Data loading, filtering, and normalization utilities for pMSSM pipeline.

This module provides functions for:
- Loading ROOT physics data files
- Train/validation splitting
- Z-score normalization (for transformer models)
- Min-max normalization (for GP models)
- Target transformations (log-space for relic density)
"""

import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import TARGET_CONFIG, PARAM_ORDER, PARAM_TO_GP_RANGE_KEY, GP_RANGE_DICT


# ===== Utility Functions =====

def running_in_notebook():
    """Check if code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


# ===== Data Loading =====

LSP_FRAC_BRANCHES = ("SP_LSP_Bino_frac", "SP_LSP_Wino_frac", "SP_LSP_Higgsino_frac")


def _load_lsp_fracs(trees, mask, logger=None):
    """Read SP_LSP_{Bino,Wino,Higgsino}_frac for `trees` and apply `mask`.

    When a branch is missing or the LSP is not a neutralino, the ntupler leaves
    the entry at its initial value; we coerce to NaN so downstream coloring
    can skip those rows cleanly.
    """
    cols = []
    for b in LSP_FRAC_BRANCHES:
        try:
            arr = np.concatenate([t[b].array(library="np") for t in trees]).astype(np.float32)
        except Exception as e:
            if logger:
                logger.warning(f"Missing LSP fraction branch '{b}' ({e}); filling with NaN")
            arr = np.full(mask.shape, np.nan, dtype=np.float32)
        cols.append(arr)
    fracs = np.stack(cols, axis=1)[mask]
    bad = (~np.isfinite(fracs).all(axis=1)) | (fracs < 0).any(axis=1)
    if bad.any():
        fracs[bad] = np.nan
    return torch.from_numpy(fracs).float()


def load_pmssm_data(n_datasets=-1, logger=None, plot_dir="plots", target="DMRD",
                    data_dir="data/18387358", return_lsp_fracs=False):
    """
    Load pMSSM ROOT data with combined filter.

    Applies ``(Y > 0) & (Y < 1.0) & (SP_m_h != -1)``:
    - ``Y > 0``: valid target value (positive, non-sentinel)
    - ``Y < 1.0``: sub-dominant dark matter candidates
    - ``SP_m_h != -1``: valid Higgs mass computation (SPheno did not fail)

    Args:
        n_datasets: Number of ROOT files to load (-1 for all)
        logger: Logger instance for output
        plot_dir: Directory to save histogram plot
        target: Target variable name (default: "DMRD")
        data_dir: Directory containing ROOT files (default: "data/18387358")
        return_lsp_fracs: If True, also return (N, 3) tensor of neutralino
            [bino, wino, higgsino] fractions from the mixing matrix.

    Returns:
        X: Input tensor (N, 19) in physical units
        Y: Target tensor (N, 1) in physical units
        lsp_fracs: (N, 3) tensor (only if return_lsp_fracs=True)
    """
    import uproot

    files = sorted(glob.glob(f"{data_dir}/*.root"))
    if logger:
        logger.info(f"Found {len(files)} ROOT files")

    if n_datasets != -1:
        if logger:
            logger.info(f"Only using {n_datasets} out of the {len(files)} datasets")
        files = files[:n_datasets]

    trees = [uproot.open(f)["susy"] for f in files]

    # Load input parameters
    branches = PARAM_ORDER

    X_raw = np.column_stack([
        np.concatenate([t[b].array(library="np") for t in trees])
        for b in branches
    ])

    # Load target variable
    target_branch = TARGET_CONFIG[target]["branch"]
    Y_raw = np.concatenate([t[target_branch].array(library="np") for t in trees])
    sp_mh = np.concatenate([t["SP_m_h"].array(library="np") for t in trees])

    # Apply combined filter
    mask = (Y_raw > 0) & (Y_raw < 1.0) & (sp_mh != -1)
    if logger:
        logger.info(
            f"Filter ({target_branch} > 0 & < 1 & SP_m_h != -1): "
            f"{mask.sum()} / {len(Y_raw)} samples kept"
        )

    # Plot target distribution
    plt.hist(Y_raw[mask], bins=20, range=[0.0, 1.0])
    if not running_in_notebook():
        plt.savefig(f"{plot_dir}/hist_dataset.png")
        plt.close()
    else:
        plt.show()

    X = torch.from_numpy(X_raw[mask]).float()
    Y = torch.from_numpy(Y_raw[mask]).float().unsqueeze(1)

    if return_lsp_fracs:
        return X, Y, _load_lsp_fracs(trees, mask, logger=logger)
    return X, Y


def load_mcmc_data(data_dir="data/19250082", target="DMRD", logger=None,
                   return_lsp_fracs=False):
    """
    Load MCMC ROOT data with the same filters as load_pmssm_data.

    Args:
        data_dir: Directory containing MCMC ROOT files
        target: Target variable name (default: "DMRD")
        logger: Logger instance for output
        return_lsp_fracs: If True, also return (N, 3) tensor of neutralino
            [bino, wino, higgsino] fractions from the mixing matrix.

    Returns:
        X: Input tensor (N, 19) in physical units
        Y: Target tensor (N, 1) in physical units
        lsp_fracs: (N, 3) tensor (only if return_lsp_fracs=True)
    """
    import uproot

    files = sorted(glob.glob(f"{data_dir}/*.root"))
    if logger:
        logger.info(f"Found {len(files)} MCMC ROOT files in {data_dir}")

    target_branch = TARGET_CONFIG[target]["branch"]
    true_value = TARGET_CONFIG[target]["true_value"]

    # Keep only files whose Omega values straddle the true value
    # and don't contain any values below omega_min
    omega_min = 0.04
    trees = []
    n_no_straddle = 0
    n_below_min = 0
    for f in files:
        t = uproot.open(f)["susy"]
        y = t[target_branch].array(library="np")
        if not (np.any(y < true_value) and np.any(y > true_value)):
            n_no_straddle += 1
            continue
        if np.any(y < omega_min):
            n_below_min += 1
            continue
        trees.append(t)

    if n_no_straddle > 0 or n_below_min > 0:
        msg = (f"Excluded {n_no_straddle + n_below_min}/{len(files)} MCMC ROOT files: "
               f"{n_no_straddle} don't straddle {target_branch}={true_value}, "
               f"{n_below_min} contain values below {omega_min}")
        if logger:
            logger.warning(msg)

    X_raw = np.column_stack([
        np.concatenate([t[b].array(library="np") for t in trees])
        for b in PARAM_ORDER
    ])

    Y_raw = np.concatenate([t[target_branch].array(library="np") for t in trees])
    sp_mh = np.concatenate([t["SP_m_h"].array(library="np") for t in trees])

    mask = (Y_raw > 0) & (Y_raw < 1.0) & (sp_mh != -1)
    if logger:
        logger.info(
            f"MCMC filter ({target_branch} > 0 & < 1 & SP_m_h != -1): "
            f"{mask.sum()} / {len(Y_raw)} samples kept"
        )

    X = torch.from_numpy(X_raw[mask]).float()
    Y = torch.from_numpy(Y_raw[mask]).float().unsqueeze(1)

    if return_lsp_fracs:
        return X, Y, _load_lsp_fracs(trees, mask, logger=logger)
    return X, Y


# ===== Train/Val Splitting =====

def make_split(X, train_split=0.9, seed=42, logger=None):
    """
    Create reproducible train/validation split.

    Args:
        X: Input tensor to split
        train_split: Fraction of data for training (default: 0.9)
        seed: Random seed for reproducibility
        logger: Logger instance

    Returns:
        idx_train: Training indices
        idx_val: Validation indices
    """
    N = len(X)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    n_train = int(train_split * N)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:]

    if logger:
        logger.info(f"Split: n_train={len(idx_train)}, n_val={len(idx_val)}")

    return idx_train, idx_val


# ===== Z-score Normalization (Transformer Models) =====

def compute_stats(X, Y, idx_train):
    """
    Compute z-score normalization statistics from training data.

    Args:
        X: Input tensor (N, D)
        Y: Target tensor (N, 1)
        idx_train: Training indices

    Returns:
        mean_X: Mean of training inputs (D,)
        std_X: Std of training inputs (D,)
        mean_Y: Mean of training targets (1,)
        std_Y: Std of training targets (1,)
    """
    mean_X = X[idx_train].mean(dim=0)
    std_X = X[idx_train].std(dim=0) + 1e-8
    mean_Y = Y[idx_train].mean(dim=0)
    std_Y = Y[idx_train].std(dim=0) + 1e-8

    return mean_X, std_X, mean_Y, std_Y


# ===== Min-Max Normalization (GP Models) =====

def build_norm_tensors(param_order=PARAM_ORDER, gp_range_dict=GP_RANGE_DICT):
    """
    Build min/max tensors for min-max normalization in PARAM_ORDER.

    Args:
        param_order: List of parameter names
        gp_range_dict: Dict mapping GP parameter names to [min, max]

    Returns:
        data_min: Minimum values tensor (D,)
        data_max: Maximum values tensor (D,)
    """
    mins, maxs = [], []
    for param in param_order:
        key = PARAM_TO_GP_RANGE_KEY[param]
        lo, hi = gp_range_dict[key]
        mins.append(lo)
        maxs.append(hi)

    return torch.tensor(mins, dtype=torch.float32), torch.tensor(maxs, dtype=torch.float32)


def normalize_x(X, data_min, data_max):
    """
    Min-max normalize inputs to [0, 1].

    Args:
        X: Input tensor (N, D)
        data_min: Minimum values (D,)
        data_max: Maximum values (D,)

    Returns:
        X_norm: Normalized tensor (N, D)
    """
    return (X - data_min) / (data_max - data_min)


def unnormalize_x(X_norm, data_min, data_max):
    """
    Reverse min-max normalization.

    Args:
        X_norm: Normalized tensor (N, D)
        data_min: Minimum values (D,)
        data_max: Maximum values (D,)

    Returns:
        X: Unnormalized tensor (N, D)
    """
    return X_norm * (data_max - data_min) + data_min


# ===== Target Transformations =====

def transform_y(Y, target="DMRD"):
    """
    Transform target values to training space.

    For DMRD/CrossSection: log(Y / true_value)
    For CLs: identity (no transformation)

    This puts the target value at 0 in transformed space,
    making it easier to set decision thresholds.

    Args:
        Y: Target tensor in physical units
        target: Target name (DMRD, CrossSection, CLs)

    Returns:
        Y_transformed: Transformed target tensor
    """
    if target == "CLs":
        return Y.clone()

    true_value = TARGET_CONFIG[target]["true_value"]
    return torch.log(Y / true_value)


def inverse_transform_y(Y_t, target="DMRD"):
    """
    Inverse transform: convert from training space back to physical units.

    Args:
        Y_t: Transformed target tensor
        target: Target name (DMRD, CrossSection, CLs)

    Returns:
        Y: Target tensor in physical units
    """
    if target == "CLs":
        return Y_t.clone()

    true_value = TARGET_CONFIG[target]["true_value"]
    return true_value * torch.exp(Y_t)


def split_mcmc_for_oracle(X, Y, F, eval_fraction=0.1, seed=42):
    """Deterministically split the MCMC dataset into a candidate pool and an
    eval slice for the theoretical-limit / oracle AL mode.

    The pool slice is what the AL surrogate sees as candidates each iteration;
    the eval slice replaces ``X_mcmc, Y_mcmc, F_mcmc`` everywhere they're used
    today (cross-eval, accuracy capture, representative-points trajectory) so
    training data and eval data stay disjoint.

    Returns:
        (X_pool, Y_pool, F_pool, X_eval, Y_eval, F_eval, pool_idx, eval_idx)
        where ``pool_idx`` and ``eval_idx`` are LongTensors of indices into
        the original X/Y/F (saved to state.pt for resume).
    """
    n = len(X)
    if n == 0:
        raise ValueError("split_mcmc_for_oracle: empty MCMC dataset")
    n_eval = max(1, int(round(n * eval_fraction)))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
    eval_idx = perm[:n_eval]
    pool_idx = perm[n_eval:]
    return (
        X[pool_idx], Y[pool_idx], F[pool_idx],
        X[eval_idx], Y[eval_idx], F[eval_idx],
        pool_idx, eval_idx,
    )
