"""
pMSSM Active Learning Package

This package provides modular, reusable components for active learning
on phenomenological Minimal Supersymmetric Standard Model (pMSSM) data.

Main components:
- config: Constants, parameter ranges, and configuration
- data: Data loading, filtering, and normalization
- datasets: PyTorch Dataset classes
- models: Neural network architectures (transformer, feedforward, GP)
- selection: Candidate generation and selection strategies
- uncertainty: Uncertainty estimation (MC Dropout, GP posterior)
- training: Training loops for transformer and GP models
- evaluation: Model evaluation metrics and diagnostics
- visualization: Plotting utilities
- logging_utils: Logging configuration
- model_generation: Run3ModelGen interface

Example usage:
    from pmssm import config, data, selection, training
    from pmssm.models import PMSSMTransformerTabular

    # Load data
    X, Y = data.load_pmssm_data(n_datasets=5, logger=logger)

    # Generate candidates with LHS
    candidates = selection.generate_candidate_pool(
        n_candidates=1000,
        method='lhs',
        seed=42
    )
"""

__version__ = "0.1.0"

# Re-export key components for easy access
from . import config
from . import data
from . import datasets
from . import models
from . import selection
from . import uncertainty
from . import training
from . import evaluation
from . import visualization
from . import logging_utils
from . import resume
from . import accuracy
from . import model_generation

# Backward compatibility: expose commonly used functions at package level
from .config import (
    PARAM_ORDER,
    PARAM_RANGES,
    CSV_TO_MODELGEN,
    TARGET_CONFIG,
    GP_RANGE_DICT,
)

from .data import (
    load_pmssm_data,
    load_mcmc_data,
    make_split,
    compute_stats,
    build_norm_tensors,
    normalize_x,
    unnormalize_x,
    transform_y,
    inverse_transform_y,
    split_mcmc_for_oracle,
)

from .datasets import (
    Dummy_PMSSMDataset,
    PMSSMDataset,
)

from .models import (
    PMSSMTransformer,
    PMSSMTransformerTabular,
    PMSSMFeedForward,
    is_transformer,
    get_model_name,
)

from .selection import (
    generate_candidate_pool,
    select_top_uncertain,
    select_top_uncertain_filtered,
    select_top_uncertain_tol_only,
    select_entropy_batch_mc,
    select_points,
)

from .uncertainty import (
    compute_uncertainty_mc_dropout,
    compute_uncertainty_ensemble,
    compute_uncertainty_gp,
)

from .training import (
    train_with_validation,
    train_model_worker,
    create_gp_model,
    train_gp_model,
    train_gp_worker,
    model_has_likelihood,
    model_is_variational,
)

from .evaluation import (
    compare_random_predictions,
    compute_gp_r2,
    compute_comprehensive_metrics,
    extract_lengthscales,
)

from .visualization import (
    plot_losses,
    scatter_true_vs_pred,
    hist_true_vs_pred,
    plot_data_histograms,
    plot_parallel_coordinates,
    plot_candidate_uncertainty,
    plot_iteration_metrics,
    plot_advanced_diagnostics,
    plot_eval_scatterplots,
    classify_lsp_type,
    pick_representative_points,
    plot_representative_trajectories,
    gp_predict,
    running_in_notebook,
)

from .logging_utils import (
    setup_logging,
    setup_worker_logging,
)

from .model_generation import (
    generate_models_from_csv,
    load_generated_data,
    save_selected_points,
    load_true_eval_dataset,
)

from .accuracy import (
    binary_accuracy,
    update_accuracy_trajectory,
    write_iter_accuracies,
)

__all__ = [
    # Modules
    'config',
    'data',
    'datasets',
    'models',
    'selection',
    'uncertainty',
    'training',
    'evaluation',
    'visualization',
    'logging_utils',
    'model_generation',

    # Config constants
    'PARAM_ORDER',
    'PARAM_RANGES',
    'CSV_TO_MODELGEN',
    'TARGET_CONFIG',
    'GP_RANGE_DICT',

    # Data functions
    'load_pmssm_data',
    'load_mcmc_data',
    'make_split',
    'compute_stats',
    'build_norm_tensors',
    'normalize_x',
    'unnormalize_x',
    'transform_y',
    'inverse_transform_y',
    'split_mcmc_for_oracle',

    # Dataset classes
    'Dummy_PMSSMDataset',
    'PMSSMDataset',

    # Model classes and utilities
    'PMSSMTransformer',
    'PMSSMTransformerTabular',
    'PMSSMFeedForward',
    'is_transformer',
    'get_model_name',

    # Selection functions
    'generate_candidate_pool',
    'select_top_uncertain',
    'select_top_uncertain_filtered',
    'select_top_uncertain_tol_only',
    'select_entropy_batch_mc',
    'select_points',

    # Uncertainty estimation
    'compute_uncertainty_mc_dropout',
    'compute_uncertainty_ensemble',
    'compute_uncertainty_gp',

    # Training
    'train_with_validation',
    'train_model_worker',
    'create_gp_model',
    'train_gp_model',
    'train_gp_worker',
    'model_has_likelihood',
    'model_is_variational',

    # Evaluation
    'compare_random_predictions',
    'compute_gp_r2',
    'compute_comprehensive_metrics',
    'extract_lengthscales',

    # Visualization
    'plot_losses',
    'scatter_true_vs_pred',
    'hist_true_vs_pred',
    'plot_data_histograms',
    'plot_parallel_coordinates',
    'plot_candidate_uncertainty',
    'plot_iteration_metrics',
    'plot_advanced_diagnostics',
    'plot_eval_scatterplots',
    'classify_lsp_type',
    'pick_representative_points',
    'plot_representative_trajectories',
    'gp_predict',
    'running_in_notebook',

    # Logging
    'setup_logging',
    'setup_worker_logging',

    # Model generation
    'generate_models_from_csv',
    'load_generated_data',
    'save_selected_points',
    'load_true_eval_dataset',

    # Classification accuracy capture
    'binary_accuracy',
    'update_accuracy_trajectory',
    'write_iter_accuracies',
]
