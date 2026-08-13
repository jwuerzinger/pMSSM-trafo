"""
Configuration constants for pMSSM active learning pipeline.

This module centralizes all configuration constants, parameter ranges,
target configurations, and default hyperparameters used across the codebase.
"""

# ===== Parameter Definitions =====

# Order of parameters in input tensors (19 parameters total)
PARAM_ORDER = [
    "IN_meL", "IN_meR", "IN_mtauL", "IN_mtauR",
    "IN_mqL1", "IN_muR", "IN_mdR", "IN_mqL3",
    "IN_mtR", "IN_mbR", "IN_M_1", "IN_M_2",
    "IN_mu", "IN_M_3", "IN_At", "IN_Ab",
    "IN_Atau", "IN_mA", "IN_tanb"
]

# pMSSM parameter ranges (from physics constraints)
# Fixed parameters have (value, value) ranges
PARAM_RANGES = {
    "IN_meL": (0, 2000),
    "IN_meR": (0, 2000),
    "IN_mtauL": (2000, 2000),  # Fixed
    "IN_mtauR": (2000, 2000),  # Fixed
    "IN_mqL1": (4000, 4000),   # Fixed
    "IN_muR": (4000, 4000),    # Fixed
    "IN_mdR": (4000, 4000),    # Fixed
    "IN_mqL3": (4000, 4000),   # Fixed
    "IN_mtR": (4000, 4000),    # Fixed
    "IN_mbR": (4000, 4000),    # Fixed
    "IN_M_1": (-2000, 2000),
    "IN_M_2": (-2000, 2000),
    "IN_mu": (-2000, 2000),
    "IN_M_3": (4000, 4000),    # Fixed
    "IN_At": (-8000, 8000),
    "IN_Ab": (-2000, 2000),
    "IN_Atau": (-2000, 2000),
    "IN_mA": (2000, 2000),     # Fixed
    "IN_tanb": (1, 60),
}

# ===== Target Configurations =====

# Canonical Run3ModelGen step definitions, keyed by step name. `gen_steps` below
# names a subset of these; pmssm.model_generation materialises them in the listed
# order. Kept here (rather than inline in model_generation) so the AL loop's
# online scan config and the target registry stay in one place.
#
# NB the SModelS step takes no `log_dir` (run_SModelS reads only input_dir and
# output_dir), and it must come LAST among the SPheno consumers: in the legacy
# in-process path it appends the NLL cross sections to the SPheno output in
# place, so any consumer ordered after it would read a mutated spectrum.
MODELGEN_STEP_DEFS = {
    "prep_input": {"name": "prep_input", "output_dir": "input", "prefix": "IN"},
    "SPheno": {"name": "SPheno", "input_dir": "input", "output_dir": "SPheno",
               "log_dir": "SPheno_log", "prefix": "SP"},
    "micromegas": {"name": "micromegas", "input_dir": "SPheno",
                   "output_dir": "micromegas", "prefix": "MO"},
    "SModelS": {"name": "SModelS", "input_dir": "SPheno",
                "output_dir": "SModelS", "prefix": "SModelS"},
}

# Target-specific configuration. Every key here must return the historical
# hardcoded value for "DMRD", so adding a target cannot change relic-density
# results. Keys:
#   true_value   physical value the transform divides by (transform_y)
#   threshold    decision threshold in transformed space
#   branch       ROOT branch holding the target
#   label        matplotlib label for the physical target
#   valid_max    exclusive upper ingest cut, or None for no upper cut. The
#                relic density uses 1.0 ("does not overclose the universe");
#                for an exclusion r-value the >1 half IS the region of
#                interest, so an upper cut there would delete it.
#   hist_range   range= for the ingest histogram, or None to autoscale
#   gen_require_neutralino_lsp
#                whether load_generated_data vetoes non-neutralino LSPs.
#                This governs NEWLY GENERATED data only. Pool ingest
#                (load_pmssm_data) keeps its own explicit argument, default
#                False, which no driver passes — do not couple the two, or the
#                relic-density pool would suddenly shrink.
#   gen_steps    Run3ModelGen steps the AL loop runs to label a candidate
#   has_mcmc_reference
#                whether an emcee posterior reference dataset exists
TARGET_CONFIG = {
    "DMRD": {
        "true_value": 0.12,          # Observed dark matter relic density
        "threshold": 0.0,            # Decision threshold in transformed space
        "branch": "MO_Omega",        # ROOT file branch name
        "label": r"$\Omega h^2$",
        "valid_max": 1.0,
        "hist_range": (0.0, 1.0),
        "gen_require_neutralino_lsp": True,
        "gen_steps": ("prep_input", "SPheno", "micromegas"),
        "has_mcmc_reference": True,
    },
    # SModelS best expected r-value: the LHC exclusion boundary. r > 1 means the
    # model is expected to be excluded, so true_value = 1.0 puts the boundary at
    # log(r/1) = 0, exactly where the relic density's 0.12 sits for DMRD.
    # Run3ModelGen fills the branch with the sentinel -1. when no analysis
    # applies, so the (Y > 0) ingest cut is precisely "SModelS returned a
    # verdict". micromegas stays in gen_steps although the target does not need
    # it: it costs ~3% of the SModelS step and keeps MO_Omega on every generated
    # point, so the two target branches remain cross-comparable.
    "ExpR": {
        "true_value": 1.0,
        "threshold": 0.0,
        "branch": "SModelS_bestExpR_r_expected",
        "label": r"$r_{\mathrm{exp}}$",
        "valid_max": None,
        "hist_range": None,
        "gen_require_neutralino_lsp": False,
        "gen_steps": ("prep_input", "SPheno", "micromegas", "SModelS"),
        "has_mcmc_reference": False,
    },
    "CrossSection": {
        "true_value": 0.03,
        "threshold": 0.0,
        "branch": "xsec_TOTAL",
        "label": r"$\sigma_{\mathrm{tot}}$",
        "valid_max": 1.0,
        "hist_range": (0.0, 1.0),
        "gen_require_neutralino_lsp": True,
        "gen_steps": ("prep_input", "SPheno", "micromegas"),
        "has_mcmc_reference": False,
    },
    "CLs": {
        "true_value": 0.05,
        "threshold": 0.05,
        "branch": "Final__CLs",
        "label": r"$\mathrm{CL}_s$",
        "valid_max": 1.0,
        "hist_range": (0.0, 1.0),
        "gen_require_neutralino_lsp": True,
        "gen_steps": ("prep_input", "SPheno", "micromegas"),
        "has_mcmc_reference": False,
    },
}

# Backward-compatible alias for DMRD true value
DMRD_TRUE_VALUE = TARGET_CONFIG["DMRD"]["true_value"]

# ===== GP Normalization Ranges =====

# Min-max normalization ranges from the GP pipeline
# Keys use the GP repo's naming convention (no IN_ prefix, AT instead of At)
GP_RANGE_DICT = {
    "M_1": [-2000, 2000],
    "M_2": [-2000, 2000],
    "tanb": [1, 60],
    "mu": [-2000, 2000],
    "M_3": [1000, 5000],
    "AT": [-8000, 8000],
    "Ab": [-2000, 2000],
    "Atau": [-2000, 2000],
    "mA": [0, 5000],
    "mqL3": [2000, 5000],
    "mtR": [2000, 5000],
    "mbR": [2000, 5000],
    "meL": [0, 10000],
    "mtauL": [0, 10000],
    "meR": [0, 10000],
    "mtauR": [0, 10000],
    "mqL1": [0, 10000],
    "muR": [0, 10000],
    "mdR": [0, 10000],
}

# ===== Mappings =====

# Mapping from CSV column names to Run3ModelGen parameter names
CSV_TO_MODELGEN = {
    "meL": "meL",
    "meR": "meR",
    "mtauL": "mtauL",
    "mtauR": "mtauR",
    "mqL1": "mqL1",
    "muR": "muR",
    "mdR": "mdR",
    "mqL3": "mqL3",
    "mtR": "mtR",
    "mbR": "mbR",
    "M_1": "M_1",
    "M_2": "M_2",
    "mu": "mu",
    "M_3": "M_3",
    "At": "AT",  # Note: Run3ModelGen uses uppercase AT
    "Ab": "Ab",
    "Atau": "Atau",
    "mA": "mA",
    "tanb": "tanb",
}

# Map from PARAM_ORDER names (IN_xxx) to GP_RANGE_DICT keys
PARAM_TO_GP_RANGE_KEY = {
    "IN_meL": "meL",
    "IN_meR": "meR",
    "IN_mtauL": "mtauL",
    "IN_mtauR": "mtauR",
    "IN_mqL1": "mqL1",
    "IN_muR": "muR",
    "IN_mdR": "mdR",
    "IN_mqL3": "mqL3",
    "IN_mtR": "mtR",
    "IN_mbR": "mbR",
    "IN_M_1": "M_1",
    "IN_M_2": "M_2",
    "IN_mu": "mu",
    "IN_M_3": "M_3",
    "IN_At": "AT",
    "IN_Ab": "Ab",
    "IN_Atau": "Atau",
    "IN_mA": "mA",
    "IN_tanb": "tanb",
}

# ===== Default Hyperparameters =====

class DefaultHyperparameters:
    """Default hyperparameters for models, training, and selection."""

    # Model defaults
    TRANSFORMER_D_MODEL = 128
    TRANSFORMER_NHEAD = 4
    TRANSFORMER_NUM_LAYERS = 3
    TRANSFORMER_DIM_FEEDFORWARD = 512
    TRANSFORMER_DROPOUT = 0.1

    # Training defaults
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 2000
    BATCH_SIZE = 256
    PATIENCE = 200
    GRAD_CLIP = 1.0

    # Active learning defaults
    MC_DROPOUT_SAMPLES = 30
    ENTROPY_BLUR = 0.15
    ENTROPY_BETA = 50.0
    ENTROPY_POOL_SIZE = 5000
    PROXIMITY_SAMPLING = 0.1
    TOLERANCE_SAMPLING = 1.0

    # Data loading defaults
    DATA_DIR = "data/18387358/*.root"
    TRAIN_SPLIT = 0.9
    RANDOM_SEED = 42
