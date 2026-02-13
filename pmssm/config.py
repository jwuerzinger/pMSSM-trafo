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

# Target-specific configuration (true values, thresholds, ROOT branch names)
TARGET_CONFIG = {
    "DMRD": {
        "true_value": 0.12,          # Observed dark matter relic density
        "threshold": 0.0,            # Decision threshold in transformed space
        "branch": "MO_Omega"         # ROOT file branch name
    },
    "CrossSection": {
        "true_value": 0.03,
        "threshold": 0.0,
        "branch": "xsec_TOTAL"
    },
    "CLs": {
        "true_value": 0.05,
        "threshold": 0.05,
        "branch": "Final__CLs"
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
