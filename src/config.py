"""
Shared defaults for src/ (DGP, learners, estimators, visualization).

Experiment design (δ_train / δ_test, sample sizes, repeats, bandwidth mode)
lives in experiments/ope/config_*.py and experiments/opl/config_*.py.
OPE/OPL entry points sync runtime fields into this module before running.
"""

import numpy as np

# --- Synthetic DGP (PricingEnvironment) ---
DIM = 3
PRICE_MIN = 0.5
PRICE_MAX = 3.0
TRAIN_X_MEAN = np.array([0.0, 0.0, 0.0])
TRAIN_X_STD = np.array([1.0, 1.0, 1.0])
TRAIN_Z_LOC = 0.0
TRAIN_Z_SCALE = 0.2

# --- Reproducibility (overwritten by experiments/ope|opl before runs) ---
SEED = 42

# --- Learner / optimizer defaults ---
DELTA = 0.2  # default δ for robust learners (CDRO, Leung2025); not used by Ai2024
LR_POLICY = 3e-3
EPOCHS = 50
ETA = 0.05
MIN_BANDWIDTH = 1e-3

ALPHA_OPT_BOUNDS = (0.01, 10.0)
ALPHA_OPT_METHOD = "bounded"
ALPHA_OPT_TOL = 1e-3
ALPHA_OPTIMIZER_LR = 5e-4
ALPHA_UPDATE_FREQ = 3
GRAD_CLIP_MAX_NORM = 1.0

# --- OPE algorithm (overwritten from experiments/ope/config_ope.py) ---
OPE_N_FOLDS = 5
OPE_N_ESTIMATORS = 20
