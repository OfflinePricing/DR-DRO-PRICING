"""
OPE experiment configuration (synthetic DGP).
"""

# Experiment design
SETTING = "linear"
DATA_TAG = "synthetic"

# Robustness radius: fit nuisances / cross-fitting at δ_train; report R_δ at each δ_test
DELTA_TRAIN = 0.2
TEST_DELTAS = [0.1, 0.2, 0.3]

# True R_δ ground truth (sampling with LDR²PE-CP on held-out data)
N_TEST_TRUE = 2500

# Training sample sizes T
TRAIN_SIZES = [500, 1000, 2000, 3000, 4000, 5000]

# Reproducibility
SEED = 42
TARGET_POLICY_SEED = 12345

# Experiment replication
N_REPEATS = 20

# Algorithm
MIN_BANDWIDTH = 1e-3
OPE_N_FOLDS = 5
OPE_N_ESTIMATORS = 20

# Default bandwidth (Silverman); override in sensitivity runs
BANDWIDTH_MODE = "silverman"
