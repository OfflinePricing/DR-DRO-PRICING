"""
OPL experiment configuration for synthetic (simulated) data.
"""

# Experiment design
SETTING = "linear"
POLICY_TYPE = "linear"
DATA_TAG = "synthetic"

# Train vs evaluation robustness
DELTA_TRAIN = 0.2
TEST_DELTAS = [0.1, 0.2, 0.3]

# Policy price bounds (synced to src.config at runtime for policy networks)
PRICE_MIN = 0.5
PRICE_MAX = 3.0

# Sample sizes
TRAIN_SIZES = [500, 1000, 1500, 2000, 2500]
N_TEST = 2500
N_TEST_ATTACK = 1000

# Reproducibility
SEED = 42
BASE_SEED = 60000

# Experiment replication
N_REPEATS = 20

# Training
NUISANCE_EPOCHS = 50
N_FOLDS = 3
BATCH_SIZE = 512
EPOCHS = 50
EPOCHS_STEP = 10  # extra epochs per 500 training samples above 500
WARM_START_LR = 1e-3
WARM_START_EPOCHS = 80
MIN_BANDWIDTH = 1e-3
ALPHA_UPDATE_FREQ = 3
GRAD_CLIP_MAX_NORM = 1.0
FOREST_N_ESTIMATORS = 20
FOREST_MAX_DEPTH = 10

# Evaluation (worst-case attacks for Hat Q_min)
M_ATTACKS = 10

# Synthetic DGP noise (logistic purchase model)
TRAIN_Z_LOC = 0.0
TRAIN_Z_SCALE = 0.2
