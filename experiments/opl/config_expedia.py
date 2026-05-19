"""
OPL experiment configuration for Expedia real-data experiments.
"""

# Experiment design
SETTING = "linear"
POLICY_TYPE = "linear"
DATA_TAG = "expedia"

# Train vs evaluation robustness
DELTA_TRAIN = 0.1
TEST_DELTAS = [0.2]

# Path and basic data settings
EXPEDIA_TRAIN_CSV = "RealData/data/train.csv"
EXPEDIA_MAX_ROWS = 200000
EXPEDIA_REWARD_MODE = "booking_bool"  # or "price_times_booking", "gross_bookings_usd"

# Train/test split: hold out one hotel type for test, rest for train
EXPEDIA_SPLIT_MODE = "hotel_type"
EXPEDIA_SPLIT_CONFIG = {}
EXPEDIA_SPLIT_SEED = 123

# Price quantile range used to set policy output bounds (from training prices)
EXPEDIA_PRICE_LOWER_Q = 1.0
EXPEDIA_PRICE_UPPER_Q = 99.0

# Whether to include delta=0 in evaluation deltas
EXPEDIA_INCLUDE_DELTA0_BASELINE = False

# Sample sizes
TRAIN_SIZES = [500]
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
EPOCHS_STEP = 10
WARM_START_LR = 1e-3
WARM_START_EPOCHS = 80
MIN_BANDWIDTH = 1e-3
ALPHA_UPDATE_FREQ = 3
GRAD_CLIP_MAX_NORM = 1.0
FOREST_N_ESTIMATORS = 20
FOREST_MAX_DEPTH = 10

# Evaluation (worst-case attacks for Hat Q_min)
M_ATTACKS = 10
