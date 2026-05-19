"""
OPL kernel bandwidth h sensitivity experiment (synthetic data, HatR_DRO only).

Compares:
  - silverman: h = 1.06 * std(P) * n^{-1/5}
  - h = 1.06 * std(P) * n^{-1/4.9}
  - h = 1.06 * std(P) * n^{-1/2.9}
"""

from experiments.opl import config_synthetic as _base

# Re-use synthetic OPL setup
SETTING = _base.SETTING
POLICY_TYPE = _base.POLICY_TYPE
DATA_TAG = "synthetic_bandwidth"
DELTA_TRAIN = _base.DELTA_TRAIN
TEST_DELTAS = _base.TEST_DELTAS
PRICE_MIN = _base.PRICE_MIN
PRICE_MAX = _base.PRICE_MAX
TRAIN_SIZES = _base.TRAIN_SIZES
N_TEST = _base.N_TEST
SEED = _base.SEED
BASE_SEED = _base.BASE_SEED
N_REPEATS = _base.N_REPEATS
NUISANCE_EPOCHS = _base.NUISANCE_EPOCHS
N_FOLDS = _base.N_FOLDS
BATCH_SIZE = _base.BATCH_SIZE
EPOCHS = _base.EPOCHS
EPOCHS_STEP = _base.EPOCHS_STEP
WARM_START_LR = _base.WARM_START_LR
WARM_START_EPOCHS = _base.WARM_START_EPOCHS
MIN_BANDWIDTH = _base.MIN_BANDWIDTH
ALPHA_UPDATE_FREQ = _base.ALPHA_UPDATE_FREQ
GRAD_CLIP_MAX_NORM = _base.GRAD_CLIP_MAX_NORM
FOREST_N_ESTIMATORS = _base.FOREST_N_ESTIMATORS
FOREST_MAX_DEPTH = _base.FOREST_MAX_DEPTH
TRAIN_Z_LOC = _base.TRAIN_Z_LOC
TRAIN_Z_SCALE = _base.TRAIN_Z_SCALE

# Bandwidth settings to compare
BANDWIDTH_MODES = ["silverman", 4.9, 2.9]
