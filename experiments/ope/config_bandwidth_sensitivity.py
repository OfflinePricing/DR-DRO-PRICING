"""
OPE kernel bandwidth h sensitivity experiment.

Compares:
  - silverman: h = 1.06 * std(P) * n^{-1/5}
  - h = 1.06 * std(P) * n^{-1/4.9}
  - h = 1.06 * std(P) * n^{-1/2.9}
"""

from experiments.ope import config_ope as _base

SETTING = _base.SETTING
DATA_TAG = "synthetic_bandwidth"
DELTA_TRAIN = _base.DELTA_TRAIN
TEST_DELTAS = _base.TEST_DELTAS
N_TEST_TRUE = _base.N_TEST_TRUE
TRAIN_SIZES = _base.TRAIN_SIZES
SEED = _base.SEED
TARGET_POLICY_SEED = _base.TARGET_POLICY_SEED
N_REPEATS = _base.N_REPEATS
MIN_BANDWIDTH = _base.MIN_BANDWIDTH
OPE_N_FOLDS = _base.OPE_N_FOLDS
OPE_N_ESTIMATORS = _base.OPE_N_ESTIMATORS

BANDWIDTH_MODES = ["silverman", 4.9, 2.9]
