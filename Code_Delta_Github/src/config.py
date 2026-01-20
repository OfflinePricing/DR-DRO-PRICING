
import numpy as np

# Global Settings
DIM = 3                 
PRICE_MIN = 0.5
PRICE_MAX = 3.0
# N_TRAIN will be dynamically changed in experiments

# Algorithm hyperparameters
SEED = 42
LR_POLICY = 3e-3        
EPOCHS = 50           
ETA = 0.05              
# --- Base Environment ---
# Use Logistic noise as baseline
TRAIN_X_MEAN = np.array([0.0, 0.0, 0.0]) 
TRAIN_X_STD  = np.array([1.0, 1.0, 1.0])
TRAIN_Z_LOC = 0.0       
TRAIN_Z_SCALE = 0.2     

# Default Delta (for training), will be overridden during testing
DELTA = 0.2

# ==========================================
# Full run mode configuration parameters
# ==========================================
# Number of experiment repeats
FULL_N_REPEATS = 20  # Full run uses 20 repeats (final paper results)

# Evaluation parameters
FULL_N_TEST = 2500  # Test set size (for Hat Q_DRO)
FULL_M_ATTACKS = 10  # Full run uses 10 attacks (for computing Hat Q_min)
FULL_N_TEST_ATTACK = 1000  # Test set size for attack phase (smaller to speed up, does not affect result accuracy)
FULL_BASE_SEED = 60000  # Base seed for evaluation phase

# Warm Start parameters
WARM_START_LR = 1e-3  # Warm start learning rate
WARM_START_EPOCHS = 80  # Warm start epochs

# Bandwidth parameters
MIN_BANDWIDTH = 1e-3  # Minimum bandwidth (lower bound of Silverman bandwidth)

# Alpha optimization parameters
ALPHA_OPT_BOUNDS = (0.01, 10.0)  # Alpha search range
ALPHA_OPT_METHOD = 'bounded'  # Optimization method: 'bounded' supports bounds parameter, 'golden'/'brent' requires bracket
ALPHA_OPT_TOL = 1e-3  # Optimization tolerance (reduced to 1e-3 to improve optimization precision, helps enhance performance)

# Gradient clipping
GRAD_CLIP_MAX_NORM = 1.0  # Maximum norm for gradient clipping

# Alpha optimizer learning rate (for alternating optimization during training)
ALPHA_OPTIMIZER_LR = 5e-4  # Reduced to 5e-4 to improve alpha optimization stability

# Alpha update frequency (optimization: reduce update frequency to improve training speed)
# Update alpha every ALPHA_UPDATE_FREQ batches instead of every batch
ALPHA_UPDATE_FREQ = 3  # Default: update alpha every 3 batches

# Figure 3 experiment parameters
FIGURE3_DELTAS = [0.1, 0.2, 0.3]  # Three delta_test values for Kallus Figure 3

# OPE experiment parameters
OPE_N_ESTIMATORS = 20  # Number of trees in forest for OPE experiments
TARGET_POLICY_SEED = 12345  # Seed for random initialization of target policy (ensures reproducibility)
OPE_N_FOLDS = 5  # Number of cross-fitting folds in OPE algorithm
OPE_N_TEST_TRUE = 10000  # Number of test samples used when computing true R_delta
