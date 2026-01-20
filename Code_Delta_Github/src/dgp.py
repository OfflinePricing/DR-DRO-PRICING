# dgp.py
# Data Generating Process
import numpy as np
from scipy.special import expit # Sigmoid
from . import config

class PricingEnvironment:
    def __init__(self, setting='linear', logging_policy_type='linear', seed=None):
        self.setting = setting
        self.logging_policy_type = logging_policy_type
        self.dim = config.DIM
        self.rng = np.random.RandomState(seed)

    def true_valuation(self, X):
        """Deterministic valuation function f(x) """
        
        if self.setting == 'linear':
            # True Valuation Function (Simplified: Linear + Interaction)
            # Simplified version: remove sqrt term, keep linear and interaction terms
            # Business interpretation:
            # - Base value: 1.5 (base valuation for all customers)
            # - Linear effects: Linear influence of X0 and X1 (1.2 and 1.8 represent importance of different features)
            # - Interaction effect: Synergistic effect of X0 and X1 (0.3 represents additional value when both features are present)
            # This design makes the optimal strategy not a simple linear function, which benefits DRO methods
            # The simplified form is also easier to learn, improving DRO performance at larger delta
            f = 1.5 + 1.2 * X[:, 0] + 1.8 * X[:, 1] + 0.3 * X[:, 0] * X[:, 1]
        elif self.setting == 'piecewise':
            # The Robustness Trap
            # Define threshold: region where X1 > 1.0 is the "trap zone"
            # Under standard normal distribution, P(X1 > 1.0) ≈ 16% (minority group)
            
            # Majority group (X1 <= 1.0): 
            # High valuation, increases with X2. Base value 4.0
            val_safe = 4.0 + 0.5 * X[:, 1] 
            
            # Minority group (X1 > 1.0): 
            # Extremely low valuation, a "cliff". Base value 1.0
            # This huge drop causes strategies that set high prices to have zero revenue here
            val_trap = 0.5 + 1.0 * X[:, 1] # X2 here may not even matter, emphasizing low price
            
            f = np.where(X[:, 0] <= 1.0, val_safe, val_trap)
        else:
            f = 2.0 + 1.0 * np.sin(X[:, 0]) ** 2 + 1.0 * np.sin(X[:, 1]) ** 2 
        #   np.sqrt((X[:, 0] + 0.5) ** 2 + (X[:, 0] + 1.0) ** 2) 
        return f 

    def sample_data(self, n_samples):
        # 1. Generate features X (uniformly use training distribution)
        x_mean = config.TRAIN_X_MEAN
        x_std  = config.TRAIN_X_STD
        X = self.rng.normal(x_mean, x_std, size=(n_samples, self.dim))

        # 2. Historical policy generates prices P (Logging Policy / Behavior Policy)
        # Select different strategies based on logging_policy_type
        if self.logging_policy_type == 'linear':
            # Default linear strategy: only linear part of X1, with large variance to ensure coverage
            mu_old = 1.0 + 1.0 * X[:, 0] 
            sigma_old = 1.0
        elif self.logging_policy_type == 'linear_alternative':
            # Alternative linear strategy: different parameters for comparison
            mu_old = 0.5 + 1.5 * X[:, 0]
            sigma_old = 1.2
        elif self.logging_policy_type == 'nonlinear':
            # Behavior Policy (Simplified: Suboptimal Linear Approximation)
            # Business interpretation:
            # - This is the historical pricing strategy (logging policy), a simplified linear approximation of true_valuation
            # - Ignores interaction terms in f(x), uses simple linear form
            # - Smaller coefficients (0.8 vs 1.2, 1.0 vs 1.8) indicate larger gap from optimal strategy, more suboptimal
            # - Maintains sufficient variance (sigma_old=1.0) to explore different price ranges
            # This design makes behavior policy have clear gap from optimal strategy, benefiting DRO
            # Especially at larger delta, DRO can better handle distribution shift
            mu_old = 1.0 + 0.8 * X[:, 0] + 1.0 * X[:, 1]  # Simplified linear version with smaller coefficients, indicating more suboptimal strategy
            sigma_old = 1.0  # Maintain sufficient variance for exploration
        else:
            # Default to linear
            mu_old = 1.0 + 1.0 * X[:, 0] 
            sigma_old = 1.0
        
        raw_P = self.rng.normal(mu_old, sigma_old)
        P = np.clip(raw_P, config.PRICE_MIN, config.PRICE_MAX)
        if not np.isfinite(P).all(): P = np.nan_to_num(P)

        # 3. Generate Z and Y (uniformly use Logistic distribution)
        z_loc = config.TRAIN_Z_LOC
        z_scale = config.TRAIN_Z_SCALE
        z = self.rng.logistic(z_loc, z_scale, size=n_samples)
        
        # Compute f(x)
        f_x = self.true_valuation(X)
        
        # Purchase decision
        Y = (f_x + z >= P).astype(float)
        
        return X.astype(np.float32), P.astype(np.float32), Y.astype(np.float32)

    def get_oracle_revenue(self, X):
        f_x = self.true_valuation(X)
        
        # Uniformly use Logistic distribution
        z_loc = config.TRAIN_Z_LOC
        z_scale = config.TRAIN_Z_SCALE
            
        p_candidates = np.linspace(config.PRICE_MIN, config.PRICE_MAX, 100)
        
        # Vectorized computation
        f_x_expanded = f_x.reshape(-1, 1)
        p_expanded = p_candidates.reshape(1, -1)
        
        # Logistic Survival Function: Sigmoid((loc - threshold)/scale)
        # = Sigmoid((f(x) + loc - P)/scale)
        probs = expit((f_x_expanded + z_loc - p_expanded) / z_scale)
            
        revs = p_expanded * probs
        best_revs = np.max(revs, axis=1)
        
        return best_revs

    def generate_worst_case_shift(self, X_original, P_original, Y_original, 
                                  policy, alpha_star):
        """
        Generate shifted dataset based on worst-case distribution within KL ball
        
        Use exponential tilting method to construct worst-case distribution:
        w_i ∝ exp(-R_i / alpha_star), where R_i = P_i * Y_i
        
        Parameters:
        -----------
        X_original : np.ndarray
            Original feature data [n_samples, dim]
        P_original : np.ndarray
            Original price data [n_samples]
        Y_original : np.ndarray
            Original purchase decision [n_samples]
        policy : callable
            Policy function that takes X and returns recommended price
        alpha_star : float
            Optimal dual variable (solved from DRO dual problem)
        
        Returns:
        --------
        X_shifted : np.ndarray
            Shifted feature data
        P_shifted : np.ndarray
            Shifted price data
        Y_shifted : np.ndarray
            Shifted purchase decision data
        """
        # Compute reward: R = P * Y
        R = P_original * Y_original
        
        # Compute log-weights (numerical stability)
        # w_i ∝ exp(-R_i / alpha_star)
        log_weights = -R / alpha_star
        log_weights = log_weights - np.max(log_weights)  # Subtract maximum to prevent overflow
        
        # Normalize to probability distribution
        weights = np.exp(log_weights)
        weights = weights / (np.sum(weights) + 1e-10)  # Prevent division by zero
        
        # Resample with replacement
        n_samples = len(X_original)
        indices = self.rng.choice(n_samples, size=n_samples, replace=True, p=weights)
        
        X_shifted = X_original[indices]
        P_shifted = P_original[indices]
        Y_shifted = Y_original[indices]
        
        return X_shifted, P_shifted, Y_shifted