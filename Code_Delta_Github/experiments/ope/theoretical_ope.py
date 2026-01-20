"""
Theoretical true R_delta computation module
Use numerical integration to directly compute theoretical value, does not depend on sampling
"""
import numpy as np
import torch
from scipy.integrate import quad, dblquad, nquad
from scipy.optimize import root_scalar
from scipy.special import expit
from scipy.stats import norm, truncnorm
from src import config
from src.dgp import PricingEnvironment


def compute_E_exp_R_given_X(X, alpha, target_policy_params, setting='linear', 
                            logging_policy_type='nonlinear'):
    """
    Compute E_P[exp(-R/α) | X], where R = P * Y
    
    Given X, integrate over P:
    E_P[exp(-P*Y/α) | X] = ∫ exp(-p*Y/α) * p_P(p|X) * p_Y(Y|X,p) dp
    
    Since Y is binary, can be written as:
    = ∫ [P(Y=1|X,p)*exp(-p/α) + P(Y=0|X,p)*1] * p_P(p|X) dp
    = ∫ [expit((f(x)+z_loc-p)/z_scale)*exp(-p/α) + 
         (1-expit((f(x)+z_loc-p)/z_scale))*1] * p_P(p|X) dp
    
    Parameters:
    -----------
    X : np.ndarray, shape (dim,)
        Feature vector
    alpha : float
        Dual variable α
    target_policy_params : dict
        Target policy parameters {'weight': np.ndarray, 'bias': float}
    setting : str
        Data generating process setting ('linear', 'piecewise', etc.)
    logging_policy_type : str
        Logging policy type ('nonlinear', 'linear', etc.)
    
    Returns:
    --------
    E_exp_R : float
        E_P[exp(-R/α) | X]
    """
    # 1. Compute f(x) - true valuation
    if setting == 'linear':
        f_x = 1.5 + 1.2 * X[0] + 1.8 * X[1] + 0.3 * X[0] * X[1]
    elif setting == 'piecewise':
        f_x = np.where(X[0] <= 1.0, 4.0 + 0.5 * X[1], 0.5 + 1.0 * X[1])
    else:
        f_x = 2.0 + 1.0 * np.sin(X[0]) ** 2 + 1.0 * np.sin(X[1]) ** 2
    
    # 2. Compute target policy output π(x)
    # Linear policy: π(x) = sigmoid(w^T x + b) * (max_p - min_p) + min_p
    w = target_policy_params['weight']  # shape (dim,)
    b = target_policy_params['bias']  # float
    pi_x_raw = np.dot(X, w) + b
    pi_x = expit(pi_x_raw) * (config.PRICE_MAX - config.PRICE_MIN) + config.PRICE_MIN
    
    # 3. Logging policy parameters
    if logging_policy_type == 'nonlinear':
        mu_old = 1.0 + 0.8 * X[0] + 1.0 * X[1]
        sigma_old = 1.0
    elif logging_policy_type == 'linear':
        mu_old = 1.0 + 1.0 * X[0]
        sigma_old = 1.0
    else:
        mu_old = 1.0 + 1.0 * X[0]
        sigma_old = 1.0
    
    # 4. Noise parameters
    z_loc = config.TRAIN_Z_LOC
    z_scale = config.TRAIN_Z_SCALE
    
    # 5. Define integrand (integrate over P)
    def integrand_p(p):
        """
        Integrand: given P=p, compute expectation of exp(-p*Y/α) (over Y)
        """
        # P(Y=1|X,p) = expit((f(x)+z_loc-p)/z_scale)
        prob_y1 = expit((f_x + z_loc - p) / z_scale)
        prob_y0 = 1.0 - prob_y1
        
        # E[exp(-p*Y/α)] = prob_y1 * exp(-p/α) + prob_y0 * 1
        exp_term = prob_y1 * np.exp(-p / alpha) + prob_y0
        
        # Density of P (truncated normal distribution)
        # Note: P is clipped to [PRICE_MIN, PRICE_MAX]
        if p < config.PRICE_MIN or p > config.PRICE_MAX:
            return 0.0
        
        # Density of truncated normal distribution
        a = (config.PRICE_MIN - mu_old) / sigma_old
        b = (config.PRICE_MAX - mu_old) / sigma_old
        p_density = truncnorm.pdf(p, a, b, loc=mu_old, scale=sigma_old)
        
        return exp_term * p_density
    
    # 6. Numerical integration over P
    # Integration range: [PRICE_MIN, PRICE_MAX]
    # For numerical stability, slightly extend the range
    p_low = max(config.PRICE_MIN - 3*sigma_old, 0.0)
    p_high = config.PRICE_MAX + 3*sigma_old
    
    result, _ = quad(integrand_p, p_low, p_high, 
                     epsabs=1e-6, epsrel=1e-6, limit=100)
    
    return result


def compute_E_exp_R(alpha, target_policy_params, setting='linear', 
                    logging_policy_type='nonlinear', n_mc_samples=10000):
    """
    Compute E_P[exp(-R/α)], integrating over X
    
    Use Monte Carlo method to integrate over X:
    E_P[exp(-R/α)] = E_X[E_P[exp(-R/α) | X]]
    
    Parameters:
    -----------
    alpha : float
        Dual variable α
    target_policy_params : dict
        Target policy parameters
    setting : str
        Data generating process setting
    logging_policy_type : str
        Logging policy type
    n_mc_samples : int
        Number of Monte Carlo samples (for integrating over X)
    
    Returns:
    --------
    E_exp_R : float
        E_P[exp(-R/α)]
    """
    # Distribution of X: X ~ N(mean, std)
    x_mean = config.TRAIN_X_MEAN
    x_std = config.TRAIN_X_STD
    
    # Generate X samples
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    X_samples = rng.normal(x_mean, x_std, size=(n_mc_samples, config.DIM))
    
    # For each X sample, compute E_P[exp(-R/α) | X]
    E_exp_R_samples = []
    for i in range(n_mc_samples):
        try:
            E_given_X = compute_E_exp_R_given_X(
                X_samples[i], alpha, target_policy_params, 
                setting, logging_policy_type
            )
            E_exp_R_samples.append(E_given_X)
        except:
            # If integration fails, skip
            continue
    
    if len(E_exp_R_samples) == 0:
        return 1e-10  # Avoid division by zero
    
    return np.mean(E_exp_R_samples)


def solve_alpha_star(delta, target_policy_params, setting='linear', 
                     logging_policy_type='nonlinear', n_mc_samples=10000):
    """
    Solve for optimal dual variable α*
    
    According to moment equation:
    -log(E_P[exp(-R/α*)]) - E_P[R*exp(-R/α*)]/(α*E_P[exp(-R/α*)]) - δ = 0
    
    For IPW method, simplified to:
    -log(E_P[exp(-R/α*)]) - δ = 0
    
    Parameters:
    -----------
    delta : float
        Uncertainty radius δ
    target_policy_params : dict
        Target policy parameters
    setting : str
        Data generating process setting
    logging_policy_type : str
        Logging policy type
    n_mc_samples : int
        Number of Monte Carlo samples
    
    Returns:
    --------
    alpha_star : float
        Optimal dual variable α*
    """
    def moment_equation(alpha_val):
        """Moment equation: -log(E_P[exp(-R/α)]) - δ = 0"""
        if alpha_val <= 1e-4:
            return 1e9
        
        E_exp_R = compute_E_exp_R(alpha_val, target_policy_params, 
                                   setting, logging_policy_type, n_mc_samples)
        
        if E_exp_R <= 1e-10:
            return 1e9
        
        moment = -np.log(E_exp_R) - delta
        return moment
    
    # Solve moment equation
    try:
        bracket_low = config.ALPHA_OPT_BOUNDS[0]
        bracket_high = config.ALPHA_OPT_BOUNDS[1]
        
        res = root_scalar(moment_equation, bracket=[bracket_low, bracket_high],
                         method='brentq', xtol=1e-5)
        alpha_star = res.root
    except:
        # Fallback: use optimization method
        from scipy.optimize import minimize_scalar
        
        def objective(alpha_val):
            moment = moment_equation(alpha_val)
            return abs(moment)
        
        res = minimize_scalar(objective, bounds=config.ALPHA_OPT_BOUNDS,
                             method=config.ALPHA_OPT_METHOD,
                             options={'xatol': config.ALPHA_OPT_TOL})
        alpha_star = res.x
    
    return alpha_star


def compute_theoretical_R_delta(target_policy, delta_test, setting='linear',
                                logging_policy_type='nonlinear', 
                                n_mc_samples=10000):
    """
    Directly compute theoretical true R_delta using numerical integration
    
    Formula: R_δ(π) = -α* log(E_P[exp(-R/α*)]) - α*δ
    
    Parameters:
    -----------
    target_policy : torch.nn.Module
        Target policy (LinearPolicy)
    delta_test : float
        Test uncertainty radius δ
    setting : str
        Data generating process setting
    logging_policy_type : str
        Logging policy type
    n_mc_samples : int
        Number of Monte Carlo samples (for integrating over X)
    
    Returns:
    --------
    R_delta : float
        Theoretical true R_delta
    alpha_star : float
        Optimal dual variable α*
    """
    # 1. Extract target policy parameters
    if hasattr(target_policy, 'linear'):
        # LinearPolicy
        weight = target_policy.linear.weight.detach().cpu().numpy().flatten()
        bias = target_policy.linear.bias.detach().cpu().numpy().item()
    else:
        raise ValueError("Only LinearPolicy is supported for theoretical computation")
    
    target_policy_params = {
        'weight': weight,
        'bias': bias
    }
    
    # 2. Solve for optimal dual variable α*
    alpha_star = solve_alpha_star(delta_test, target_policy_params, 
                                   setting, logging_policy_type, n_mc_samples)
    
    # 3. Compute E_P[exp(-R/α*)]
    E_exp_R_star = compute_E_exp_R(alpha_star, target_policy_params, 
                                    setting, logging_policy_type, n_mc_samples)
    
    # 4. Compute R_δ
    R_delta = -alpha_star * np.log(max(E_exp_R_star, 1e-10)) - alpha_star * delta_test
    
    return R_delta, alpha_star
