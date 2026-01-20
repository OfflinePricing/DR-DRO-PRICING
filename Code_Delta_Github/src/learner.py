
# learner.py
import torch
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar
from . import config
from sklearn.model_selection import KFold
from .estimators import PropensityEstimator, ProbabilityOutcomeEstimator
from econml.grf import RegressionForest  # type: ignore


class BaseLearner:
    def __init__(self, policy_net, bandwidth, delta=config.DELTA):
        self.policy = policy_net
        self.h = bandwidth
        self.delta = delta
        # All algorithms use Adam to optimize policy network
        self.opt_pi = optim.Adam(self.policy.parameters(), lr=config.LR_POLICY)

    def gaussian_kernel(self, u):
        return (1.0 / np.sqrt(2 * np.pi)) * torch.exp(-0.5 * u**2)

    def get_optimal_alpha(self, X_batch, P_batch, Y_batch, Pi0_batch, current_delta, alpha_init=None):
        """
        For given test set and Delta, compute optimal dual variable Alpha,
        and return theoretical worst-case value (Hat Q_DRO).
        
        Parameters:
        -----------
        alpha_init : float, optional
            Initial guess for alpha (warm start). If provided, will be used to set a better bracket.
        """
        X = torch.FloatTensor(X_batch)
        P = torch.FloatTensor(P_batch).view(-1, 1)
        Y = torch.FloatTensor(Y_batch).view(-1, 1)
        Pi0 = torch.FloatTensor(Pi0_batch).view(-1, 1)
        
        with torch.no_grad():
            pi_x = self.policy(X)
            u = (P - pi_x) / self.h
            kernel_val = self.gaussian_kernel(u)
            Pi0_safe = torch.clamp(Pi0, min=config.ETA)
            
            # Base weights C_t
            C_t = kernel_val / (self.h * Pi0_safe)
            C_t = torch.clamp(C_t, max=20.0) 
            S_T = torch.mean(C_t) + 1e-8
            
            R = P * Y
            
            # Dual objective function g(alpha)
            # Dual = - alpha * log(E[exp(-R/alpha)]) - alpha * delta (Maximize)
            # Minimize -Dual => alpha * log_W_hat + alpha * delta
            def objective(alpha_val):
                if alpha_val <= 1e-4: return 1e9
                log_C = torch.log(C_t + 1e-10)
                exponent = log_C - (R / alpha_val)
                # LogSumExp trick
                max_exp = torch.max(exponent)
                sum_exp = torch.sum(torch.exp(exponent - max_exp))
                log_mean_W = max_exp + torch.log(sum_exp) - np.log(len(X))
                
                log_W_hat = log_mean_W - torch.log(S_T)
                return alpha_val * log_W_hat.item() + alpha_val * current_delta

            # Search for optimal Alpha
            if config.ALPHA_OPT_METHOD in ['bounded']:
                res = minimize_scalar(objective, bounds=config.ALPHA_OPT_BOUNDS, 
                                    method=config.ALPHA_OPT_METHOD, 
                                    options={'xatol': config.ALPHA_OPT_TOL})
            else:
                # 'golden' and 'brent' methods require bracket parameter
                # If alpha_init is provided, use it to set a better bracket
                if alpha_init is not None and config.ALPHA_OPT_BOUNDS[0] < alpha_init < config.ALPHA_OPT_BOUNDS[1]:
                    # Use warm start alpha as midpoint of bracket
                    bracket_range = min(alpha_init - config.ALPHA_OPT_BOUNDS[0], 
                                       config.ALPHA_OPT_BOUNDS[1] - alpha_init) * 0.5
                    bracket = (max(config.ALPHA_OPT_BOUNDS[0], alpha_init - bracket_range),
                              alpha_init,
                              min(config.ALPHA_OPT_BOUNDS[1], alpha_init + bracket_range))
                else:
                    # Default bracket
                    bracket = (config.ALPHA_OPT_BOUNDS[0], 
                              (config.ALPHA_OPT_BOUNDS[0] + config.ALPHA_OPT_BOUNDS[1]) / 2,
                              config.ALPHA_OPT_BOUNDS[1])
                res = minimize_scalar(objective, bracket=bracket, 
                                    method=config.ALPHA_OPT_METHOD, 
                                    options={'xatol': config.ALPHA_OPT_TOL})
            
            # res.x is optimal alpha, -res.fun is maximized objective value (Hat Q_DRO)
            return res.x, -res.fun

    def get_worst_case_weights(self, X_batch, P_batch, Y_batch, Pi0_batch, alpha_star):
        """
        Compute sample resampling weights under worst-case distribution.
        Used to construct "Sampled on KL-sphere" attack dataset.
        Weights w_i \propto C_t * exp(-R_i / alpha*)
        """
        X = torch.FloatTensor(X_batch)
        P = torch.FloatTensor(P_batch).view(-1, 1)
        Y = torch.FloatTensor(Y_batch).view(-1, 1)
        Pi0 = torch.FloatTensor(Pi0_batch).view(-1, 1)
        
        with torch.no_grad():
            pi_x = self.policy(X)
            u = (P - pi_x) / self.h
            kernel_val = self.gaussian_kernel(u)
            Pi0_safe = torch.clamp(Pi0, min=config.ETA)
            C_t = kernel_val / (self.h * Pi0_safe)
            C_t = torch.clamp(C_t, max=20.0)
            
            R = P * Y
            
            # Compute unnormalized Tilt weights (compute in log domain to prevent overflow)
            log_weights = torch.log(C_t + 1e-10) - (R / alpha_star)
            
            # Softmax normalization to get probability distribution
            weights = torch.softmax(log_weights, dim=0).flatten().numpy()
            
            return weights

class CDROLearner(BaseLearner):
    """
    Continuum Doubly Robust DRO OPL for Continuous Pricing (CDR²O²PL-CP)
    Implements Algorithm 2 from offline_pricing.tex
    """
    def __init__(self, policy_net, bandwidth, delta=config.DELTA, 
                 X_train=None, P_train=None, Y_train=None, n_folds=5,
                 alpha_update_freq=5):
        super().__init__(policy_net, bandwidth, delta)
        
        # Alpha parameter for alternating optimization
        init_val = 1.0
        init_raw = np.log(np.exp(init_val) - 1.0)
        self.raw_alpha = torch.tensor(init_raw, requires_grad=True)
        self.opt_alpha = optim.Adam([self.raw_alpha], lr=config.ALPHA_OPTIMIZER_LR)
        
        # Learning rate scheduler for policy optimizer
        # Use CosineAnnealingLR for smooth decay
        # T_max will be adjusted based on actual training epochs
        # Default to a reasonable value, can be updated if needed
        self.scheduler_pi = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_pi, T_max=config.EPOCHS, eta_min=config.LR_POLICY * 0.01
        )
        
        # Alpha update frequency control (optimization: update alpha less frequently)
        # Update alpha every alpha_update_freq batches instead of every batch
        self.alpha_update_freq = alpha_update_freq
        self.batch_counter = 0  # Track batch count for alpha update frequency
        self.epoch_counter = 0  # Track epoch count for learning rate scheduling
        
        # Phase 1: Cross-fitting + Forest Training
        if X_train is not None and P_train is not None and Y_train is not None:
            self._setup_cross_fitting(X_train, P_train, Y_train, n_folds)
        else:
            # For backward compatibility, allow initialization without data
            # But Phase 1 must be called separately before training
            self.fold_indices = None
            self.prop_models = None
            self.forest_models = None
            self.X_train = None
            self.P_train = None
            self.Y_train = None
            self.n_folds = n_folds

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.raw_alpha) + 1e-4

    def _setup_cross_fitting(self, X_train, P_train, Y_train, n_folds=5):
        """
        Phase 1: Cross-fitting setup
        Partition data into L folds and train propensity + forest models for each fold
        """
        n_samples = len(X_train)
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.P_train = np.asarray(P_train, dtype=np.float64)
        self.Y_train = np.asarray(Y_train, dtype=np.float64)
        self.n_folds = n_folds
        
        # Compute revenue for forest training
        R_train = self.P_train * self.Y_train
        
        # Create folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)
        self.fold_indices = []
        self.prop_models = []
        self.forest_models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            self.fold_indices.append({
                'train': train_idx,
                'val': val_idx
            })
            
            # Train propensity model on out-of-fold data
            prop_model = PropensityEstimator()
            prop_model.fit(X_train[train_idx], P_train[train_idx])
            self.prop_models.append(prop_model)
            
            # Train forest model on out-of-fold data
            # Features: (X, P), Target: R = P * Y
            X_P_train = np.column_stack([X_train[train_idx], P_train[train_idx]])
            
            # Use econml GRF (RegressionForest)
            # Use fewer estimators and shallower trees in test mode (if available via config)
            n_estimators = getattr(config, 'TEST_N_ESTIMATORS', 100)
            max_depth = getattr(config, 'TEST_FOREST_MAX_DEPTH', 10)
            forest = RegressionForest(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=config.SEED + fold_idx
            )
            
            forest.fit(X_P_train, R_train[train_idx])
            self.forest_models.append(forest)
        
        print(f"Phase 1 complete: {n_folds} folds, {n_samples} samples")

    def _get_forest_weights(self, x_query, p_query, fold_idx):
        """
        Compute forest-induced weights ω_t(x, p) for query point (x, p)
        Returns weights for all training samples in fold ℓ's training set
        Formula: ω_t(x, p) = (1/B) Σ_{b=1}^B 1[(x_t, p_t) ∈ L_b(x, p)] / |L_b(x, p)|
        
        Optimized: Unified implementation that calls batch version for efficiency.
        """
        # Use batch version for unified implementation (single query is just batch of size 1)
        weights_matrix = self._get_forest_weights_matrix(x_query, p_query, fold_idx)
        # Return first (and only) row as 1D array
        return weights_matrix[0] if weights_matrix.shape[0] == 1 else weights_matrix.flatten()

    def _get_forest_weights_matrix(self, X_batch, P_batch, fold_idx):
        """
        Batch version: Compute forest-induced weights for multiple query points
        Returns [n_query, n_train] weight matrix
        Each row corresponds to one query point's weights over training samples
        """
        if self.forest_models is None:
            raise ValueError("Cross-fitting not initialized. Call _setup_cross_fitting first.")
        
        forest = self.forest_models[fold_idx]
        train_idx = self.fold_indices[fold_idx]['train']
        
        # Optimization: Use views instead of copies where possible
        X_train_fold = self.X_train[train_idx]
        P_train_fold = self.P_train[train_idx]
        # Pre-allocate X_P_train to avoid repeated column_stack
        X_P_train = np.empty((len(train_idx), X_train_fold.shape[1] + 1), dtype=X_train_fold.dtype)
        X_P_train[:, :-1] = X_train_fold
        X_P_train[:, -1] = P_train_fold
        
        # Ensure proper shapes (use reshape instead of asarray when possible to avoid copy)
        if not isinstance(X_batch, np.ndarray):
            X_batch = np.asarray(X_batch)
        if not isinstance(P_batch, np.ndarray):
            P_batch = np.asarray(P_batch)
        
        if X_batch.ndim == 1:
            X_batch = X_batch.reshape(1, -1)
        if P_batch.ndim == 0:
            P_batch = P_batch.reshape(1)
        elif P_batch.ndim == 1 and len(P_batch) == 1 and X_batch.shape[0] > 1:
            P_batch = np.repeat(P_batch, X_batch.shape[0])
        
        n_query = X_batch.shape[0]
        n_train = len(train_idx)
        # Pre-allocate X_P_query to avoid repeated column_stack
        X_P_query = np.empty((n_query, X_batch.shape[1] + 1), dtype=X_batch.dtype)
        X_P_query[:, :-1] = X_batch
        X_P_query[:, -1] = P_batch.reshape(-1)
        
        # Initialize weight matrix: [n_query, n_train]
        weights_matrix = np.zeros((n_query, n_train))
        
        # Get trees
        if hasattr(forest, 'estimators_'):
            trees = forest.estimators_
        elif hasattr(forest, 'grf_'):
            trees = forest.grf_.estimators_ if hasattr(forest.grf_, 'estimators_') else []
        else:
            # Fallback: uniform weights
            weights_matrix[:] = 1.0 / n_train
            return weights_matrix
        
        n_trees = len(trees) if len(trees) > 0 else 1
        
        # Vectorized: process all trees and all query points efficiently
        # Optimization: Pre-allocate arrays and use in-place operations where possible
        for tree in trees:
            # Get leaf indices
            train_leaves = tree.apply(X_P_train)  # [n_train]
            query_leaves = tree.apply(X_P_query)  # [n_query]
            
            # Vectorized: compare all query points with all training samples
            # matches: [n_train, n_query] - True where train sample and query are in same leaf
            # Optimization: Use broadcasting directly without intermediate array if possible
            matches = (train_leaves[:, None] == query_leaves[None, :])  # [n_train, n_query]
            
            # Compute leaf sizes for each query point
            leaf_sizes = np.sum(matches, axis=0, dtype=np.float64)  # [n_query] - use float64 for division
            
            # Compute weight contributions: [n_train, n_query]
            # Optimization: Avoid division by zero and use efficient operations
            leaf_sizes_safe = np.maximum(leaf_sizes, 1.0)  # Avoid division by zero
            # Use in-place division where possible (matches is boolean, so we convert once)
            weight_contributions = matches.astype(np.float64)  # [n_train, n_query]
            weight_contributions /= leaf_sizes_safe[None, :]  # In-place division
            
            # Accumulate weights: transpose to get [n_query, n_train]
            # Optimization: Use += for in-place addition
            weights_matrix += weight_contributions.T  # [n_query, n_train]
        
        # Average over trees
        if n_trees > 0:
            weights_matrix = weights_matrix / n_trees
        else:
            weights_matrix[:] = 1.0 / n_train
        
        return weights_matrix

    def _compute_m_hat(self, x, p, alpha, fold_idx):
        """
        Compute m̂_0^(ℓ)(x, p; α) = Σ_{t ∈ I_ℓ^c} ω_t^(ℓ)(x, p) exp(-p_t y_t / α)
        Single query point version (backward compatible)
        """
        if self.forest_models is None:
            raise ValueError("Cross-fitting not initialized.")
        
        train_idx = self.fold_indices[fold_idx]['train']
        weights = self._get_forest_weights(x, p, fold_idx)
        
        P_train_fold = self.P_train[train_idx]
        Y_train_fold = self.Y_train[train_idx]
        R_train_fold = P_train_fold * Y_train_fold
        
        # Compute weighted sum: Σ ω_t * exp(-R_t / α)
        exp_terms = np.exp(-R_train_fold / alpha)
        m_hat = np.sum(weights * exp_terms)
        
        return m_hat

    def _compute_m_hat_batch(self, X_batch, P_batch, alpha, fold_idx):
        """
        Batch version: Compute m̂_0^(ℓ)(x, p; α) for all query points at once
        X_batch: [n_query, dim_x] or [dim_x] (will be reshaped)
        P_batch: [n_query] or scalar
        Returns: [n_query] array of m_hat values
        """
        if self.forest_models is None:
            raise ValueError("Cross-fitting not initialized.")
        
        # Ensure proper shapes
        X_batch = np.asarray(X_batch)
        P_batch = np.asarray(P_batch)
        
        if X_batch.ndim == 1:
            X_batch = X_batch.reshape(1, -1)
        if P_batch.ndim == 0:
            P_batch = P_batch.reshape(1)
        elif P_batch.ndim == 1 and len(P_batch) == 1 and X_batch.shape[0] > 1:
            # Broadcast single p to all x
            P_batch = np.repeat(P_batch, X_batch.shape[0])
        
        n_query = X_batch.shape[0]
        train_idx = self.fold_indices[fold_idx]['train']
        n_train = len(train_idx)
        
        # Pre-compute exp terms (same for all query points)
        P_train_fold = self.P_train[train_idx]
        Y_train_fold = self.Y_train[train_idx]
        R_train_fold = P_train_fold * Y_train_fold
        exp_terms = np.exp(-R_train_fold / alpha)  # [n_train]
        
        # Optimized: Batch compute weights for all query points
        # Use _get_forest_weights_matrix to get [n_query, n_train] weight matrix at once
        weights_matrix = self._get_forest_weights_matrix(X_batch, P_batch, fold_idx)
        
        # Vectorized batch computation: [n_query] = sum over train samples
        m_hat_values = np.sum(weights_matrix * exp_terms[None, :], axis=1)  # [n_query]
        
        return m_hat_values

    def _compute_W_hat(self, X_batch, P_batch, Y_batch, pi_x_batch, alpha, fold_idx, 
                       cached_propensity=None, cached_kernel=None):
        """
        Compute Ŵ^(l)(π, α) using DR estimator (Equation 511-516)
        Ŵ^(l)(π, α) = (1/|I_l|) Σ_{t ∈ I_l} [m̂_0(x_t, π(x_t); α) + 
                      (1/Ŝ_l^π) * (K(...)/h*π̂_0) * (exp(-p_t y_t/α) - m̂_0(x_t, p_t; α))]
        
        Parameters:
        -----------
        cached_propensity : dict, optional
            Cached propensity values and kernel weights to avoid recomputation
        cached_kernel : dict, optional
            Cached kernel weights and S_ell_pi
        """
        if self.prop_models is None or self.forest_models is None:
            raise ValueError("Cross-fitting not initialized.")
        
        # Avoid unnecessary array conversions if already numpy arrays
        if not isinstance(X_batch, np.ndarray):
            X_batch = np.asarray(X_batch)
        if not isinstance(P_batch, np.ndarray):
            P_batch = np.asarray(P_batch)
        if not isinstance(Y_batch, np.ndarray):
            Y_batch = np.asarray(Y_batch)
        if not isinstance(pi_x_batch, np.ndarray):
            pi_x_batch = np.asarray(pi_x_batch)
        n_batch = len(X_batch)
        
        prop_model = self.prop_models[fold_idx]
        
        # Optimized: Batch compute m̂_0 for all query points at once
        # Merge policy and observed query points to compute in one batch
        X_all = np.vstack([X_batch, X_batch])  # [2*n_batch, dim_x]
        P_all = np.concatenate([pi_x_batch, P_batch])  # [2*n_batch]
        
        # Batch compute all m_hat values at once
        m_all_values = self._compute_m_hat_batch(X_all, P_all, alpha, fold_idx)
        
        # Separate policy and observed results
        m_pi_values = m_all_values[:n_batch]  # [n_batch]
        m_obs_values = m_all_values[n_batch:]  # [n_batch]
        
        # Use cached values if available, otherwise compute
        if cached_propensity is not None and fold_idx in cached_propensity:
            pi0_values = cached_propensity[fold_idx]
        else:
            pi0_values = prop_model.predict(X_batch, P_batch)
            pi0_values = np.maximum(pi0_values, config.ETA)
        
        if cached_kernel is not None and fold_idx in cached_kernel:
            kernel_weights = cached_kernel[fold_idx]['kernel_weights']
            S_ell_pi = cached_kernel[fold_idx]['S_ell_pi']
        else:
            # Kernel: K((p_t - π(x_t)) / h) / (h * π̂_0(p_t | x_t))
            u_vals = (P_batch - pi_x_batch) / self.h
            kernel_vals = np.exp(-0.5 * u_vals**2) / np.sqrt(2 * np.pi)
            kernel_weights = kernel_vals / (self.h * pi0_values)
            kernel_weights = np.clip(kernel_weights, 0, 20.0)
            # Compute Ŝ_ℓ^π = (1/|I_ℓ|) Σ_{t ∈ I_ℓ} K(...) / (h * π̂_0(...))
            S_ell_pi = np.mean(kernel_weights) + 1e-8
        
        # DR estimator components
        R_batch = P_batch * Y_batch
        exp_terms = np.exp(-R_batch / alpha)
        
        # IPW correction term: (K(...)/h*π̂_0) * (exp(-p_t y_t/α) - m̂_0(x_t, p_t; α))
        ipw_correction = kernel_weights * (exp_terms - m_obs_values)
        
        # Final DR estimator: (1/|I_ℓ|) Σ [m̂_0(x_t, π(x_t); α) + (1/Ŝ_ℓ^π) * IPW correction]
        W_hat = np.mean(m_pi_values) + np.mean(ipw_correction) / S_ell_pi
        
        return W_hat

    def train_step(self, X_batch, P_batch, Y_batch, Propensity_batch):
        """
        Alternating optimization: update π and α using DR estimator
        """
        if self.forest_models is None:
            raise ValueError("Cross-fitting not initialized. Call _setup_cross_fitting first.")
        
        X = torch.FloatTensor(X_batch)
        P = torch.FloatTensor(P_batch).view(-1, 1)
        Y = torch.FloatTensor(Y_batch).view(-1, 1)
        
        # 1. Update Policy (Minimization Step)
        curr_alpha = self.alpha.detach().item()
        
        # Convert to numpy once (optimization: avoid repeated conversions)
        # Note: We'll recompute pi_x for gradients, but cache other conversions
        with torch.no_grad():
            pi_x_no_grad = self.policy(X)
        
        pi_x_np_no_grad = pi_x_no_grad.numpy().flatten()
        X_np = X.numpy()
        P_np = P.numpy().flatten()
        Y_np = Y.numpy().flatten()
        
        # Optimization: Removed unnecessary W_hat monitoring computation
        # This was only for monitoring and not used in gradient computation
        # Removing it significantly reduces computation time, especially for large datasets
        
        # Policy loss: minimize Ŵ_T(π, α)
        # We need to make this differentiable w.r.t. π
        # For efficiency, we use a surrogate loss that approximates the DR objective
        # The key is that m_hat terms are pre-computed and don't depend on π gradients
        # We use IPW-like computation with DR correction approximated via m_hat
        
        pi_x = self.policy(X)  # Recompute for gradients (required for backprop)
        u = (P - pi_x) / self.h
        kernel_val = self.gaussian_kernel(u)
        Pi0 = torch.FloatTensor(Propensity_batch).view(-1, 1)
        Pi0_safe = torch.clamp(Pi0, min=config.ETA)
        C_t = kernel_val / (self.h * Pi0_safe)
        C_t = torch.clamp(C_t, max=20.0)
        S_T = torch.mean(C_t).detach()
        
        R = P * Y
        # Simplified policy loss: approximate DR by using IPW with exp(-R/α)
        # The full DR would require recomputing m_hat for each π update, which is expensive
        # This approximation is reasonable when m_hat is well-estimated
        weights = C_t * torch.exp(-R / curr_alpha)
        loss_pi = torch.mean(weights) / (S_T + 1e-8)
        
        self.opt_pi.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=config.GRAD_CLIP_MAX_NORM)
        self.opt_pi.step()
        
        # 2. Update Alpha (Maximization Step) - Optimization: Dynamic update frequency
        # Update alpha more frequently in early training, less frequently later
        self.batch_counter += 1
        
        # Dynamic alpha update frequency: more frequent in early training
        # Early: every 2-3 batches, Later: every 5-7 batches
        if self.epoch_counter < 10:  # First 10 epochs: more frequent
            dynamic_freq = max(2, self.alpha_update_freq - 1)
        elif self.epoch_counter < 20:  # Epochs 10-20: medium frequency
            dynamic_freq = self.alpha_update_freq
        else:  # Later epochs: less frequent
            dynamic_freq = min(7, self.alpha_update_freq + 2)
        
        should_update_alpha = (self.batch_counter % dynamic_freq == 0)
        
        if should_update_alpha:
            # Reuse pi_x from policy update (no need to recompute if policy didn't change much)
            # But we use the updated policy for accuracy
            with torch.no_grad():
                pi_x_const = self.policy(X)
            
            pi_x_const_np = pi_x_const.numpy().flatten()
            
            # Compute Ŵ_T(π, α) for alpha update using full DR estimator
            # Reuse X_np, P_np, Y_np from policy step (already converted)
            W_hat_alpha_values = []
            for fold_idx in range(self.n_folds):
                try:
                    W_hat_ell = self._compute_W_hat(X_np, P_np, Y_np, pi_x_const_np, 
                                                    self.alpha.item(), fold_idx)
                    W_hat_alpha_values.append(W_hat_ell)
                except Exception as e:
                    print(f"Warning: Fold {fold_idx} alpha computation failed: {e}")
                    continue
            
            if len(W_hat_alpha_values) == 0:
                W_hat_alpha = 1.0
            else:
                W_hat_alpha = np.mean(W_hat_alpha_values)
            
            # Alpha loss: maximize -α log(Ŵ_T) - αδ
            # Minimize: α log(Ŵ_T) + αδ
            alpha_var = self.alpha
            log_W_hat = torch.log(torch.clamp(torch.tensor(W_hat_alpha, dtype=torch.float32), min=1e-10))
            loss_alpha = alpha_var * log_W_hat + alpha_var * self.delta
            
            self.opt_alpha.zero_grad()
            loss_alpha.backward()
            self.opt_alpha.step()
            
            alpha_loss_value = -loss_alpha.item()
            alpha_value = alpha_var.item()
        else:
            # Skip alpha update, return previous values
            alpha_loss_value = 0.0
            alpha_value = self.alpha.item()
        
        return loss_pi.item(), alpha_loss_value, alpha_value
    
    def step_scheduler(self):
        """
        Update learning rate scheduler (call at end of each epoch)
        """
        if hasattr(self, 'scheduler_pi'):
            self.scheduler_pi.step()
    
    def set_epoch(self, epoch):
        """
        Set current epoch number for dynamic alpha update frequency
        """
        self.epoch_counter = epoch

    def get_optimal_alpha(self, X_batch, P_batch, Y_batch, Pi0_batch, current_delta, alpha_init=None):
        """
        Compute optimal dual variable Alpha and worst-case value using DR estimator
        """
        if self.forest_models is None:
            # Fallback to IPW if cross-fitting not initialized
            return super().get_optimal_alpha(X_batch, P_batch, Y_batch, Pi0_batch, current_delta, alpha_init)
        
        X = torch.FloatTensor(X_batch)
        P = torch.FloatTensor(P_batch).view(-1, 1)
        Y = torch.FloatTensor(Y_batch).view(-1, 1)
        
        with torch.no_grad():
            pi_x = self.policy(X)
        
        pi_x_np = pi_x.numpy().flatten()
        X_np = X.numpy()
        P_np = P.numpy().flatten()
        Y_np = Y.numpy().flatten()
        R_np = P_np * Y_np
        
        # Pre-compute cache for values that don't depend on alpha (optimization)
        # Cache propensity predictions and kernel weights for each fold
        cached_data = {}
        for fold_idx in range(self.n_folds):
            try:
                prop_model = self.prop_models[fold_idx]
                pi0_values = prop_model.predict(X_np, P_np)
                pi0_values = np.maximum(pi0_values, config.ETA)
                
                # Kernel weights (don't depend on alpha)
                u_vals = (P_np - pi_x_np) / self.h
                kernel_vals = np.exp(-0.5 * u_vals**2) / np.sqrt(2 * np.pi)
                kernel_weights = kernel_vals / (self.h * pi0_values)
                kernel_weights = np.clip(kernel_weights, 0, 20.0)
                S_ell_pi = np.mean(kernel_weights) + 1e-8
                
                cached_data[fold_idx] = {
                    'pi0_values': pi0_values,
                    'kernel_weights': kernel_weights,
                    'S_ell_pi': S_ell_pi
                }
            except:
                continue
        
        # Objective function for alpha optimization using DR estimator
        def objective(alpha_val):
            if alpha_val <= 1e-4:
                return 1e9
            
            # Compute Ŵ_T(π, α) using DR estimator across all folds
            # Use cached data to avoid recomputing alpha-independent values
            W_hat_values = []
            for fold_idx, cache in cached_data.items():
                try:
                    # Use cached values
                    kernel_weights = cache['kernel_weights']
                    S_ell_pi = cache['S_ell_pi']
                    
                    # Compute m_hat values (depend on alpha)
                    X_all = np.vstack([X_np, X_np])
                    P_all = np.concatenate([pi_x_np, P_np])
                    m_all_values = self._compute_m_hat_batch(X_all, P_all, alpha_val, fold_idx)
                    m_pi_values = m_all_values[:len(X_np)]
                    m_obs_values = m_all_values[len(X_np):]
                    
                    # DR estimator components
                    exp_terms = np.exp(-R_np / alpha_val)
                    ipw_correction = kernel_weights * (exp_terms - m_obs_values)
                    W_hat_ell = np.mean(m_pi_values) + np.mean(ipw_correction) / S_ell_pi
                    W_hat_values.append(W_hat_ell)
                except:
                    continue
            
            if len(W_hat_values) == 0:
                return 1e9
            
            W_hat_mean = np.mean(W_hat_values)
            log_W_hat = np.log(max(W_hat_mean, 1e-10))
            
            # Minimize: α log(Ŵ_T) + αδ
            return alpha_val * log_W_hat + alpha_val * current_delta
        
        # Search for optimal alpha
        if config.ALPHA_OPT_METHOD in ['bounded']:
            res = minimize_scalar(objective, bounds=config.ALPHA_OPT_BOUNDS, 
                                method=config.ALPHA_OPT_METHOD,
                                options={'xatol': config.ALPHA_OPT_TOL})
        else:
            # 'golden' and 'brent' methods require bracket parameter
            # If alpha_init is provided, use it to set a better bracket
            if alpha_init is not None and config.ALPHA_OPT_BOUNDS[0] < alpha_init < config.ALPHA_OPT_BOUNDS[1]:
                bracket_range = min(alpha_init - config.ALPHA_OPT_BOUNDS[0], 
                                   config.ALPHA_OPT_BOUNDS[1] - alpha_init) * 0.5
                bracket = (max(config.ALPHA_OPT_BOUNDS[0], alpha_init - bracket_range),
                          alpha_init,
                          min(config.ALPHA_OPT_BOUNDS[1], alpha_init + bracket_range))
            else:
                bracket = (config.ALPHA_OPT_BOUNDS[0], 
                          (config.ALPHA_OPT_BOUNDS[0] + config.ALPHA_OPT_BOUNDS[1]) / 2,
                          config.ALPHA_OPT_BOUNDS[1])
            res = minimize_scalar(objective, bracket=bracket, 
                                method=config.ALPHA_OPT_METHOD,
                                options={'xatol': config.ALPHA_OPT_TOL})
        alpha_star = res.x
        
        # Compute worst-case value: -α* log(Ŵ_T) - α*δ
        # Reuse cached data for efficiency
        W_hat_final_values = []
        for fold_idx, cache in cached_data.items():
            try:
                kernel_weights = cache['kernel_weights']
                S_ell_pi = cache['S_ell_pi']
                
                # Compute m_hat values with alpha_star
                X_all = np.vstack([X_np, X_np])
                P_all = np.concatenate([pi_x_np, P_np])
                m_all_values = self._compute_m_hat_batch(X_all, P_all, alpha_star, fold_idx)
                m_pi_values = m_all_values[:len(X_np)]
                m_obs_values = m_all_values[len(X_np):]
                
                # DR estimator
                exp_terms = np.exp(-R_np / alpha_star)
                ipw_correction = kernel_weights * (exp_terms - m_obs_values)
                W_hat_ell = np.mean(m_pi_values) + np.mean(ipw_correction) / S_ell_pi
                W_hat_final_values.append(W_hat_ell)
            except:
                continue
        
        if len(W_hat_final_values) == 0:
            Q_DRO = 0.0
        else:
            W_hat_final = np.mean(W_hat_final_values)
            Q_DRO = -alpha_star * np.log(max(W_hat_final, 1e-10)) - alpha_star * current_delta
        
        return alpha_star, Q_DRO

    def get_worst_case_weights(self, X_batch, P_batch, Y_batch, Pi0_batch, alpha_star):
        """
        Compute sample resampling weights under worst-case distribution using DR estimator
        """
        if self.forest_models is None:
            # Fallback to IPW if cross-fitting not initialized
            return super().get_worst_case_weights(X_batch, P_batch, Y_batch, Pi0_batch, alpha_star)
        
        X = torch.FloatTensor(X_batch)
        P = torch.FloatTensor(P_batch).view(-1, 1)
        Y = torch.FloatTensor(Y_batch).view(-1, 1)
        
        with torch.no_grad():
            pi_x = self.policy(X)
        
        pi_x_np = pi_x.numpy().flatten()
        X_np = X.numpy()
        P_np = P.numpy().flatten()
        Y_np = Y.numpy().flatten()
        R_np = P_np * Y_np
        
        # Compute weights using DR estimator across all folds
        # Weight for sample i: proportional to exp(-R_i / α*) with DR correction
        weights_list = []
        
        for fold_idx in range(self.n_folds):
            try:
                prop_model = self.prop_models[fold_idx]
                pi0_values = prop_model.predict(X_np, P_np)
                pi0_values = np.maximum(pi0_values, config.ETA)
                
                # Kernel weights
                u_vals = (P_np - pi_x_np) / self.h
                kernel_vals = np.exp(-0.5 * u_vals**2) / np.sqrt(2 * np.pi)
                kernel_weights = kernel_vals / (self.h * pi0_values)
                kernel_weights = np.clip(kernel_weights, 0, 20.0)
                
                # Compute m_hat for observed actions (optimized: use batch version instead of loop)
                m_obs_values = self._compute_m_hat_batch(X_np, P_np, alpha_star, fold_idx)
                
                # DR-corrected weights: C_t * (exp(-R/α*) - m_hat) + m_hat contribution
                exp_terms = np.exp(-R_np / alpha_star)
                dr_weights = kernel_weights * (exp_terms - m_obs_values) + m_obs_values
                weights_list.append(dr_weights)
            except:
                continue
        
        if len(weights_list) == 0:
            # Fallback to simple IPW
            return super().get_worst_case_weights(X_batch, P_batch, Y_batch, Pi0_batch, alpha_star)
        
        # Average weights across folds
        weights_avg = np.mean(weights_list, axis=0)
        
        # Ensure weights are non-negative (required for np.random.choice)
        # DR correction may produce negative values, so we clip them
        weights_avg = np.maximum(weights_avg, 0)
        
        # Check if all weights are zero or negative (should not happen, but safety check)
        weight_sum = np.sum(weights_avg)
        if weight_sum <= 1e-10:
            # Fallback: use uniform distribution if all weights are invalid
            weights_avg = np.ones(len(weights_avg)) / len(weights_avg)
        else:
            # Normalize to probability distribution
            weights_avg = weights_avg / weight_sum
        
        return weights_avg

class Ai2024Learner(BaseLearner):
    def __init__(self, policy_net, outcome_model, bandwidth):
        super().__init__(policy_net, bandwidth, delta=config.DELTA)
        self.m_model = outcome_model
        for param in self.m_model.parameters():
            param.requires_grad = False

    def train_step(self, X_batch, P_batch, Y_batch, Propensity_batch):
        """
        Double-Debiased (DD) estimator from Ai et al. (2024)
        Formula: (1/h) K((p_t - π_θ(x_t))/h) * (p_t y_t - m̂(p_t, x_t)) / π_0(p_t|x_t) + m̂(π_θ(x_t), x_t)
        """
        X = torch.FloatTensor(X_batch)
        P = torch.FloatTensor(P_batch).view(-1, 1)
        Y = torch.FloatTensor(Y_batch).view(-1, 1)
        Pi0 = torch.FloatTensor(Propensity_batch).view(-1, 1)
        R_obs = P * Y 

        pi_x = self.policy(X)
        u = (P - pi_x) / self.h
        kernel_val = self.gaussian_kernel(u)
        
        Pi0_safe = torch.clamp(Pi0, min=config.ETA)
        weights = kernel_val / (self.h * Pi0_safe)
        weights = torch.clamp(weights, max=5.0)
        
        # Compute outcome regression for observed actions: m̂(p_t, x_t)
        m_obs = self.m_model(X, P)
        # Compute outcome regression for policy actions: m̂(π_θ(x_t), x_t)
        m_pi = self.m_model(X, pi_x)
        
        # Double Robust Estimator:
        # IPW correction term: weights * (R_obs - m_obs)
        # Direct method term: m_pi
        dr_reward = weights * (R_obs - m_obs) + m_pi
        loss = -torch.mean(dr_reward)
        
        self.opt_pi.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=config.GRAD_CLIP_MAX_NORM)
        self.opt_pi.step()
        return loss.item(), 0.0, float('inf')

class Leung2025Learner(BaseLearner):
    """
    Leung et al. (2025) Algorithm 2: IPW-based DRO for Continuous Treatment
    
    This is an approximate implementation.
    Leung2025 requires f0(y|a,x) to be three-times differentiable w.r.t. a,
    which is violated in our binary outcome setting (Y ∈ {0,1}).
    
    Uses real reward: R = P * Y.
    This baseline is implemented for empirical comparison only.
    
    Uses IPW estimator (Eqn. 6b from Leung paper):
    W_N(π,α) = W̄_N(π,α) / S_N
    where W̄_N(π,α) = (1/N) Σ [K_h(π(X_i) - A_i) / f_0(A_i|X_i)] * exp(-R_i/α)
    and R_i = P_i * Y_i (real reward)
    """
    def __init__(self, policy_net, outcome_model, bandwidth, delta=config.DELTA):
        super().__init__(policy_net, bandwidth, delta)
        # outcome_model parameter kept for interface compatibility
        self.prob_model = outcome_model
        for param in self.prob_model.parameters():
            param.requires_grad = False
        
        # Alpha parameter for alternating optimization
        init_val = 1.0
        init_raw = np.log(np.exp(init_val) - 1.0)
        self.raw_alpha = torch.tensor(init_raw, requires_grad=True)
        self.opt_alpha = optim.Adam([self.raw_alpha], lr=config.ALPHA_OPTIMIZER_LR)

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.raw_alpha) + 1e-4

    def train_step(self, X_batch, P_batch, Y_batch, Propensity_batch):
        """
        Alternating optimization: Algorithm 2 from Leung et al. (2025)
        1. Update policy: π ← argmin W_N(π,α)
        2. Update alpha: α ← argmax -α log W_N(π,α) - αη
        
        Uses real reward: R = P * Y
        """
        X = torch.FloatTensor(X_batch)
        P = torch.FloatTensor(P_batch).view(-1, 1)
        Y = torch.FloatTensor(Y_batch).view(-1, 1)
        Pi0 = torch.FloatTensor(Propensity_batch).view(-1, 1)
        
        # 1. Update Policy (Minimization Step)
        curr_alpha = self.alpha.detach()
        pi_x = self.policy(X)
        
        # Compute real reward: R = P * Y
        R = P * Y
        
        # Kernel weights: K_h(π(X_i) - A_i) / (h * f_0(A_i|X_i))
        u = (P - pi_x) / self.h
        kernel_val = self.gaussian_kernel(u)
        Pi0_safe = torch.clamp(Pi0, min=config.ETA)
        C_t = kernel_val / (self.h * Pi0_safe)
        C_t = torch.clamp(C_t, max=20.0)
        S_N = torch.mean(C_t).detach() + 1e-8
        
        # W_N(π,α) = (1/N) Σ C_t * exp(-R/α) / S_N
        exp_terms = torch.exp(-R / curr_alpha)
        W_hat = torch.mean(C_t * exp_terms) / S_N
        
        # Minimize W_N(π,α) w.r.t. π
        loss_pi = W_hat
        
        self.opt_pi.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=config.GRAD_CLIP_MAX_NORM)
        self.opt_pi.step()
        
        # 2. Update Alpha (Maximization Step)
        with torch.no_grad():
            pi_x_const = self.policy(X)
            u = (P - pi_x_const) / self.h
            kernel_val = self.gaussian_kernel(u)
            C_t = kernel_val / (self.h * Pi0_safe)
            C_t = torch.clamp(C_t, max=20.0)
            S_N = torch.mean(C_t) + 1e-8
        
        # Compute real reward: R = P * Y
        R = P * Y
        
        alpha_var = self.alpha
        exp_terms = torch.exp(-R / alpha_var)
        W_hat_alpha = torch.mean(C_t * exp_terms) / (S_N + 1e-8)
        
        # Maximize: -α log W_N(π,α) - αδ
        # Minimize: α log W_N(π,α) + αδ
        log_W = torch.log(W_hat_alpha + 1e-10)
        loss_alpha = alpha_var * log_W + alpha_var * self.delta
        
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()
        
        return loss_pi.item(), loss_alpha.item(), curr_alpha.item()

    def get_optimal_alpha(self, X_batch, P_batch, Y_batch, Pi0_batch, current_delta, alpha_init=None):
        """
        Compute optimal alpha using IPW estimator
        Uses real reward: R = P * Y
        """
        X = torch.FloatTensor(X_batch)
        P = torch.FloatTensor(P_batch).view(-1, 1)
        Y = torch.FloatTensor(Y_batch).view(-1, 1)
        Pi0 = torch.FloatTensor(Pi0_batch).view(-1, 1)
        
        with torch.no_grad():
            pi_x = self.policy(X)
            
            # Compute real reward: R = P * Y
            R = P * Y
            
            # Kernel weights
            u = (P - pi_x) / self.h
            kernel_val = self.gaussian_kernel(u)
            Pi0_safe = torch.clamp(Pi0, min=config.ETA)
            C_t = kernel_val / (self.h * Pi0_safe)
            C_t = torch.clamp(C_t, max=100.0)
            S_N = torch.mean(C_t) + 1e-8
            
            def objective(alpha_val):
                if alpha_val <= 1e-4:
                    return 1e9
                
                exp_terms = torch.exp(-R / alpha_val)
                W_bar = torch.mean(C_t * exp_terms)
                W_hat = W_bar / S_N
                log_W_hat = torch.log(W_hat + 1e-10)
                
                # Paper requirement: max [-α log(Ŵ_N^h) - αη]
                # Using minimize_scalar to minimize α log(Ŵ_N^h) + αη is equivalent to maximizing -α log(Ŵ_N^h) - αη
                return alpha_val * log_W_hat.item() + alpha_val * current_delta
            
            # Search for optimal alpha
            if config.ALPHA_OPT_METHOD in ['bounded']:
                res = minimize_scalar(objective, bounds=config.ALPHA_OPT_BOUNDS, 
                                    method=config.ALPHA_OPT_METHOD,
                                    options={'xatol': config.ALPHA_OPT_TOL})
            else:
                # 'golden' and 'brent' methods require bracket parameter
                # If alpha_init is provided, use it to set a better bracket
                if alpha_init is not None and config.ALPHA_OPT_BOUNDS[0] < alpha_init < config.ALPHA_OPT_BOUNDS[1]:
                    bracket_range = min(alpha_init - config.ALPHA_OPT_BOUNDS[0], 
                                       config.ALPHA_OPT_BOUNDS[1] - alpha_init) * 0.5
                    bracket = (max(config.ALPHA_OPT_BOUNDS[0], alpha_init - bracket_range),
                              alpha_init,
                              min(config.ALPHA_OPT_BOUNDS[1], alpha_init + bracket_range))
                else:
                    bracket = (config.ALPHA_OPT_BOUNDS[0], 
                              (config.ALPHA_OPT_BOUNDS[0] + config.ALPHA_OPT_BOUNDS[1]) / 2,
                              config.ALPHA_OPT_BOUNDS[1])
                res = minimize_scalar(objective, bracket=bracket, 
                                    method=config.ALPHA_OPT_METHOD,
                                    options={'xatol': config.ALPHA_OPT_TOL})
            alpha_star = res.x
            
            # Compute worst-case value: -α* log(W_N) - α*δ
            exp_terms = torch.exp(-R / alpha_star)
            W_bar = torch.mean(C_t * exp_terms)
            W_hat = W_bar / S_N
            Q_DRO = -alpha_star * np.log(max(W_hat.item(), 1e-10)) - alpha_star * current_delta
            
            return alpha_star, Q_DRO
    
    def evaluate_policy(self, target_policy, X_batch, P_batch, Y_batch, Pi0_batch, delta_test):
        """
        Evaluate a given target policy using Leung2025's IPW-based DRO OPE (Algorithm 1)
        
        Parameters:
        -----------
        target_policy : torch.nn.Module
            The policy to evaluate (fixed, not trained)
        X_batch, P_batch, Y_batch, Pi0_batch : np.ndarray
            Evaluation data
        delta_test : float
            Test uncertainty radius δ
        
        Returns:
        --------
        R_delta : float
            Estimated distributionally robust value R_δ
        alpha_star : float
            Optimal dual variable α*
        """
        X = torch.FloatTensor(X_batch)
        P = torch.FloatTensor(P_batch).view(-1, 1)
        Y = torch.FloatTensor(Y_batch).view(-1, 1)
        Pi0 = torch.FloatTensor(Pi0_batch).view(-1, 1)
        
        with torch.no_grad():
            # Get target policy predictions
            pi_x = target_policy(X).numpy().flatten()
            
            # Compute real reward: R = P * Y
            R = P * Y
            
            # Kernel weights: K_h(π(X_i) - A_i) / (h * f_0(A_i|X_i))
            u = (P - torch.FloatTensor(pi_x).view(-1, 1)) / self.h
            kernel_val = self.gaussian_kernel(u)
            Pi0_safe = torch.clamp(Pi0, min=config.ETA)
            C_t = kernel_val / (self.h * Pi0_safe)
            C_t = torch.clamp(C_t, max=100.0)  # Upper bound limit
            S_N = torch.mean(C_t) + 1e-8
            
            # Objective function for alpha optimization
            # Paper requirement: max_{α≥0} [-α log(Ŵ_N^h) - αη]
            # Since minimize_scalar minimizes the objective function, to maximize the original function, we need to add a negative sign to the objective function
            def objective(alpha_val):
                if alpha_val <= 1e-4:
                    return 1e9
                
                # Improve numerical stability: limit exp input range to prevent overflow
                R_scaled = torch.clamp(R / alpha_val, min=-100, max=100)
                exp_terms = torch.exp(-R_scaled)
                W_bar = torch.mean(C_t * exp_terms)
                W_hat = W_bar / S_N
                log_W_hat = torch.log(W_hat + 1e-10)
                
                # Paper requirement: max [-α log(Ŵ_N^h) - αη]
                # Using minimize_scalar to minimize α log(Ŵ_N^h) + αη is equivalent to maximizing -α log(Ŵ_N^h) - αη
                return alpha_val * log_W_hat.item() + alpha_val * delta_test
            
            # Find optimal alpha
            res = minimize_scalar(objective, bounds=config.ALPHA_OPT_BOUNDS,
                                 method=config.ALPHA_OPT_METHOD,
                                 options={'xatol': config.ALPHA_OPT_TOL})
            alpha_star = res.x
            
            # Compute R_δ = -α* log(W_N) - α*δ
            # Improve numerical stability: limit exp input range to prevent overflow
            R_scaled = torch.clamp(R / alpha_star, min=-100, max=100)
            exp_terms = torch.exp(-R_scaled)
            W_bar = torch.mean(C_t * exp_terms)
            W_hat = W_bar / S_N
            R_delta = -alpha_star * np.log(max(W_hat.item(), 1e-10)) - alpha_star * delta_test
            
            return R_delta, alpha_star

    def get_worst_case_weights(self, X_batch, P_batch, Y_batch, Pi0_batch, alpha_star):
        """
        Compute sample resampling weights under worst-case distribution using real reward
        Uses real reward: R = P * Y
        """
        X = torch.FloatTensor(X_batch)
        P = torch.FloatTensor(P_batch).view(-1, 1)
        Y = torch.FloatTensor(Y_batch).view(-1, 1)
        Pi0 = torch.FloatTensor(Pi0_batch).view(-1, 1)
        
        with torch.no_grad():
            pi_x = self.policy(X)
            
            # Compute real reward: R = P * Y
            R = P * Y
            
            # Kernel weights
            u = (P - pi_x) / self.h
            kernel_val = self.gaussian_kernel(u)
            Pi0_safe = torch.clamp(Pi0, min=config.ETA)
            C_t = kernel_val / (self.h * Pi0_safe)
            C_t = torch.clamp(C_t, max=20.0)
            
            # Weights w_i \propto C_t * exp(-R/α*)
            log_weights = torch.log(C_t + 1e-10) - (R / alpha_star)
            weights = torch.softmax(log_weights, dim=0).flatten().numpy()
            
            return weights


class OPELearner:
    """
    Localized Doubly Robust Distributionally Robust Policy Evaluation for Continuous Pricing
    Implements Algorithm 1 from offline_pricing.tex
    
    This class evaluates a given target policy (not learning it).
    """
    def __init__(self, target_policy, bandwidth, delta, X_train, P_train, Y_train, n_folds=None):
        """
        Parameters:
        -----------
        target_policy : torch.nn.Module
            The policy to evaluate (fixed, not trained)
        bandwidth : float
            Kernel bandwidth h
        delta : float
            Uncertainty radius δ
        X_train, P_train, Y_train : np.ndarray
            Training data
        n_folds : int, optional
            Number of cross-fitting folds (default: config.OPE_N_FOLDS)
        """
        self.target_policy = target_policy
        self.h = bandwidth
        self.delta = delta
        self.n_folds = n_folds if n_folds is not None else config.OPE_N_FOLDS
        
        # Store training data
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.P_train = np.asarray(P_train, dtype=np.float64)
        self.Y_train = np.asarray(Y_train, dtype=np.float64)
        self.R_train = self.P_train * self.Y_train
        
        # Cache for performance optimization (cleared when needed)
        self._cache_pi_x = {}  # Cache target policy predictions: key = fold_idx
        self._cache_propensity = {}  # Cache propensity scores: key = fold_idx
        
        # Setup cross-fitting
        self._setup_cross_fitting()
    
    def _setup_cross_fitting(self):
        """Phase 1: Cross-fitting setup (Algorithm 1, lines 2-8)"""
        n_samples = len(self.X_train)
        
        # Create folds
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=config.SEED)
        self.fold_indices = []
        self.prop_models = []
        self.forest_models = []
        self.alpha_init_values = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
            self.fold_indices.append({
                'train': train_idx,
                'val': val_idx
            })
            
            # Train propensity model on out-of-fold data (line 4)
            prop_model = PropensityEstimator()
            prop_model.fit(self.X_train[train_idx], self.P_train[train_idx])
            self.prop_models.append(prop_model)
            
            # Split out-of-fold data into two halves: J1, J2 (line 5)
            n_train_fold = len(train_idx)
            mid_point = n_train_fold // 2
            J1_idx = train_idx[:mid_point]
            J2_idx = train_idx[mid_point:]
            
            # Initial estimate of alpha using J1 (line 6)
            alpha_init = self._initial_estimate(J1_idx)
            self.alpha_init_values.append(alpha_init)
            
            # Train outcome regression m_j at alpha_init using J2 (line 7)
            forest = self._train_outcome_regression(J2_idx, alpha_init, fold_idx)
            self.forest_models.append(forest)
        
        print(f"OPE Phase 1 complete: {self.n_folds} folds, {n_samples} samples")
    
    def _initial_estimate(self, data_idx):
        """
        InitialEstimate: Compute initial alpha using IPW method (line 6)
        Uses IPW estimator to get preliminary consistent estimate
        """
        X_data = self.X_train[data_idx]
        P_data = self.P_train[data_idx]
        Y_data = self.Y_train[data_idx]
        R_data = self.R_train[data_idx]
        
        # Get target policy predictions
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_data)
            pi_x = self.target_policy(X_tensor).numpy().flatten()
        
        # Compute propensity scores
        prop_model = PropensityEstimator()
        prop_model.fit(X_data, P_data)
        pi0_values = prop_model.predict(X_data, P_data)
        pi0_values = np.maximum(pi0_values, config.ETA)
        
        # Kernel weights
        u_vals = (P_data - pi_x) / self.h
        kernel_vals = np.exp(-0.5 * u_vals**2) / np.sqrt(2 * np.pi)
        C_t = kernel_vals / (self.h * pi0_values)
        C_t = np.clip(C_t, 0, 100.0)  # Upper bound limit
        S_N = np.mean(C_t) + 1e-8
        
        # Objective function for IPW-based initial estimate
        def objective(alpha_val):
            if alpha_val <= 1e-4:
                return 1e9
            exp_terms = np.exp(-R_data / alpha_val)
            W_bar = np.mean(C_t * exp_terms)
            W_hat = W_bar / S_N
            log_W_hat = np.log(max(W_hat, 1e-10))
            return alpha_val * log_W_hat + alpha_val * self.delta
        
        # Find optimal alpha
        res = minimize_scalar(objective, bounds=config.ALPHA_OPT_BOUNDS,
                             method=config.ALPHA_OPT_METHOD,
                             options={'xatol': config.ALPHA_OPT_TOL})
        
        return res.x
    
    def _train_outcome_regression(self, data_idx, alpha_init, fold_idx):
        """
        Train outcome regression m_j at alpha_init (line 7)
        Uses 25 trees RegressionForest to fit weights (matching kallus22)
        """
        X_data = self.X_train[data_idx]
        P_data = self.P_train[data_idx]
        R_data = self.R_train[data_idx]
        
        # Features: (X, P), Target: exp(-R/alpha_init)
        X_P_data = np.column_stack([X_data, P_data])
        target = np.exp(-R_data / alpha_init)
        
        # Use 24 trees (close to kallus22's 25, 24=4×6 divisible by subforest_size=4)
        forest = RegressionForest(
            n_estimators=config.OPE_N_ESTIMATORS,  # 24 trees
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=config.SEED + fold_idx
        )
        
        forest.fit(X_P_data, target)
        return forest
    
    def _get_forest_weights(self, x, p, fold_idx):
        """
        Get forest-induced weights ω_t(x, p) for query point (x, p)
        Single query point version (backward compatible)
        """
        # Use batch version for efficiency
        weights_matrix = self._get_forest_weights_batch(x, p, fold_idx)
        return weights_matrix[0] if weights_matrix.shape[0] == 1 else weights_matrix.flatten()
    
    def _get_forest_weights_batch(self, X_batch, P_batch, fold_idx):
        """
        Batch version: Compute forest-induced weights for multiple query points
        Returns [n_query, n_train] weight matrix
        Optimized version similar to CDROLearner._get_forest_weights_matrix
        """
        forest = self.forest_models[fold_idx]
        train_idx = self.fold_indices[fold_idx]['train']
        
        X_train_fold = self.X_train[train_idx]
        P_train_fold = self.P_train[train_idx]
        # Pre-allocate to avoid repeated column_stack
        X_P_train = np.empty((len(train_idx), X_train_fold.shape[1] + 1), dtype=X_train_fold.dtype)
        X_P_train[:, :-1] = X_train_fold
        X_P_train[:, -1] = P_train_fold
        
        # Ensure proper shapes
        if not isinstance(X_batch, np.ndarray):
            X_batch = np.asarray(X_batch)
        if not isinstance(P_batch, np.ndarray):
            P_batch = np.asarray(P_batch)
        
        if X_batch.ndim == 1:
            X_batch = X_batch.reshape(1, -1)
        if P_batch.ndim == 0:
            P_batch = P_batch.reshape(1)
        elif P_batch.ndim == 1 and len(P_batch) == 1 and X_batch.shape[0] > 1:
            P_batch = np.repeat(P_batch, X_batch.shape[0])
        
        n_query = X_batch.shape[0]
        n_train = len(train_idx)
        # Pre-allocate X_P_query
        X_P_query = np.empty((n_query, X_batch.shape[1] + 1), dtype=X_batch.dtype)
        X_P_query[:, :-1] = X_batch
        X_P_query[:, -1] = P_batch.reshape(-1)
        
        # Initialize weight matrix: [n_query, n_train]
        weights_matrix = np.zeros((n_query, n_train), dtype=np.float64)
        
        # Get trees
        if hasattr(forest, 'estimators_'):
            trees = forest.estimators_
        elif hasattr(forest, 'grf_'):
            trees = forest.grf_.estimators_ if hasattr(forest.grf_, 'estimators_') else []
        else:
            weights_matrix[:] = 1.0 / n_train
            return weights_matrix
        
        n_trees = len(trees) if len(trees) > 0 else 1
        
        # Vectorized: process all trees efficiently
        for tree in trees:
            train_leaves = tree.apply(X_P_train)  # [n_train]
            query_leaves = tree.apply(X_P_query)  # [n_query]
            
            # Vectorized: compare all query points with all training samples
            matches = (train_leaves[:, None] == query_leaves[None, :])  # [n_train, n_query]
            
            # Compute leaf sizes for each query point
            leaf_sizes = np.sum(matches, axis=0, dtype=np.float64)  # [n_query]
            
            # Compute weight contributions
            leaf_sizes_safe = np.maximum(leaf_sizes, 1.0)  # Avoid division by zero
            weight_contributions = matches.astype(np.float64)  # [n_train, n_query]
            weight_contributions /= leaf_sizes_safe[None, :]  # In-place division
            
            # Accumulate weights: transpose to get [n_query, n_train]
            weights_matrix += weight_contributions.T  # [n_query, n_train]
        
        # Average over trees
        if n_trees > 0:
            weights_matrix /= n_trees
        else:
            weights_matrix[:] = 1.0 / n_train
        
        return weights_matrix
    
    def _compute_m_hat(self, x, p, alpha, fold_idx):
        """
        Compute m̂_j^(ℓ)(x, p; α) = Σ_{t ∈ I_ℓ^c} ω_t^(ℓ)(x, p) * (p_t y_t)^j * exp(-p_t y_t / α)
        For j=0,1
        Single query point version (backward compatible)
        """
        # Use batch version for efficiency
        m_0_batch, m_1_batch = self._compute_m_hat_batch(x, p, alpha, fold_idx)
        return m_0_batch[0], m_1_batch[0]
    
    def _compute_m_hat_batch(self, X_batch, P_batch, alpha, fold_idx):
        """
        Batch version: Compute m̂_j^(ℓ)(x, p; α) for all query points at once
        Returns [n_query] arrays of m_0 and m_1 values
        
        Note: Forest weights ω_t are trained at alpha_init (fixed), but we use current alpha for exp terms.
        """
        train_idx = self.fold_indices[fold_idx]['train']
        n_train = len(train_idx)
        
        # Pre-compute exp terms (same for all query points)
        R_train_fold = self.R_train[train_idx]
        exp_terms = np.exp(-R_train_fold / alpha)  # [n_train]
        R_exp_terms = R_train_fold * exp_terms  # [n_train] for m_1
        
        # Batch compute forest weights for all query points
        weights_matrix = self._get_forest_weights_batch(X_batch, P_batch, fold_idx)  # [n_query, n_train]
        
        # Vectorized batch computation: [n_query] = sum over train samples
        m_0_values = np.sum(weights_matrix * exp_terms[None, :], axis=1)  # [n_query]
        m_1_values = np.sum(weights_matrix * R_exp_terms[None, :], axis=1)  # [n_query]
        
        return m_0_values, m_1_values
    
    def _compute_W_hat(self, alpha, fold_idx):
        """
        Compute Ŵ_j^(ℓ)(α) for fold ℓ (line 9-10 of Algorithm 1)
        Optimized version using vectorized operations
        """
        val_idx = self.fold_indices[fold_idx]['val']
        X_val = self.X_train[val_idx]
        P_val = self.P_train[val_idx]
        Y_val = self.Y_train[val_idx]
        R_val = self.R_train[val_idx]
        n_val = len(val_idx)
        
        # Get target policy predictions (batch) - with caching
        # Cache key is fold_idx since X_val is fixed for each fold
        if fold_idx not in self._cache_pi_x:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_val)
                pi_x = self.target_policy(X_tensor).numpy().flatten()
            self._cache_pi_x[fold_idx] = pi_x
        else:
            pi_x = self._cache_pi_x[fold_idx]
        
        # Cache propensity scores (same fold always has same X_val, P_val)
        prop_model = self.prop_models[fold_idx]
        if fold_idx not in self._cache_propensity:
            pi0_values = prop_model.predict(X_val, P_val)
            pi0_values = np.maximum(pi0_values, config.ETA)
            self._cache_propensity[fold_idx] = pi0_values
        else:
            pi0_values = self._cache_propensity[fold_idx]
        
        # Kernel weights (vectorized)
        u_vals = (P_val - pi_x) / self.h
        kernel_vals = np.exp(-0.5 * u_vals**2) / np.sqrt(2 * np.pi)
        kernel_weights = kernel_vals / (self.h * pi0_values)
        kernel_weights = np.clip(kernel_weights, 0, 100.0)
        
        # Compute m_hat at alpha_init for all points (batch computation)
        alpha_init = self.alpha_init_values[fold_idx]
        
        # Batch compute m_hat for observed points (x_t, p_t)
        m_0_obs_init, m_1_obs_init = self._compute_m_hat_batch(X_val, P_val, alpha_init, fold_idx)
        
        # Batch compute m_hat for policy points (x_t, π(x_t))
        m_0_pi_init, m_1_pi_init = self._compute_m_hat_batch(X_val, pi_x, alpha_init, fold_idx)
        
        # IPW correction terms - use current alpha (not alpha_init) for exp (vectorized)
        # Improve numerical stability: limit exp input range to prevent overflow
        R_scaled = np.clip(R_val / alpha, -100, 100)  # Prevent exp overflow
        exp_term_0 = np.exp(-R_scaled)  # [n_val]
        exp_term_1 = R_val * exp_term_0  # [n_val]
        
        # Vectorized W_0 and W_1 computation
        # W_0 term: m_0_pi(alpha_init) + (K/h*π0) * (exp(-r/α) - m_0_obs(alpha_init))
        W_0_terms = m_0_pi_init + kernel_weights * (exp_term_0 - m_0_obs_init)  # [n_val]
        
        # W_1 term: m_1_pi(alpha_init) + (K/h*π0) * (r*exp(-r/α) - m_1_obs(alpha_init))
        W_1_terms = m_1_pi_init + kernel_weights * (exp_term_1 - m_1_obs_init)  # [n_val]
        
        W_0 = np.mean(W_0_terms)
        W_1 = np.mean(W_1_terms)
        
        return W_0, W_1
    
    def evaluate(self, delta_test=None):
        """
        Evaluate target policy: Compute R̂_δ (Algorithm 1, lines 9-11)
        
        Parameters:
        -----------
        delta_test : float, optional
            Test delta (default: self.delta)
        
        Returns:
        --------
        R_delta : float
            Estimated distributionally robust value R̂_δ
        alpha_star : float
            Optimal dual variable α̂
        """
        if delta_test is None:
            delta_test = self.delta
        
        # Initialize alpha (line 9)
        alpha_init = np.mean(self.alpha_init_values)
        
        # Solve moment equation (line 10)
        def moment_equation(alpha_val):
            if alpha_val <= 1e-4:
                return 1e9
            
            # Compute W_0 and W_1 across all folds (vectorized)
            # Pre-allocate arrays for better performance
            W_0_all = np.empty(self.n_folds, dtype=np.float64)
            W_1_all = np.empty(self.n_folds, dtype=np.float64)
            
            for fold_idx in range(self.n_folds):
                W_0, W_1 = self._compute_W_hat(alpha_val, fold_idx)
                W_0_all[fold_idx] = W_0
                W_1_all[fold_idx] = W_1
            
            W_0_mean = np.mean(W_0_all)
            W_1_mean = np.mean(W_1_all)
            
            # Moment equation: -log(W_0) - W_1/(α*W_0) - δ = 0
            if W_0_mean <= 1e-10:
                return 1e9
            
            moment = -np.log(max(W_0_mean, 1e-10)) - W_1_mean / (alpha_val * W_0_mean) - delta_test
            return moment
        
        # Solve using root finding
        try:
            # Dynamically adjust bracket based on alpha_init
            alpha_init_mean = np.mean(self.alpha_init_values)
            bracket_low = max(config.ALPHA_OPT_BOUNDS[0], alpha_init_mean * 0.1)
            bracket_high = min(config.ALPHA_OPT_BOUNDS[1], alpha_init_mean * 10.0)
            
            res = root_scalar(moment_equation, bracket=[bracket_low, bracket_high],
                             method='brentq', xtol=1e-5)  # Root finding tolerance
            alpha_star = res.root
        except:
            # Fallback to minimization
            def objective(alpha_val):
                moment = moment_equation(alpha_val)
                return abs(moment)
            
            res = minimize_scalar(objective, bounds=config.ALPHA_OPT_BOUNDS,
                                 method=config.ALPHA_OPT_METHOD,
                                 options={'xatol': config.ALPHA_OPT_TOL})
            alpha_star = res.x
        
        # Compute R_δ (line 11) - optimized with pre-allocated array
        W_0_all = np.empty(self.n_folds, dtype=np.float64)
        for fold_idx in range(self.n_folds):
            W_0, _ = self._compute_W_hat(alpha_star, fold_idx)
            W_0_all[fold_idx] = W_0
        
        W_0_mean = np.mean(W_0_all)
        R_delta = -alpha_star * np.log(max(W_0_mean, 1e-10)) - alpha_star * delta_test
        
        return R_delta, alpha_star