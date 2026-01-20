# estimators.py
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import norm
from . import config
import torch
import torch.nn as nn
import torch.optim as optim

class PropensityEstimator:
    def __init__(self, oracle_params=None):
        self.oracle_params = oracle_params
        
        if self.oracle_params is None:
            # Use svd solver
            self.model = Ridge(alpha=1.0, solver='svd') 
            self.std = 1.0
            self.learned_beta = None
            self.learned_bias = None
        else:
            self.model = None
            self.std = self.oracle_params['sigma']

    def fit(self, X, P):
        if self.oracle_params is not None:
            return self

        # 1. Data preparation
        X = np.ascontiguousarray(X, dtype=np.float64)
        P = np.ascontiguousarray(P, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        P = np.nan_to_num(P, nan=config.PRICE_MIN)
        
        try:
            # 2. Train model
            self.model.fit(X, P)
            
            # Extract parameters
            self.learned_beta = self.model.coef_
            self.learned_bias = self.model.intercept_
            
            if self.learned_beta.ndim > 1:
                self.learned_beta = self.learned_beta.flatten()
                
            P_pred = np.dot(X, self.learned_beta) + self.learned_bias
            self.std = np.std(P - P_pred)
            
        except Exception as e:
            print(f"Warning: Ridge fit failed ({e}), using fallback.")
            self.std = 1.0
            self.learned_beta = np.zeros(X.shape[1])
            self.learned_bias = float(np.mean(P))
            
        self.std = max(self.std, 0.1) 
        return self

    def predict(self, X, P):
        X = np.ascontiguousarray(X, dtype=np.float64)
        P = np.ascontiguousarray(P, dtype=np.float64)
        
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        X_safe = np.clip(X, -20.0, 20.0)

        if self.oracle_params is not None:
            beta = np.array(self.oracle_params['beta'], dtype=np.float64)
            bias = float(self.oracle_params['bias'])
            sigma = float(self.oracle_params['sigma'])
        else:
            if self.learned_beta is None:
                beta = np.zeros(X.shape[1])
                bias = 5.0
                sigma = 1.0
            else:
                beta = self.learned_beta
                bias = self.learned_bias
                sigma = self.std
        
        mu = np.dot(X_safe, beta) + bias
        mu = np.nan_to_num(mu, nan=5.0, posinf=10.0, neginf=1.0)
        sigma = max(sigma, 0.1)
        
        pdf = norm.pdf(P, loc=mu, scale=sigma)
        pdf = np.nan_to_num(pdf, nan=0.0)
        
        return np.maximum(pdf, config.ETA)

def silverman_bandwidth(P):
    P = np.array(P, dtype=np.float64)
    P = P[np.isfinite(P)]
    n = len(P)
    if n < 2: return 0.5
    std_p = np.std(P)
    if std_p < 1e-6: std_p = 1.0
    h = 1.06 * std_p * (n ** (-1/4.9))  #  OPL  4.9
    return float(h)

class DifferentiableOutcomeEstimator(nn.Module):
    """
    Linear Outcome Model (Linear Regression)
    For Ai et al. (2024).
    Fitting target: E[Revenue | X, T]
    Model structure: Linear(X_dim + 1 -> 1)
    
    Note: If the true environment is nonlinear, using this model will cause bias,
    but this is exactly for testing Double Robust methods under nuisance model bias.
    """
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        # Input dimension: X (dim) + T (1) = dim + 1
        # Output dimension: Revenue (1)
        # Even if hidden_dim is passed, it is not used, forced to single-layer linear
        self.net = nn.Linear(input_dim + 1, 1)
        
        # Linear model is relatively simple, can use slightly larger learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, t):
        """
        x: [batch, dim]
        t: [batch, 1]
        """
        # Concatenate X and T
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

    def fit(self, X_np, P_np, Y_np, epochs=200, batch_size=256):
        self.train()
        X = torch.FloatTensor(X_np)
        P = torch.FloatTensor(P_np).view(-1, 1)
        Y_binary = torch.FloatTensor(Y_np).view(-1, 1)
        Target = P * Y_binary 
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        for epoch in range(epochs):
            # The shuffle here depends on numpy's random seed, which is fixed in run_main
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                idx = indices[start_idx : start_idx + batch_size]
                
                batch_X = X[idx]
                batch_P = P[idx]
                batch_Target = Target[idx]
                
                self.optimizer.zero_grad()
                pred_revenue = self.forward(batch_X, batch_P)
                loss = self.loss_fn(pred_revenue, batch_Target)
                loss.backward()
                self.optimizer.step()
                
        self.eval()
        return self

class ProbabilityOutcomeEstimator(nn.Module):
    """
    Probability Outcome Model for Leung2025 baseline
    Estimates E[Y|X,P] (purchase probability) instead of E[Revenue|X,P]
    Uses logistic regression or neural network with sigmoid output
    """
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        # Input dimension: X (dim) + P (1) = dim + 1
        # Output dimension: Probability (1), with sigmoid activation
        if hidden_dim is None:
            # Simple linear model with sigmoid
            self.net = nn.Sequential(
                nn.Linear(input_dim + 1, 1),
                nn.Sigmoid()
            )
        else:
            # Two-layer neural network
            self.net = nn.Sequential(
                nn.Linear(input_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        self.loss_fn = nn.BCELoss()

    def forward(self, x, p):
        """
        x: [batch, dim]
        p: [batch, 1]
        Returns: [batch, 1] probability in [0, 1]
        """
        # Concatenate X and P
        inp = torch.cat([x, p], dim=1)
        return self.net(inp)

    def fit(self, X_np, P_np, Y_np, epochs=200, batch_size=256):
        """
        Fit the model to predict E[Y|X,P] where Y ∈ {0,1}
        """
        self.train()
        X = torch.FloatTensor(X_np)
        P = torch.FloatTensor(P_np).view(-1, 1)
        Y_binary = torch.FloatTensor(Y_np).view(-1, 1)
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                idx = indices[start_idx : start_idx + batch_size]
                
                batch_X = X[idx]
                batch_P = P[idx]
                batch_Y = Y_binary[idx]
                
                self.optimizer.zero_grad()
                pred_prob = self.forward(batch_X, batch_P)
                loss = self.loss_fn(pred_prob, batch_Y)
                loss.backward()
                self.optimizer.step()
        
        self.eval()
        return self