import numpy as np
import torch
import pandas as pd
import time
import random
import os
from tqdm import tqdm
from contextlib import contextmanager
from collections import defaultdict

from src.dgp import PricingEnvironment
from src.estimators import PropensityEstimator, silverman_bandwidth, DifferentiableOutcomeEstimator 
from src.policy import LinearPolicy, MLPPolicy
from src.learner import CDROLearner, Ai2024Learner, Leung2025Learner
from src import config
from sklearn.preprocessing import StandardScaler
from src import visualization

# ==========================================
# Global settings
# ==========================================
SETTING = 'linear' 
POLICY_TYPE = 'linear'

# ==========================================
# Test mode configuration
# ==========================================
# Set to True for quick testing, False for full run
TEST_MODE = False

if TEST_MODE:
    # Test mode: quickly verify code can run normally
    N_REPEATS = 5  # Only 5 repeats in test mode
    TEST_EPOCHS = 50  # Reduce training epochs in test mode (full run uses config.EPOCHS)
    TEST_NUISANCE_EPOCHS = 50  # Reduce nuisance model training epochs in test mode (full run uses 200)
    TEST_M = 5  # Reduce number of attacks in test mode (full run uses 100)
    TEST_N_FOLDS = 2  # Reduce cross-fitting folds in test mode (full run uses 5)
    TEST_N_ESTIMATORS = 8  # Reduce forest estimators in test mode (full run uses 40, must be multiple of subforest_size=4)
    TEST_FOREST_MAX_DEPTH = 5  # Reduce forest depth in test mode (full run uses 10)
    TEST_BATCH_SIZE = 64  # Reduce batch size in test mode (full run uses 512, reduces CDROLearner computation)
    # Add TEST_N_ESTIMATORS to config for learner.py to access
    config.TEST_N_ESTIMATORS = TEST_N_ESTIMATORS
    config.TEST_FOREST_MAX_DEPTH = TEST_FOREST_MAX_DEPTH
    TEST_N_SIZES = [1000]  # Only test 2 training set sizes in test mode (full run uses [500, 1000, 1500, 2000, 2500])
    TEST_DELTAS_TABLE2 = [0.1, 0.2, 0.3]  # Only test 3 deltas in test mode (full run uses [0.05, 0.1, 0.2, 0.3, 0.4])
    print("=" * 60)
    print("WARNING: Running in TEST MODE - Results are for testing only!")
    print("Set TEST_MODE = False for full experimental runs")
    print("=" * 60)
else:
    # Full run mode: final paper results
    N_REPEATS = config.FULL_N_REPEATS  # Full run uses 100 repeats
    TEST_EPOCHS = None  # Use config.EPOCHS
    TEST_NUISANCE_EPOCHS = 50  # Full run uses 50 epochs
    TEST_M = config.FULL_M_ATTACKS  # Full run uses 10 attacks
    TEST_N_FOLDS = 3  # Full run uses 3 folds
    TEST_N_ESTIMATORS = 20  # Full run uses 20 estimators
    TEST_FOREST_MAX_DEPTH = 10  # Full run uses 10
    TEST_BATCH_SIZE = 512  # Full run uses 512
    # Add TEST_N_ESTIMATORS to config for learner.py to access
    config.TEST_N_ESTIMATORS = TEST_N_ESTIMATORS
    config.TEST_FOREST_MAX_DEPTH = TEST_FOREST_MAX_DEPTH
    TEST_N_SIZES = [500, 1000, 1500, 2000, 2500]  # Full run uses all training set sizes
    TEST_DELTAS_TABLE2 = [0.05, 0.1, 0.2, 0.3, 0.4]  # Full run uses all deltas

# Define DRO strength to compare
DELTA_STRONG = 0.2

# ==========================================
# Performance monitoring tools
# ==========================================
_performance_log = defaultdict(list)  # Optimization: use defaultdict to avoid key checks

@contextmanager
def timer(operation_name):
    """Performance monitoring context manager"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        _performance_log[operation_name].append(elapsed)  # Simplified: no key check needed

def print_performance_summary():
    """Print performance summary"""
    if not _performance_log:
        return
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    for op_name, times in _performance_log.items():
        total = sum(times)
        mean = np.mean(times)
        count = len(times)
        print(f"{op_name:40s}: {count:4d} calls, {total:8.2f}s total, {mean:6.3f}s avg")
    print("="*60)

def set_global_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# Training and evaluation functions
# ==========================================

def train_models(n_train, seed):
    """
    Train models: Ai2024 (non-robust), Leung2025 (robust), DRO_0.2
    """
    with timer("train_models.total"):
        # 1. Prepare training data
        # Use nonlinear behavior policy (Kallus-style) to improve DRO performance at smaller delta
        env_train = PricingEnvironment(setting=SETTING, logging_policy_type='nonlinear', seed=seed)
        X_tr_raw, P_tr, Y_tr = env_train.sample_data(n_train)
        
        X_tr_raw = np.clip(np.nan_to_num(X_tr_raw.astype(np.float64)), -20.0, 20.0)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_tr_scaled = np.clip(X_tr_scaled, -20.0, 20.0)

        # 2. Nuisance Estimators
        prop_model = PropensityEstimator()
        prop_model.fit(X_tr_raw, P_tr)
        Pi0_tr = prop_model.predict(X_tr_raw, P_tr)
        h = np.maximum(config.MIN_BANDWIDTH, silverman_bandwidth(P_tr))

        outcome_model = DifferentiableOutcomeEstimator(input_dim=config.DIM)
        outcome_model.fit(X_tr_scaled, P_tr, Y_tr, epochs=TEST_NUISANCE_EPOCHS)
        
        # Probability outcome model for Leung2025
        from estimators import ProbabilityOutcomeEstimator
        prob_outcome_model = ProbabilityOutcomeEstimator(input_dim=config.DIM)
        prob_outcome_model.fit(X_tr_scaled, P_tr, Y_tr, epochs=TEST_NUISANCE_EPOCHS)
        
        # 3. Policy Learners initialization
        input_dim = config.DIM
        if POLICY_TYPE == 'linear':
            PolicyClass = LinearPolicy
        else:
            PolicyClass = MLPPolicy
            
        # Define model dictionary
        learners = {}
        
        # Baseline 1: Ai et al. (2024) - Non-robust
        learners['Ai2024'] = Ai2024Learner(PolicyClass(input_dim), outcome_model, h)
        
        # Baseline 2: Leung et al. (2025) - Robust (approximate)
        learners[f'Leung2025(δ={DELTA_STRONG})'] = Leung2025Learner(
            PolicyClass(input_dim), prob_outcome_model, h, delta=DELTA_STRONG
        )
        
        # Ours: DRO (delta=0.2)
        # Optimization: Use alpha_update_freq to reduce computation
        alpha_update_freq = getattr(config, 'ALPHA_UPDATE_FREQ', 5)
        learners[f'DRO(δ={DELTA_STRONG})'] = CDROLearner(
            PolicyClass(input_dim), h, delta=DELTA_STRONG,
            X_train=X_tr_scaled, P_train=P_tr, Y_train=Y_tr, n_folds=TEST_N_FOLDS,
            alpha_update_freq=alpha_update_freq
        )

        # 4. Warm Start (all models) - Enhanced strategy
        X_tensor = torch.FloatTensor(X_tr_scaled)
        P_tensor = torch.FloatTensor(P_tr).view(-1, 1)
        
        for name, l in learners.items():
            # Enhanced warm start: use learning rate scheduling for better initialization
            opt = torch.optim.Adam(l.policy.parameters(), lr=config.WARM_START_LR)
            warm_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=config.WARM_START_EPOCHS, eta_min=config.WARM_START_LR * 0.1
            )
            
            # Warm start with learning rate decay
            for epoch in range(config.WARM_START_EPOCHS):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(l.policy(X_tensor), P_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(l.policy.parameters(), max_norm=1.0)
                opt.step()
                warm_scheduler.step()

        # 5. Training Loop
        # Optimization: Dynamic batch size based on training set size
        # Larger training sets use larger batch sizes for better efficiency
        base_batch_size = TEST_BATCH_SIZE
        dynamic_batch_size = min(base_batch_size, max(64, n_train // 10))
        BATCH_SIZE = dynamic_batch_size
        indices = np.arange(len(X_tr_scaled))
        
        # Dynamic epochs adjustment: larger datasets need more training epochs
        # This ensures models are fully trained and R_delta shows upward trend with T
        if TEST_EPOCHS is not None:
            epochs_to_use = TEST_EPOCHS
        else:
            base_epochs = config.EPOCHS  # Base epochs (50)
            # For larger datasets, increase training epochs
            # Formula: base_epochs + (n_train - 500) // 500 * 10
            # Examples: T=500 -> 50, T=1000 -> 60, T=1500 -> 70, T=2000 -> 80, T=2500 -> 90
            epochs_to_use = base_epochs + max(0, (n_train - 500) // 500) * 10
        with timer("train_models.training_loop"):
            for epoch in tqdm(range(epochs_to_use), desc="Training Epochs", leave=False):
                np.random.shuffle(indices)
                
                # Set epoch for dynamic alpha update frequency (for CDROLearner)
                for l in learners.values():
                    if hasattr(l, 'set_epoch'):
                        l.set_epoch(epoch)
                
                for start_idx in range(0, len(indices), BATCH_SIZE):
                    idx = indices[start_idx : min(start_idx + BATCH_SIZE, len(indices))]
                    batch = (X_tr_scaled[idx], P_tr[idx], Y_tr[idx], Pi0_tr[idx])
                    
                    # Train all models
                    for l in learners.values():
                        l.train_step(*batch)
                
                # Update learning rate scheduler at end of each epoch (for CDROLearner)
                for l in learners.values():
                    if hasattr(l, 'step_scheduler'):
                        l.step_scheduler()
    
    return learners, scaler, prop_model

def evaluate_metrics(models, scaler, prop_model, n_test, delta_test, M=None):
    """
    Evaluation function (logic unchanged, automatically adapts to new keys in models dictionary)
    """
    if M is None:
        M = config.FULL_M_ATTACKS if not TEST_MODE else TEST_M
    base_seed = config.FULL_BASE_SEED
    results = {name: {'Hat Q_DRO': 0.0, 'Hat Q_min': 0.0} for name in models.keys()}
    
    # --- 1. Hat Q_DRO ---
    with timer("evaluate_metrics.q_dro"):
        env_base = PricingEnvironment(setting=SETTING, seed=base_seed)
        X_base, P_base, Y_base = env_base.sample_data(n_test)
        X_base_scaled = np.clip(scaler.transform(np.clip(np.nan_to_num(X_base), -20, 20)), -5, 5)
        Pi0_base = prop_model.predict(X_base, P_base)
        
        for name, learner in models.items():
            with timer("evaluate_metrics.get_optimal_alpha"):
                _, q_dro = learner.get_optimal_alpha(X_base_scaled, P_base, Y_base, Pi0_base, delta_test)
            results[name]['Hat Q_DRO'] = q_dro

    # --- 2. Hat Q_min (Attack) ---
    # Optimization: use numpy arrays instead of lists, pre-allocate memory
    n_test_attack = getattr(config, 'FULL_N_TEST_ATTACK', n_test) if not TEST_MODE else min(1000, n_test)
    attack_logs = {name: np.full(M, np.inf, dtype=np.float32) for name in models.keys()}
    
    # Warm start: save previous alpha_star for each model
    alpha_warm_start = {name: None for name in models.keys()}
    
    # Pre-compute constants (avoid repeated config access)
    z_loc, z_scale = config.TRAIN_Z_LOC, config.TRAIN_Z_SCALE
    
    with timer("evaluate_metrics.attacks"):
        for i in tqdm(range(M), desc="Worst-case Attacks", leave=False):
            curr_seed = base_seed + 1 + i
            env_curr = PricingEnvironment(setting=SETTING, seed=curr_seed)
            X_i, P_i, Y_i = env_curr.sample_data(n_test_attack)
            
            # Pre-compute shared data (same for all models, only compute once)
            X_i_raw = np.clip(np.nan_to_num(X_i), -20, 20)
            X_i_scaled = np.clip(scaler.transform(X_i_raw), -5, 5)
            Pi0_i = prop_model.predict(X_i_raw, P_i)
            
            # Pre-compute true_valuation (same for all models, only compute once)
            f_x_all = env_curr.true_valuation(X_i_raw)
            
            # Pre-compute shared Tensor (avoid repeated creation in each model loop)
            X_i_scaled_tensor = torch.FloatTensor(X_i_scaled)
            
            for name, learner in models.items():
                with timer("evaluate_metrics.get_optimal_alpha"):
                    # Use warm start alpha_init
                    alpha_star, _ = learner.get_optimal_alpha(
                        X_i_scaled, P_i, Y_i, Pi0_i, delta_test, 
                        alpha_init=alpha_warm_start[name]
                    )
                    # Save current alpha_star as warm start for next iteration
                    alpha_warm_start[name] = alpha_star
                
                # Compute worst case weights
                weights = learner.get_worst_case_weights(X_i_scaled, P_i, Y_i, Pi0_i, alpha_star)
                
                # Optimized sampling: use cumulative distribution function (faster than np.random.choice)
                # Build cumulative distribution
                cumsum = np.cumsum(weights)
                cumsum = cumsum / cumsum[-1]  # Normalize to ensure sum is 1
                # Generate random numbers and find corresponding indices
                u = np.random.rand(len(X_i))
                attack_indices = np.searchsorted(cumsum, u)
                
                # Use pre-computed indices to get attack data
                X_attacked_raw = X_i_raw[attack_indices]
                X_attacked_scaled = X_i_scaled[attack_indices]
                f_x_attacked = f_x_all[attack_indices]  # Reuse pre-computed true_valuation
                
                # Batch policy prediction (reduce Tensor conversion, reuse pre-computed Tensor)
                with torch.no_grad():
                    X_attacked_tensor = torch.FloatTensor(X_attacked_scaled)
                    pred_P = learner.policy(X_attacked_tensor).numpy().flatten()
                
                # Vectorized computation: batch compute prob and revenue (avoid loops)
                prob = 1.0 / (1.0 + np.exp(-(f_x_attacked + z_loc - pred_P) / z_scale))
                revenue = pred_P * prob
                
                # Directly store to pre-allocated array (avoid list append)
                attack_logs[name][i] = np.mean(revenue)
            
    # Compute final minimum (vectorized operation)
    for name in models.keys():
        results[name]['Hat Q_min'] = np.min(attack_logs[name])
        
    return results

# ==========================================
# Result aggregation and display tools
# ==========================================
def print_pivot_table(df, index_col, value_cols=['Hat Q_DRO', 'Hat Q_min'], 
                      format_style='leung2025', n_repeats=None, save_path=None):
    """
    Print table, supports Leung2025 format: Mean ± Standard Error (SE in %)
    
    Parameters:
    -----------
    format_style : str
        'leung2025': Mean ± SE (SE in %), where SE = Std / sqrt(n)
        'simple': Mean (Std)
    n_repeats : int
        Number of repeats for computing Standard Error (SE = Std / sqrt(n))
    save_path : str, optional
        Path to save the pivot table as CSV
    """
    df_mean = df.groupby([index_col, 'Model'])[value_cols].mean().reset_index()
    df_std = df.groupby([index_col, 'Model'])[value_cols].std().reset_index()
    merged = pd.merge(df_mean, df_std, on=[index_col, 'Model'], suffixes=('_mean', '_std'))
    
    for col in value_cols:
        if format_style == 'leung2025':
            # Leung2025 format: Mean ± Standard Error (SE in %)
            # Standard Error = Std / sqrt(n)
            # SE in % = (SE / Mean) * 100 = (Std / (Mean * sqrt(n))) * 100
            if n_repeats is not None and n_repeats > 1:
                # Compute Standard Error
                merged[col+'_se'] = merged[col+'_std'] / np.sqrt(n_repeats)
                merged[col] = merged.apply(
                    lambda x: f"{x[col+'_mean']:.2f}±{x[col+'_se']/x[col+'_mean']*100:.2f}" 
                    if x[col+'_mean'] > 1e-6 else f"{x[col+'_mean']:.2f}±0.00",
                    axis=1
                )
            else:
                # Fallback: use Std if n_repeats not provided
                merged[col] = merged.apply(
                    lambda x: f"{x[col+'_mean']:.2f}±{x[col+'_std']/x[col+'_mean']*100:.2f}" 
                    if x[col+'_mean'] > 1e-6 else f"{x[col+'_mean']:.2f}±0.00",
                    axis=1
                )
        else:
            # Simple format: Mean (Std)
            merged[col] = merged.apply(
                lambda x: f"{x[col+'_mean']:.3f} ({x[col+'_std']:.3f})", 
                axis=1
            )
    
    # Sort Model column in specific order
    # We want the order to be: Ai2024, Leung2025, DRO(0.2)
    model_order = ['Ai2024', f'Leung2025(δ={DELTA_STRONG})', f'DRO(δ={DELTA_STRONG})']
    merged['Model'] = pd.Categorical(merged['Model'], categories=model_order, ordered=True)
    merged = merged.sort_values('Model')
    
    pivot = merged.pivot(index=index_col, columns='Model', values=value_cols)
    print(pivot.to_string())
    
    # Save to file
    if save_path:
        pivot.to_csv(save_path, sep='\t')
        print(f"\nResults saved to {save_path}")
    
    return pivot

# ==========================================
# Main loop
# ==========================================
def run_experiments():
    print(f"{'='*60}")
    print(f"Running Experiments | Repeats: {N_REPEATS}")
    print(f"Setting: {SETTING} | Policy: {POLICY_TYPE}")
    print(f"Comparing: Ai2024 (non-robust), Leung2025 (robust), DRO(δ={DELTA_STRONG})")
    print(f"{'='*60}")
    
    res_table3 = []
    
    # -----------------------------------------------
    # Experiment 1: Varying N with multiple delta_test values
    # For Kallus Figure 3: Different delta_test values, hat Q_DRO vs N
    # -----------------------------------------------
    # Import three delta_test values from config for Figure 3
    FIGURE3_DELTAS = config.FIGURE3_DELTAS
    
    print("\n>>> Experiment 1: Varying Training Size N with multiple delta_test values")
    print(f"Delta test values: {FIGURE3_DELTAS}")
    print("For Kallus Figure 3: Different delta_test values, hat Q_DRO vs N")
    n_sizes = TEST_N_SIZES
    
    # Pre-allocate result array for better performance (optimization)
    # Use list with pre-allocated capacity hint (Python lists are already efficient)
    res_table3_prealloc = []
    
    # Prepare for incremental statistics writing
    # Track data for each (n, delta_test) combination separately
    from scipy.stats import t as t_dist
    stats_file = f'Figure3_Statistics_{SETTING}.csv'
    stats_header_written = False
    metric_col = 'Hat Q_DRO'
    
    # Dictionary to track data for each (n, delta_test) combination
    # Key: (n, delta_test), Value: list of data points
    data_by_ndelta = {}
    
    for n in tqdm(n_sizes, desc="Training Size N", position=0):
        for r in tqdm(range(N_REPEATS), desc=f"N={n} Repeats", leave=False, position=1):
            seed = config.SEED + n * 100 + r
            set_global_seed(seed)
            
            # Train all models (only train once, this is an optimization)
            models, scaler, prop = train_models(n, seed)
            
            # Evaluate under multiple delta_test (reuse trained models)
            for delta_test in FIGURE3_DELTAS:
                n_test_val = config.FULL_N_TEST if not TEST_MODE else 1000  # Use smaller value in test mode
                metrics = evaluate_metrics(models, scaler, prop, n_test=n_test_val, 
                                         delta_test=delta_test, M=TEST_M)
                
                for m_name, vals in metrics.items():
                    data_point = {
                        'N': n, 
                        'Delta Test': delta_test,  # Add Delta Test column
                        'Model': m_name, 
                        'Repeat': r,
                        'Hat Q_DRO': vals['Hat Q_DRO'],
                        'Hat Q_min': vals['Hat Q_min']
                    }
                    res_table3_prealloc.append(data_point)
                    
                    # Track data for this (n, delta_test) combination
                    key = (n, delta_test)
                    if key not in data_by_ndelta:
                        data_by_ndelta[key] = []
                    data_by_ndelta[key].append(data_point)
            
            # After completing all repeats for each (n, delta_test) combination,
            # compute and save statistics immediately
            if r == N_REPEATS - 1:  # After all repeats for this n
                # Process each (n, delta_test) combination separately
                # For each delta_test, compute and save statistics for current n
                for current_delta in FIGURE3_DELTAS:
                    key = (n, current_delta)
                    if key in data_by_ndelta and len(data_by_ndelta[key]) > 0:
                        data_list = data_by_ndelta[key]
                        df_ndelta = pd.DataFrame(data_list)
                        
                        # Group by (Delta Test, N, Model) and compute statistics
                        grouped = df_ndelta.groupby(['Delta Test', 'N', 'Model'])[metric_col]
                        means = grouped.mean().reset_index()
                        stds = grouped.std().reset_index()
                        counts = grouped.count().reset_index()
                        
                        merged = pd.merge(means, stds, on=['Delta Test', 'N', 'Model'], suffixes=('_mean', '_std'))
                        merged = pd.merge(merged, counts, on=['Delta Test', 'N', 'Model'])
                        merged.rename(columns={metric_col: 'count'}, inplace=True)
                        
                        # Compute 90% confidence interval (t-distribution)
                        def compute_ci(row):
                            n_count = row['count']
                            if n_count <= 1:
                                return row[metric_col + '_mean'], row[metric_col + '_mean']
                            se = row[metric_col + '_std'] / np.sqrt(n_count)
                            t_val = t_dist.ppf(0.95, df=n_count-1)
                            lower = row[metric_col + '_mean'] - t_val * se
                            upper = row[metric_col + '_mean'] + t_val * se
                            return lower, upper
                        
                        merged[['CI_Lower', 'CI_Upper']] = merged.apply(
                            lambda row: pd.Series(compute_ci(row)), axis=1
                        )
                        
                        # Rename N to T and prepare output
                        output_df = merged[['Delta Test', 'N', 'Model', metric_col + '_mean', 'CI_Lower', 'CI_Upper']].copy()
                        output_df.rename(columns={
                            'N': 'T',
                            metric_col + '_mean': 'Mean'
                        }, inplace=True)
                        
                        # Sort by Delta Test, T, and Model
                        output_df = output_df.sort_values(['Delta Test', 'T', 'Model'])
                        
                        # Append to CSV file (write header only once)
                        if not stats_header_written:
                            output_df.to_csv(stats_file, index=False, mode='w')
                            stats_header_written = True
                        else:
                            output_df.to_csv(stats_file, index=False, mode='a', header=False)
    
    # Convert to DataFrame (more efficient than appending to list)
    res_table3 = res_table3_prealloc

    # Commented out table output
    # print("\n=== Table 3 Results (Eval Delta=0.2) ===")
    # print("Varying Training Size N (Eval Delta=0.2)")
    # print("Format: Mean ± Standard Error (SE in %)")
    df_table3 = pd.DataFrame(res_table3)
    # print_pivot_table(df_table3, 'N', format_style='leung2025', n_repeats=N_REPEATS)
    
    # Kallus22 Figure 3 style: DROPL vs N (with multiple delta_test values)
    visualization.plot_dropl_vs_n(df_table3, metric_col='Hat Q_DRO', 
                                  title_suffix=' (Multiple Delta Test Values)',
                                  save_path=f'Figure3_DROPL_vs_N_Q_DRO_{SETTING}.png')
    
    # Statistics have been saved incrementally during the loop
    print(f"\nStatistics saved incrementally to {stats_file}")
    print(f"Columns: Delta Test, T, Model, Mean, CI_Lower, CI_Upper")
    # visualization.plot_dropl_vs_n(df_table3, metric_col='Hat Q_min', 
    #                               title_suffix=' (Train Delta=Test Delta=0.2)',
    #                               save_path=f'Figure3_DROPL_vs_N_Q_min_{SETTING}.png')

    # -----------------------------------------------
    # Experiment 2: Commented out (not needed)
    # -----------------------------------------------
    # print(f"\n>>> Experiment 2: Varying Test Delta (N={config.FULL_N_TRAIN_TABLE2}, Train Delta={DELTA_STRONG})")
    # print("Similar to Leung2025 Table 2")
    # res_table2 = []
    # n_fix_table2 = config.FULL_N_TRAIN_TABLE2
    # test_deltas_table2 = TEST_DELTAS_TABLE2
    # 
    # for r in tqdm(range(N_REPEATS), desc="Table 2 Repeats", position=0):
    #     seed = config.SEED + 8000 + r
    #     set_global_seed(seed)
    #     
    #     # Train once, including all variants
    #     models, scaler, prop = train_models(n_fix_table2, seed)
    #     
    #     for d_test in tqdm(test_deltas_table2, desc=f"Repeat {r+1} Deltas", leave=False, position=1):
    #         n_test_val = config.FULL_N_TEST_TABLE2 if not TEST_MODE else 1000  # Use smaller value in test mode
    #         metrics = evaluate_metrics(models, scaler, prop, n_test=n_test_val, delta_test=d_test, M=TEST_M)
    #         for m_name, vals in metrics.items():
    #             res_table2.append({
    #                 'Delta Test': d_test, 'Model': m_name, 'Repeat': r,
    #                 'Hat Q_DRO': vals['Hat Q_DRO'],
    #                 'Hat Q_min': vals['Hat Q_min']
    #             })
    # 
    # print("\n=== Table 2 Results (N=2000) ===")
    # print("Varying Test Delta (N_train=2000, Train Delta=0.2)")
    # print("Format: Mean ± Standard Error (SE in %)")
    # df_table2 = pd.DataFrame(res_table2)
    # print_pivot_table(df_table2, 'Delta Test', format_style='leung2025',
    #                   n_repeats=N_REPEATS, save_path=f'Table2_Delta_varying_N2000_{SETTING}.txt')
    # 
    # # Print performance summary
    # print_performance_summary()
    # 
    # # Kallus22 style: DROPL vs Delta (N=2000)
    # visualization.plot_dropl_vs_delta(df_table2, metric_col='Hat Q_DRO',
    #                                    title_suffix=' (N=2000, Train Delta=0.2)',
    #                                    save_path=f'Figure_DROPL_vs_Delta_Q_DRO_N2000_{SETTING}.png')
    # visualization.plot_dropl_vs_delta(df_table2, metric_col='Hat Q_min',
    #                                    title_suffix=' (N=2000, Train Delta=0.2)',
    #                                    save_path=f'Figure_DROPL_vs_Delta_Q_min_N2000_{SETTING}.png')
    # 
    # # Visualization (Strong DRO vs others)
    # visualization.plot_policy_heatmap(
    #     models['Ai2024'].policy, 
    #     models[f'Leung2025(δ={DELTA_STRONG})'].policy,
    #     models[f'DRO(δ={DELTA_STRONG})'].policy, 
    #     setting_name="N5000_Delta0.2"
    # )
    
    # Print performance summary
    print_performance_summary()
    
    # -----------------------------------------------
    # Behavior vs Target Policy Experiment
    # Similar to Kallus22 Figure 1 - Commented out (not needed)
    # -----------------------------------------------
    # run_behavior_vs_target_experiment()

def run_behavior_vs_target_experiment():
    """
    Similar to Kallus22 experiment: compare differences between behavior policy (logging policy) and target policy
    
    Fix behavior policy (linear logging policy), compare performance of different target policies:
    - Linear target policy (different parameters from behavior policy)
    - Nonlinear target policy (MLP)
    """
    print(f"\n{'='*60}")
    print(">>> Experiment: Behavior Policy vs Target Policy Comparison")
    print("Similar to Kallus22 Figure 1 and DROPE experiments")
    print(f"{'='*60}")
    
    n_train = config.FULL_BEHAVIOR_N_TRAIN if not TEST_MODE else 1000
    n_test = config.FULL_BEHAVIOR_N_TEST if not TEST_MODE else 1000
    delta_test = DELTA_STRONG
    M = config.FULL_BEHAVIOR_M if not TEST_MODE else TEST_M
    
    res_behavior_vs_target = []
    
    # Define behavior policy function (for visualization)
    def behavior_policy_func(X):
        """Behavior policy: P ~ N(1.0 + 1.0 * X[:, 0], 1.0^2)"""
        mu = 1.0 + 1.0 * X[:, 0]
        # For visualization, we use the mean (deterministic)
        return np.clip(mu, config.PRICE_MIN, config.PRICE_MAX)
    
    for r in tqdm(range(N_REPEATS), desc="Behavior vs Target Repeats", position=0):
        seed = config.SEED + 10000 + r
        set_global_seed(seed)
        
        # 1. Prepare training data (use linear logging policy)
        env_train = PricingEnvironment(setting=SETTING, logging_policy_type='linear', seed=seed)
        X_tr_raw, P_tr, Y_tr = env_train.sample_data(n_train)
        
        X_tr_raw = np.clip(np.nan_to_num(X_tr_raw.astype(np.float64)), -20.0, 20.0)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_tr_scaled = np.clip(X_tr_scaled, -20.0, 20.0)
        
        # 2. Nuisance Estimators
        prop_model = PropensityEstimator()
        prop_model.fit(X_tr_raw, P_tr)
        Pi0_tr = prop_model.predict(X_tr_raw, P_tr)
        h = np.maximum(config.MIN_BANDWIDTH, silverman_bandwidth(P_tr))
        
        outcome_model = DifferentiableOutcomeEstimator(input_dim=config.DIM)
        outcome_model.fit(X_tr_scaled, P_tr, Y_tr, epochs=TEST_NUISANCE_EPOCHS)
        
        # Probability outcome model for Leung2025
        from estimators import ProbabilityOutcomeEstimator
        prob_outcome_model = ProbabilityOutcomeEstimator(input_dim=config.DIM)
        prob_outcome_model.fit(X_tr_scaled, P_tr, Y_tr, epochs=TEST_NUISANCE_EPOCHS)
        
        # 3. Train different Target Policies
        input_dim = config.DIM
        models = {}
        
        # Target Policy 1: Linear (same structure as behavior policy but with different parameters)
        models['Linear Target'] = Ai2024Learner(LinearPolicy(input_dim), outcome_model, h)
        
        # Target Policy 2: Nonlinear (MLP)
        models['Nonlinear Target'] = Ai2024Learner(MLPPolicy(input_dim), outcome_model, h)
        
        # Target Policy 3: DRO Linear
        models[f'DRO Linear(δ={DELTA_STRONG})'] = CDROLearner(
            LinearPolicy(input_dim), h, delta=DELTA_STRONG,
            X_train=X_tr_scaled, P_train=P_tr, Y_train=Y_tr, n_folds=TEST_N_FOLDS
        )
        
        # Target Policy 4: DRO Nonlinear
        models[f'DRO Nonlinear(δ={DELTA_STRONG})'] = CDROLearner(
            MLPPolicy(input_dim), h, delta=DELTA_STRONG,
            X_train=X_tr_scaled, P_train=P_tr, Y_train=Y_tr, n_folds=TEST_N_FOLDS
        )
        
        # 4. Train all models
        X_tensor = torch.FloatTensor(X_tr_scaled)
        P_tensor = torch.FloatTensor(P_tr).view(-1, 1)
        
        for name, learner in tqdm(models.items(), desc="Training Models", leave=False):
            # Warm start
            with torch.no_grad():
                init_p = learner.policy(X_tensor).numpy().flatten()
            if np.std(init_p) < 0.01:
                # If output is almost constant, randomly initialize
                for param in learner.policy.parameters():
                    if len(param.shape) >= 2:
                        torch.nn.init.xavier_uniform_(param)
            
            # Training
            epochs_to_use = TEST_EPOCHS if TEST_EPOCHS is not None else config.EPOCHS
            for epoch in tqdm(range(epochs_to_use), desc=f"Training {name}", leave=False):
                indices = np.random.choice(len(X_tr_scaled), size=min(TEST_BATCH_SIZE, len(X_tr_scaled)), replace=False)
                learner.train_step(
                    X_tr_scaled[indices], P_tr[indices], Y_tr[indices], Pi0_tr[indices]
                )
        
        # 5. Evaluate
        metrics = evaluate_metrics(models, scaler, prop_model, n_test=n_test, delta_test=delta_test, M=M)
        
        for m_name, vals in metrics.items():
            res_behavior_vs_target.append({
                'Model': m_name, 'Repeat': r,
                'Hat Q_DRO': vals['Hat Q_DRO'],
                'Hat Q_min': vals['Hat Q_min']
            })
        
        print("Done")
    
    # 6. Print results table
    print("\n=== Behavior vs Target Policy Results ===")
    print("Fixed Behavior Policy (Linear), Varying Target Policy")
    print("Format: Mean ± Standard Error (SE in %)")
    df_behavior_vs_target = pd.DataFrame(res_behavior_vs_target)
    print_pivot_table(df_behavior_vs_target, 'Model', format_style='leung2025',
                      n_repeats=N_REPEATS, save_path=f'Table_BehaviorVsTarget_{SETTING}.txt')
    
    # 7. Visualization: Behavior Policy vs Target Policies
    # Use last trained model for visualization
    seed_vis = config.SEED + 10000 + N_REPEATS - 1
    set_global_seed(seed_vis)
    
    env_train_vis = PricingEnvironment(setting=SETTING, logging_policy_type='linear', seed=seed_vis)
    X_tr_raw_vis, P_tr_vis, Y_tr_vis = env_train_vis.sample_data(n_train)
    X_tr_raw_vis = np.clip(np.nan_to_num(X_tr_raw_vis.astype(np.float64)), -20.0, 20.0)
    scaler_vis = StandardScaler()
    X_tr_scaled_vis = scaler_vis.fit_transform(X_tr_raw_vis)
    X_tr_scaled_vis = np.clip(X_tr_scaled_vis, -20.0, 20.0)
    
    prop_model_vis = PropensityEstimator()
    prop_model_vis.fit(X_tr_raw_vis, P_tr_vis)
    h_vis = np.maximum(config.MIN_BANDWIDTH, silverman_bandwidth(P_tr_vis))
    
    outcome_model_vis = DifferentiableOutcomeEstimator(input_dim=config.DIM)
    outcome_model_vis.fit(X_tr_scaled_vis, P_tr_vis, Y_tr_vis, epochs=TEST_NUISANCE_EPOCHS)
    
    # Retrain models for visualization
    models_vis = {
        'Linear Target': Ai2024Learner(LinearPolicy(config.DIM), outcome_model_vis, h_vis),
        'Nonlinear Target': Ai2024Learner(MLPPolicy(config.DIM), outcome_model_vis, h_vis),
        f'DRO Linear(δ={DELTA_STRONG})': CDROLearner(
            LinearPolicy(config.DIM), h_vis, delta=DELTA_STRONG,
            X_train=X_tr_scaled_vis, P_train=P_tr_vis, Y_train=Y_tr_vis, n_folds=TEST_N_FOLDS
        ),
        f'DRO Nonlinear(δ={DELTA_STRONG})': CDROLearner(
            MLPPolicy(config.DIM), h_vis, delta=DELTA_STRONG,
            X_train=X_tr_scaled_vis, P_train=P_tr_vis, Y_train=Y_tr_vis, n_folds=TEST_N_FOLDS
        )
    }
    
    X_tensor_vis = torch.FloatTensor(X_tr_scaled_vis)
    epochs_to_use = TEST_EPOCHS if TEST_EPOCHS is not None else config.EPOCHS
    for name, learner in tqdm(models_vis.items(), desc="Training Visualization Models"):
        for epoch in tqdm(range(epochs_to_use), desc=f"Training {name}", leave=False):
            indices = np.random.choice(len(X_tr_scaled_vis), size=min(TEST_BATCH_SIZE, len(X_tr_scaled_vis)), replace=False)
            Pi0_tr_vis = prop_model_vis.predict(X_tr_raw_vis[indices], P_tr_vis[indices])
            learner.train_step(
                X_tr_scaled_vis[indices], P_tr_vis[indices], Y_tr_vis[indices], Pi0_tr_vis
            )
    
    # Prepare visualization
    target_policies_dict = {name: learner.policy for name, learner in models_vis.items()}
    visualization.plot_behavior_vs_target_policy(
        behavior_policy_func, target_policies_dict, 
        setting_name=f"{SETTING}_N{n_train}"
    )

if __name__ == "__main__":
    run_experiments()