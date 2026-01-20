import numpy as np
import torch
import pandas as pd
import time
import random
import os
from tqdm import tqdm
from contextlib import contextmanager
from collections import defaultdict
from scipy.stats import t as t_dist

from src.dgp import PricingEnvironment
from src.estimators import PropensityEstimator, silverman_bandwidth
from src.policy import LinearPolicy
from src.learner import OPELearner, Leung2025Learner
from src import config
from sklearn.preprocessing import StandardScaler
from theoretical_ope import compute_theoretical_R_delta

# ==========================================
# Global settings
# ==========================================
SETTING = 'linear'

# ==========================================
# True R_delta computation method selection
# ==========================================
# 'sampling': Use sampling method (current method, based on large number of samples)
# 'theoretical': Use numerical integration method (theoretical computation, more accurate but slower)
TRUE_R_DELTA_METHOD = 'sampling'  # or 'theoretical'

# ==========================================
# Test mode configuration
# ==========================================
TEST_MODE = False
# TEST_MODE = True

if TEST_MODE:
    N_REPEATS = 5
    TEST_N_SIZES = [1000,4000,5000]  # Test only 2 training set sizes in test mode
    TEST_DELTAS = [0.1]  # Test only 2 deltas in test mode
    print("=" * 60)
    print("WARNING: Running in TEST MODE - Results are for testing only!")
    print("Set TEST_MODE = False for full experimental runs")
    print("=" * 60)
else:
    N_REPEATS =  config.FULL_N_REPEATS  # Full run uses 20 repeats
    TEST_N_SIZES = [500,1000,2000,3000,4000,5000]  # Full run uses all training set sizes
    TEST_DELTAS = config.FIGURE3_DELTAS  # [0.1, 0.2, 0.3]

# ==========================================
# Utility functions
# ==========================================
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

def create_target_policy():
    """
    Create randomly initialized target policy (created only once, fixed usage)
    """
    target_policy = LinearPolicy(input_dim=config.DIM)
    
    # Use fixed random seed for initialization to ensure reproducibility
    torch.manual_seed(config.TARGET_POLICY_SEED)
    for param in target_policy.parameters():
        if len(param.shape) >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.normal_(param, mean=0.0, std=0.1)
    
    target_policy.eval()  # Fixed to evaluation mode
    return target_policy

def compute_true_R_delta(target_policy, delta_test, n_test=10000, method='sampling'):
    """
    Compute true R_delta as ground truth
    
    Supports two methods:
    1. 'sampling': Use large number of test samples computed via Algorithm 1 (current method)
    2. 'theoretical': Use numerical integration to directly compute theoretical value (more accurate but slower)
    
    Parameters:
    -----------
    target_policy : torch.nn.Module
        Target policy to evaluate
    delta_test : float
        Test uncertainty radius δ
    n_test : int
        Number of test samples (default: 10000, only used for sampling method)
    method : str
        Computation method: 'sampling' or 'theoretical'
    
    Returns:
    --------
    true_R_delta : float
        True distributionally robust value
    """
    if method == 'theoretical':
        # Use numerical integration method
        print(f"  Computing theoretical true R_delta using numerical integration...")
        true_R_delta, alpha_star = compute_theoretical_R_delta(
            target_policy=target_policy,
            delta_test=delta_test,
            setting=SETTING,
            logging_policy_type='nonlinear',
            n_mc_samples=10000  # Number of Monte Carlo samples for integrating over X
        )
        print(f"  Theoretical R_delta = {true_R_delta:.6f}, alpha* = {alpha_star:.6f}")
        return true_R_delta
    else:
        # Use sampling method (original method)
        # Generate large number of test data
        env_test = PricingEnvironment(setting=SETTING, logging_policy_type='nonlinear', seed=config.SEED + 99999)
        X_test, P_test, Y_test = env_test.sample_data(n_test)
        
        X_test = np.clip(np.nan_to_num(X_test.astype(np.float64)), -20.0, 20.0)
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        X_test_scaled = np.clip(X_test_scaled, -20.0, 20.0)
        
        # Compute bandwidth: consistent with OPL, use Silverman bandwidth with minimum limit
        h = np.maximum(config.MIN_BANDWIDTH, silverman_bandwidth(P_test))
        
        # Use Algorithm 1 (OPELearner) to compute true R_delta
        ope_learner = OPELearner(
            target_policy=target_policy,
            bandwidth=h,
            delta=delta_test,
            X_train=X_test_scaled,
            P_train=P_test,
            Y_train=Y_test,
            n_folds=config.OPE_N_FOLDS
        )
        
        true_R_delta, _ = ope_learner.evaluate(delta_test=delta_test)
        return true_R_delta

def evaluate_with_ldr2pe_cp(target_policy, X_train, P_train, Y_train, delta_test, h, scaler=None):
    """
    Evaluate target policy using LDR²PE-CP (Algorithm 1)
    
    Parameters:
    -----------
    scaler : StandardScaler, optional
        Pre-fitted scaler (if None, will fit new one)
    
    Returns:
    --------
    R_delta : float
        Estimated R_δ
    """
    X_train_clean = np.clip(np.nan_to_num(X_train.astype(np.float64)), -20.0, 20.0)
    
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
    else:
        X_train_scaled = scaler.transform(X_train_clean)
    
    X_train_scaled = np.clip(X_train_scaled, -20.0, 20.0)
    
    ope_learner = OPELearner(
        target_policy=target_policy,
        bandwidth=h,
        delta=delta_test,
        X_train=X_train_scaled,
        P_train=P_train,
        Y_train=Y_train,
        n_folds=config.OPE_N_FOLDS
    )
    
    R_delta, _ = ope_learner.evaluate(delta_test=delta_test)
    return R_delta

def evaluate_with_dro_ipw(target_policy, X_train, P_train, Y_train, delta_test, h, prop_model, scaler=None):
    """
    Evaluate target policy using DRO-IPW (Leung2025 method)
    
    Parameters:
    -----------
    scaler : StandardScaler, optional
        Pre-fitted scaler (if None, will fit new one)
    
    Returns:
    --------
    R_delta : float
        Estimated R_δ
    """
    X_train_clean = np.clip(np.nan_to_num(X_train.astype(np.float64)), -20.0, 20.0)
    
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
    else:
        X_train_scaled = scaler.transform(X_train_clean)
    
    X_train_scaled = np.clip(X_train_scaled, -20.0, 20.0)
    
    # Create Leung2025Learner (no need to train policy, only need evaluate_policy method)
    # Use a dummy policy since we only use evaluate_policy method
    dummy_policy = LinearPolicy(input_dim=config.DIM)
    # Leung2025Learner requires outcome_model parameter, but we don't need it, pass a dummy
    from estimators import ProbabilityOutcomeEstimator
    dummy_outcome = ProbabilityOutcomeEstimator(input_dim=config.DIM)
    leung_learner = Leung2025Learner(
        policy_net=dummy_policy,
        outcome_model=dummy_outcome,  # Dummy, not used in evaluate_policy
        bandwidth=h,
        delta=delta_test
    )
    
    # Compute propensity scores
    Pi0_train = prop_model.predict(X_train, P_train)
    
    # Evaluate target policy
    R_delta, _ = leung_learner.evaluate_policy(
        target_policy=target_policy,
        X_batch=X_train_scaled,
        P_batch=P_train,
        Y_batch=Y_train,
        Pi0_batch=Pi0_train,
        delta_test=delta_test
    )
    
    return R_delta

def run_ope_experiments():
    """
    Main experiment function: compute MSE of distributionally robust value for different delta_test and T
    """
    print(f"{'='*60}")
    print(f"Running OPE Experiments | Repeats: {N_REPEATS}")
    print(f"Setting: {SETTING}")
    print(f"Comparing: LDR²PE-CP vs DRO-IPW")
    print(f"{'='*60}")
    
    # Step 0: Create randomly initialized target policy (created only once, fixed usage)
    print("\n>>> Creating target policy...")
    target_policy = create_target_policy()
    print("Target policy created and fixed.")
    
    # Compute true R_delta for each delta_test (computed only once)
    print("\n>>> Computing true R_delta for each delta_test...")
    print(f"Method: {TRUE_R_DELTA_METHOD}")
    true_R_delta_dict = {}
    for delta_test in tqdm(TEST_DELTAS, desc="Computing True R_delta"):
        if TRUE_R_DELTA_METHOD == 'theoretical':
            true_R_delta = compute_true_R_delta(target_policy, delta_test, 
                                                 method='theoretical')
        else:
            true_R_delta = compute_true_R_delta(target_policy, delta_test, 
                                                 n_test=config.OPE_N_TEST_TRUE, 
                                                 method='sampling')
        true_R_delta_dict[delta_test] = true_R_delta
        print(f"True R_delta (δ={delta_test}) = {true_R_delta:.6f}")
    
    # Pre-allocate result arrays (optimization: avoid memory reallocation from list.append)
    print("\n>>> Running OPE experiments...")
    total_experiments = len(TEST_DELTAS) * len(TEST_N_SIZES) * N_REPEATS * 2  # 2 methods
    pbar_total = tqdm(total=total_experiments, desc="Overall Progress", position=0)
    
    # Prepare incremental statistics saving
    stats_file = f'OPE_Statistics_{SETTING}.csv'
    stats_header_written = False
    data_by_delta_T = {}  # Key: (delta_test, T), Value: list of data points
    
    # Pre-allocate result arrays for better performance
    results_delta_test = np.empty(total_experiments, dtype=np.float64)
    results_T = np.empty(total_experiments, dtype=np.int32)
    results_method = np.empty(total_experiments, dtype=object)
    results_repeat = np.empty(total_experiments, dtype=np.int32)
    results_estimated = np.empty(total_experiments, dtype=np.float64)
    results_true = np.empty(total_experiments, dtype=np.float64)
    results_mse = np.empty(total_experiments, dtype=np.float64)
    
    result_idx = 0
    
    for delta_test in tqdm(TEST_DELTAS, desc="Delta Test", position=1, leave=False):
        true_R_delta = true_R_delta_dict[delta_test]
        
        for T in tqdm(TEST_N_SIZES, desc=f"T (δ={delta_test})", leave=False, position=2):
            # Initialize data tracking for current (delta_test, T) combination
            key = (delta_test, T)
            if key not in data_by_delta_T:
                data_by_delta_T[key] = []
            
            for repeat in range(N_REPEATS):
                pbar_total.set_description(f"δ={delta_test}, T={T}, Repeat={repeat+1}/{N_REPEATS}")
                seed = int(config.SEED + delta_test * 10000 + T * 100 + repeat)
                set_global_seed(seed)
                
                # 1. Generate training data (size T)
                env_train = PricingEnvironment(setting=SETTING, logging_policy_type='nonlinear', seed=seed)
                X_tr, P_tr, Y_tr = env_train.sample_data(T)
                
                X_tr = np.clip(np.nan_to_num(X_tr.astype(np.float64)), -20.0, 20.0)
                # Compute bandwidth: consistent with OPL, use Silverman bandwidth with minimum limit
                h = np.maximum(config.MIN_BANDWIDTH, silverman_bandwidth(P_tr))
                
                # Create scaler (shared by both methods)
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_tr_scaled = np.clip(X_tr_scaled, -20.0, 20.0)
                
                # Train propensity model (for DRO-IPW method)
                prop_model = PropensityEstimator()
                prop_model.fit(X_tr, P_tr)
                
                # 2. Estimate R_delta using LDR²PE-CP method
                try:
                    R_delta_ldr2pe = evaluate_with_ldr2pe_cp(target_policy, X_tr, P_tr, Y_tr, delta_test, h, scaler)
                    mse_ldr2pe = (R_delta_ldr2pe - true_R_delta) ** 2
                except Exception as e:
                    print(f"Warning: LDR²PE-CP failed for T={T}, δ={delta_test}, repeat={repeat}: {e}")
                    R_delta_ldr2pe = np.nan
                    mse_ldr2pe = np.nan
                
                # 3. Estimate R_delta using DRO-IPW method
                try:
                    R_delta_dro_ipw = evaluate_with_dro_ipw(target_policy, X_tr, P_tr, Y_tr, delta_test, h, prop_model, scaler)
                    mse_dro_ipw = (R_delta_dro_ipw - true_R_delta) ** 2
                except Exception as e:
                    print(f"Warning: DRO-IPW failed for T={T}, δ={delta_test}, repeat={repeat}: {e}")
                    R_delta_dro_ipw = np.nan
                    mse_dro_ipw = np.nan
                
                # 4. Save results (use pre-allocated arrays, avoid append)
                # Also add to data_by_delta_T for incremental statistics saving
                
                # LDR2PE-CP results
                results_delta_test[result_idx] = delta_test
                results_T[result_idx] = T
                results_method[result_idx] = 'LDR2PE-CP'
                results_repeat[result_idx] = repeat
                results_estimated[result_idx] = R_delta_ldr2pe
                results_true[result_idx] = true_R_delta
                results_mse[result_idx] = mse_ldr2pe
                result_idx += 1
                pbar_total.update(1)
                
                # Add to data_by_delta_T
                data_by_delta_T[key].append({
                    'Delta Test': delta_test,
                    'T': T,
                    'Method': 'LDR2PE-CP',
                    'Repeat': repeat,
                    'Estimated_R_delta': R_delta_ldr2pe,
                    'True_R_delta': true_R_delta,
                    'MSE': mse_ldr2pe
                })
                
                # DRO-IPW results
                results_delta_test[result_idx] = delta_test
                results_T[result_idx] = T
                results_method[result_idx] = 'DRO-IPW'
                results_repeat[result_idx] = repeat
                results_estimated[result_idx] = R_delta_dro_ipw
                results_true[result_idx] = true_R_delta
                results_mse[result_idx] = mse_dro_ipw
                result_idx += 1
                pbar_total.update(1)
                
                # Add to data_by_delta_T
                data_by_delta_T[key].append({
                    'Delta Test': delta_test,
                    'T': T,
                    'Method': 'DRO-IPW',
                    'Repeat': repeat,
                    'Estimated_R_delta': R_delta_dro_ipw,
                    'True_R_delta': true_R_delta,
                    'MSE': mse_dro_ipw
                })
            
            # After completing all repeats, immediately compute and save statistics
            if key in data_by_delta_T and len(data_by_delta_T[key]) > 0:
                data_list = data_by_delta_T[key]
                df_ndelta = pd.DataFrame(data_list)
                
                # Group by Method to compute statistics
                grouped_mse = df_ndelta.groupby(['Delta Test', 'T', 'Method'])['MSE']
                means_mse = grouped_mse.mean().reset_index()
                stds_mse = grouped_mse.std().reset_index()
                counts_mse = grouped_mse.count().reset_index()
                
                grouped_est = df_ndelta.groupby(['Delta Test', 'T', 'Method'])['Estimated_R_delta']
                means_est = grouped_est.mean().reset_index()
                stds_est = grouped_est.std().reset_index()
                counts_est = grouped_est.count().reset_index()
                
                # Merge statistics
                merged_mse = pd.merge(means_mse, stds_mse, on=['Delta Test', 'T', 'Method'], suffixes=('_mean', '_std'))
                merged_mse = pd.merge(merged_mse, counts_mse, on=['Delta Test', 'T', 'Method'])
                merged_mse.rename(columns={'MSE': 'count_mse'}, inplace=True)
                
                merged_est = pd.merge(means_est, stds_est, on=['Delta Test', 'T', 'Method'], suffixes=('_mean', '_std'))
                merged_est = pd.merge(merged_est, counts_est, on=['Delta Test', 'T', 'Method'])
                merged_est.rename(columns={'Estimated_R_delta': 'count_est'}, inplace=True)
                
                # Merge statistics for MSE and Estimated_R_delta
                merged = pd.merge(merged_mse, merged_est, on=['Delta Test', 'T', 'Method'])
                
                # Compute confidence intervals (90%, consistent with OPL)
                def compute_ci_mse(row):
                    n_count = row['count_mse']
                    if n_count <= 1:
                        return row['MSE_mean'], row['MSE_mean']
                    se = row['MSE_std'] / np.sqrt(n_count)
                    t_val = t_dist.ppf(0.95, df=n_count-1)  # 90% CI
                    lower = row['MSE_mean'] - t_val * se
                    upper = row['MSE_mean'] + t_val * se
                    return lower, upper
                
                def compute_ci_est(row):
                    n_count = row['count_est']
                    if n_count <= 1:
                        return row['Estimated_R_delta_mean'], row['Estimated_R_delta_mean']
                    se = row['Estimated_R_delta_std'] / np.sqrt(n_count)
                    t_val = t_dist.ppf(0.95, df=n_count-1)  # 90% CI
                    lower = row['Estimated_R_delta_mean'] - t_val * se
                    upper = row['Estimated_R_delta_mean'] + t_val * se
                    return lower, upper
                
                merged[['CI_Lower_MSE', 'CI_Upper_MSE']] = merged.apply(
                    lambda row: pd.Series(compute_ci_mse(row)), axis=1
                )
                merged[['CI_Lower_R_delta', 'CI_Upper_R_delta']] = merged.apply(
                    lambda row: pd.Series(compute_ci_est(row)), axis=1
                )
                
                # Prepare output DataFrame
                output_df = merged[[
                    'Delta Test', 'T', 'Method',
                    'MSE_mean', 'CI_Lower_MSE', 'CI_Upper_MSE',
                    'Estimated_R_delta_mean', 'CI_Lower_R_delta', 'CI_Upper_R_delta'
                ]].copy()
                output_df.rename(columns={
                    'MSE_mean': 'Mean_MSE',
                    'Estimated_R_delta_mean': 'Mean_Estimated_R_delta'
                }, inplace=True)
                
                # Sort
                output_df = output_df.sort_values(['Delta Test', 'T', 'Method'])
                
                # Incrementally save to CSV
                if not stats_header_written:
                    output_df.to_csv(stats_file, index=False, mode='w')
                    stats_header_written = True
                else:
                    output_df.to_csv(stats_file, index=False, mode='a', header=False)
    
    pbar_total.close()
    
    # 5. Convert to DataFrame and save as CSV (raw data)
    df_results = pd.DataFrame({
        'Delta Test': results_delta_test,
        'T': results_T,
        'Method': results_method,
        'Repeat': results_repeat,
        'Estimated_R_delta': results_estimated,
        'True_R_delta': results_true,
        'MSE': results_mse
    })
    output_file = f'OPE_Results_{SETTING}.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n>>> Results saved to {output_file}")
    
    # Statistics have been saved incrementally, print confirmation
    print(f"\n>>> Statistics saved incrementally to {stats_file}")
    print(f"Columns: Delta Test, T, Method, Mean_MSE, CI_Lower_MSE, CI_Upper_MSE, Mean_Estimated_R_delta, CI_Lower_R_delta, CI_Upper_R_delta")
    
    # 6. Print summary statistics (read from statistics file)
    if stats_header_written:
        try:
            df_stats = pd.read_csv(stats_file)
            print("\n>>> Summary Statistics (from statistics file):")
            print(df_stats.to_string(index=False))
        except Exception as e:
            print(f"\n>>> Warning: Could not read statistics file: {e}")
            # Fallback to computing from raw data
            print("\n>>> Summary Statistics (Mean MSE by Method, Delta Test, and T):")
            summary = df_results.groupby(['Delta Test', 'T', 'Method'])['MSE'].agg(['mean', 'std', 'count']).reset_index()
            summary.columns = ['Delta Test', 'T', 'Method', 'Mean_MSE', 'Std_MSE', 'Count']
            print(summary.to_string(index=False))
    
    # 7. Plot Figure 2
    from visualization import plot_ope_figure2
    figure_path = f'OPE_Figure2_{SETTING}.png'
    plot_ope_figure2(df_results, save_path=figure_path)
    print(f"\n>>> Figure 2 saved to {figure_path}")
    
    return df_results

if __name__ == "__main__":
    results = run_ope_experiments()
    print("\n>>> OPE experiments completed!")
