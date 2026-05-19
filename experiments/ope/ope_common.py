"""
Shared OPE experiment logic.

OPE uses two robustness radii:
  - delta_train: Phase-1 cross-fitting / nuisance fitting (OPELearner.__init__ delta)
  - delta_test:  Phase-2 evaluation of R_hat_delta (OPELearner.evaluate, DRO-IPW evaluate_policy)
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as t_dist
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src import config
from src.dgp import PricingEnvironment
from src.estimators import PropensityEstimator, ProbabilityOutcomeEstimator, resolve_bandwidth
from src.learner import Leung2025Learner, OPELearner
from src.policy import LinearPolicy


def _sync_core_config(exp_cfg):
    """Mirror OPE experiment settings into src.config for learner / DGP code."""
    config.SEED = exp_cfg.SEED
    config.MIN_BANDWIDTH = exp_cfg.MIN_BANDWIDTH
    config.OPE_N_FOLDS = exp_cfg.OPE_N_FOLDS
    config.OPE_N_ESTIMATORS = exp_cfg.OPE_N_ESTIMATORS


def set_global_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_target_policy(exp_cfg):
    target_policy = LinearPolicy(input_dim=config.DIM)
    torch.manual_seed(exp_cfg.TARGET_POLICY_SEED)
    for param in target_policy.parameters():
        if len(param.shape) >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.normal_(param, mean=0.0, std=0.1)
    target_policy.eval()
    return target_policy


def compute_true_R_delta(exp_cfg, target_policy, delta_train, delta_test, n_test=None):
    """Ground-truth R_delta at delta_test (LDR²PE-CP on held-out data, nuisances at delta_train)."""
    n_test = n_test or exp_cfg.N_TEST

    env_test = PricingEnvironment(
        setting=exp_cfg.SETTING, logging_policy_type="nonlinear", seed=exp_cfg.SEED + 99999
    )
    X_test, P_test, Y_test = env_test.sample_data(n_test)

    X_test = np.clip(np.nan_to_num(X_test.astype(np.float64)), -20.0, 20.0)
    scaler = StandardScaler()
    X_test_scaled = np.clip(scaler.fit_transform(X_test), -20.0, 20.0)

    h = resolve_bandwidth(P_test, mode="silverman", min_bandwidth=exp_cfg.MIN_BANDWIDTH)

    ope_learner = OPELearner(
        target_policy=target_policy,
        bandwidth=h,
        delta=delta_train,
        X_train=X_test_scaled,
        P_train=P_test,
        Y_train=Y_test,
        n_folds=exp_cfg.OPE_N_FOLDS,
    )
    true_R_delta, _ = ope_learner.evaluate(delta_test=delta_test)
    return true_R_delta


def evaluate_with_ldr2pe_cp(
    exp_cfg, target_policy, X_train, P_train, Y_train, delta_train, delta_test, h, scaler=None
):
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
        delta=delta_train,
        X_train=X_train_scaled,
        P_train=P_train,
        Y_train=Y_train,
        n_folds=exp_cfg.OPE_N_FOLDS,
    )
    R_delta, _ = ope_learner.evaluate(delta_test=delta_test)
    return R_delta


def evaluate_with_dro_ipw(
    exp_cfg,
    target_policy,
    X_train,
    P_train,
    Y_train,
    delta_train,
    delta_test,
    h,
    prop_model,
    scaler=None,
):
    X_train_clean = np.clip(np.nan_to_num(X_train.astype(np.float64)), -20.0, 20.0)
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
    else:
        X_train_scaled = scaler.transform(X_train_clean)
    X_train_scaled = np.clip(X_train_scaled, -20.0, 20.0)

    dummy_policy = LinearPolicy(input_dim=config.DIM)
    dummy_outcome = ProbabilityOutcomeEstimator(input_dim=config.DIM)
    leung_learner = Leung2025Learner(
        policy_net=dummy_policy,
        outcome_model=dummy_outcome,
        bandwidth=h,
        delta=delta_train,
    )
    Pi0_train = prop_model.predict(X_train, P_train)
    R_delta, _ = leung_learner.evaluate_policy(
        target_policy=target_policy,
        X_batch=X_train_scaled,
        P_batch=P_train,
        Y_batch=Y_train,
        Pi0_batch=Pi0_train,
        delta_test=delta_test,
    )
    return R_delta


def _resolve_h(exp_cfg, P_tr, bandwidth_mode=None):
    mode = bandwidth_mode if bandwidth_mode is not None else getattr(
        exp_cfg, "BANDWIDTH_MODE", "silverman"
    )
    return resolve_bandwidth(P_tr, mode=mode, min_bandwidth=exp_cfg.MIN_BANDWIDTH)


def _append_ope_stats(data_list, stats_file, stats_header_written):
    if not data_list:
        return stats_header_written

    df_ndelta = pd.DataFrame(data_list)
    group_cols = ["Train Delta", "Test Delta", "T", "Method"]
    grouped_mse = df_ndelta.groupby(group_cols)["MSE"]
    means_mse = grouped_mse.mean().reset_index()
    stds_mse = grouped_mse.std().reset_index()
    counts_mse = grouped_mse.count().reset_index()

    grouped_est = df_ndelta.groupby(group_cols)["Estimated_R_delta"]
    means_est = grouped_est.mean().reset_index()
    stds_est = grouped_est.std().reset_index()
    counts_est = grouped_est.count().reset_index()

    merged_mse = pd.merge(
        means_mse, stds_mse, on=group_cols, suffixes=("_mean", "_std")
    )
    merged_mse = pd.merge(merged_mse, counts_mse, on=group_cols)
    merged_mse.rename(columns={"MSE": "count_mse"}, inplace=True)

    merged_est = pd.merge(
        means_est, stds_est, on=group_cols, suffixes=("_mean", "_std")
    )
    merged_est = pd.merge(merged_est, counts_est, on=group_cols)
    merged_est.rename(columns={"Estimated_R_delta": "count_est"}, inplace=True)

    merged = pd.merge(merged_mse, merged_est, on=group_cols)

    def compute_ci_mse(row):
        n_count = row["count_mse"]
        if n_count <= 1:
            return row["MSE_mean"], row["MSE_mean"]
        se = row["MSE_std"] / (n_count ** 0.5)
        t_val = t_dist.ppf(0.95, df=n_count - 1)
        return row["MSE_mean"] - t_val * se, row["MSE_mean"] + t_val * se

    def compute_ci_est(row):
        n_count = row["count_est"]
        if n_count <= 1:
            return row["Estimated_R_delta_mean"], row["Estimated_R_delta_mean"]
        se = row["Estimated_R_delta_std"] / (n_count ** 0.5)
        t_val = t_dist.ppf(0.95, df=n_count - 1)
        return (
            row["Estimated_R_delta_mean"] - t_val * se,
            row["Estimated_R_delta_mean"] + t_val * se,
        )

    merged[["CI_Lower_MSE", "CI_Upper_MSE"]] = merged.apply(
        lambda row: pd.Series(compute_ci_mse(row)), axis=1
    )
    merged[["CI_Lower_R_delta", "CI_Upper_R_delta"]] = merged.apply(
        lambda row: pd.Series(compute_ci_est(row)), axis=1
    )

    output_df = merged[
        [
            "Train Delta",
            "Test Delta",
            "T",
            "Method",
            "MSE_mean",
            "CI_Lower_MSE",
            "CI_Upper_MSE",
            "Estimated_R_delta_mean",
            "CI_Lower_R_delta",
            "CI_Upper_R_delta",
        ]
    ].copy()
    output_df.rename(
        columns={
            "MSE_mean": "Mean_MSE",
            "Estimated_R_delta_mean": "Mean_Estimated_R_delta",
        },
        inplace=True,
    )
    output_df = output_df.sort_values(["Train Delta", "Test Delta", "T", "Method"])

    if not stats_header_written:
        output_df.to_csv(stats_file, index=False, mode="w")
        return True
    output_df.to_csv(stats_file, index=False, mode="a", header=False)
    return True


def run_ope_experiments(exp_cfg, bandwidth_mode=None, stats_file=None, results_file=None):
    """
    Run OPE: offline data size T, nuisances at delta_train, R_hat_delta at each delta_test.
    """
    _sync_core_config(exp_cfg)

    delta_train = exp_cfg.DELTA_TRAIN
    test_deltas = list(exp_cfg.TEST_DELTAS)
    train_sizes = list(exp_cfg.TRAIN_SIZES)
    n_repeats = exp_cfg.N_REPEATS

    if stats_file is None:
        from experiments.ope.ope_plotting import ope_stats_path

        stats_file = ope_stats_path(exp_cfg)
    if results_file is None:
        from experiments.ope.ope_plotting import ope_results_path

        results_file = ope_results_path(exp_cfg)

    print(f"{'=' * 60}")
    print(f"Running OPE Experiments | Repeats: {n_repeats}")
    print(f"Setting: {exp_cfg.SETTING}")
    print(f"δ_train (nuisance / cross-fit): {delta_train}")
    print(f"δ_test (evaluate R_δ): {test_deltas}")
    print(f"Offline sample sizes T: {train_sizes}")
    print(f"N_test (true R_δ, sampling)={exp_cfg.N_TEST}")
    if bandwidth_mode is not None:
        print(f"Bandwidth mode: {bandwidth_mode}")
    print("Comparing: LDR²PE-CP vs DRO-IPW")
    print(f"{'=' * 60}")

    target_policy = create_target_policy(exp_cfg)
    print("\n>>> Computing true R_delta at each test δ (nuisances at train δ)...")
    true_R_delta_dict = {}
    for delta_test in tqdm(test_deltas, desc="True R_delta"):
        true_R_delta = compute_true_R_delta(
            exp_cfg, target_policy, delta_train, delta_test
        )
        true_R_delta_dict[delta_test] = true_R_delta
        print(
            f"True R_delta (δ_train={delta_train}, δ_test={delta_test}) = {true_R_delta:.6f}"
        )

    total_experiments = len(test_deltas) * len(train_sizes) * n_repeats * 2
    pbar_total = tqdm(total=total_experiments, desc="Overall Progress", position=0)

    stats_header_written = False
    data_by_key = {}
    results_rows = []

    for delta_test in tqdm(test_deltas, desc="Test Delta", position=1, leave=False):
        true_R_delta = true_R_delta_dict[delta_test]

        for T in tqdm(
            train_sizes, desc=f"T (δ_test={delta_test})", leave=False, position=2
        ):
            key = (delta_train, delta_test, T)
            data_by_key.setdefault(key, [])

            for repeat in range(n_repeats):
                pbar_total.set_description(
                    f"δ_train={delta_train}, δ_test={delta_test}, T={T}, "
                    f"r={repeat + 1}/{n_repeats}"
                )
                seed = int(exp_cfg.SEED + delta_test * 10000 + T * 100 + repeat)
                set_global_seed(seed)

                env_train = PricingEnvironment(
                    setting=exp_cfg.SETTING, logging_policy_type="nonlinear", seed=seed
                )
                X_tr, P_tr, Y_tr = env_train.sample_data(T)
                X_tr = np.clip(np.nan_to_num(X_tr.astype(np.float64)), -20.0, 20.0)

                h = _resolve_h(exp_cfg, P_tr, bandwidth_mode)

                scaler = StandardScaler()
                X_tr_scaled = np.clip(scaler.fit_transform(X_tr), -20.0, 20.0)

                prop_model = PropensityEstimator()
                prop_model.fit(X_tr, P_tr)

                try:
                    R_ldr = evaluate_with_ldr2pe_cp(
                        exp_cfg,
                        target_policy,
                        X_tr,
                        P_tr,
                        Y_tr,
                        delta_train,
                        delta_test,
                        h,
                        scaler,
                    )
                    mse_ldr = (R_ldr - true_R_delta) ** 2
                except Exception as e:
                    print(
                        f"Warning: LDR²PE-CP failed T={T}, "
                        f"δ_test={delta_test}, r={repeat}: {e}"
                    )
                    R_ldr, mse_ldr = np.nan, np.nan

                try:
                    R_ipw = evaluate_with_dro_ipw(
                        exp_cfg,
                        target_policy,
                        X_tr,
                        P_tr,
                        Y_tr,
                        delta_train,
                        delta_test,
                        h,
                        prop_model,
                        scaler,
                    )
                    mse_ipw = (R_ipw - true_R_delta) ** 2
                except Exception as e:
                    print(
                        f"Warning: DRO-IPW failed T={T}, "
                        f"δ_test={delta_test}, r={repeat}: {e}"
                    )
                    R_ipw, mse_ipw = np.nan, np.nan

                for method, R_est, mse in [
                    ("LDR2PE-CP", R_ldr, mse_ldr),
                    ("DRO-IPW", R_ipw, mse_ipw),
                ]:
                    row = {
                        "Train Delta": delta_train,
                        "Test Delta": delta_test,
                        "T": T,
                        "Method": method,
                        "Repeat": repeat,
                        "Estimated_R_delta": R_est,
                        "True_R_delta": true_R_delta,
                        "MSE": mse,
                        "h": h,
                    }
                    if bandwidth_mode is not None:
                        row["Bandwidth"] = (
                            "silverman"
                            if isinstance(bandwidth_mode, str)
                            and bandwidth_mode.lower() == "silverman"
                            else str(float(bandwidth_mode))
                        )
                    results_rows.append(row)
                    data_by_key[key].append(row)
                    pbar_total.update(1)

            if data_by_key[key]:
                stats_header_written = _append_ope_stats(
                    data_by_key[key], stats_file, stats_header_written
                )
                data_by_key[key] = []

    pbar_total.close()

    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    print(f"Statistics saved incrementally to {stats_file}")

    return df_results, stats_file
