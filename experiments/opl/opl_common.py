"""
Shared OPL experiment logic for synthetic and Expedia runs.
"""

import os
import random
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as t_dist
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src import config
from src.dgp import PricingEnvironment
from src.estimators import (
    DifferentiableOutcomeEstimator,
    PropensityEstimator,
    resolve_bandwidth,
)
from src.learner import Ai2026Learner, CDROLearner, Leung2025Learner
from src.policy import LinearPolicy, MLPPolicy
from src.realdata_expedia import get_shifted_split, load_expedia_train_triplets

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.linear_model\._ridge")

_performance_log = defaultdict(list)
_expedia_global_split = None


@contextmanager
def timer(operation_name):
    start = time.time()
    try:
        yield
    finally:
        _performance_log[operation_name].append(time.time() - start)


def print_performance_summary():
    if not _performance_log:
        return
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    for op_name, times in _performance_log.items():
        total = sum(times)
        mean = np.mean(times)
        count = len(times)
        print(f"{op_name:40s}: {count:4d} calls, {total:8.2f}s total, {mean:6.3f}s avg")
    print("=" * 60)


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


def _sync_core_config(exp_cfg):
    """Mirror OPL experiment settings into src.config for learner/policy code."""
    config.SEED = exp_cfg.SEED
    config.MIN_BANDWIDTH = exp_cfg.MIN_BANDWIDTH
    config.WARM_START_LR = exp_cfg.WARM_START_LR
    config.WARM_START_EPOCHS = exp_cfg.WARM_START_EPOCHS
    config.EPOCHS = exp_cfg.EPOCHS
    config.ALPHA_UPDATE_FREQ = exp_cfg.ALPHA_UPDATE_FREQ
    config.GRAD_CLIP_MAX_NORM = exp_cfg.GRAD_CLIP_MAX_NORM
    config.TEST_N_ESTIMATORS = exp_cfg.FOREST_N_ESTIMATORS
    config.TEST_FOREST_MAX_DEPTH = exp_cfg.FOREST_MAX_DEPTH
    if hasattr(exp_cfg, "TRAIN_Z_LOC"):
        config.TRAIN_Z_LOC = exp_cfg.TRAIN_Z_LOC
        config.TRAIN_Z_SCALE = exp_cfg.TRAIN_Z_SCALE


def resolve_runtime(exp_cfg):
    """Build runtime knobs from experiment config."""
    _sync_core_config(exp_cfg)

    train_sizes = list(exp_cfg.TRAIN_SIZES)
    test_deltas = list(exp_cfg.TEST_DELTAS)

    if getattr(exp_cfg, "EXPEDIA_INCLUDE_DELTA0_BASELINE", False) and 0.0 not in test_deltas:
        test_deltas = [0.0] + test_deltas

    return {
        "n_repeats": exp_cfg.N_REPEATS,
        "train_sizes": train_sizes,
        "test_deltas": test_deltas,
        "n_test": exp_cfg.N_TEST,
        "n_test_attack": exp_cfg.N_TEST_ATTACK,
        "base_seed": exp_cfg.BASE_SEED,
        "nuisance_epochs": exp_cfg.NUISANCE_EPOCHS,
        "m_attacks": exp_cfg.M_ATTACKS,
        "n_folds": exp_cfg.N_FOLDS,
        "batch_size": exp_cfg.BATCH_SIZE,
        "epochs": exp_cfg.EPOCHS,
        "epochs_step": exp_cfg.EPOCHS_STEP,
        "alpha_update_freq": exp_cfg.ALPHA_UPDATE_FREQ,
        "warm_start_lr": exp_cfg.WARM_START_LR,
        "warm_start_epochs": exp_cfg.WARM_START_EPOCHS,
        "grad_clip_max_norm": exp_cfg.GRAD_CLIP_MAX_NORM,
    }


def _get_expedia_split(exp_cfg):
    global _expedia_global_split
    if _expedia_global_split is not None:
        return _expedia_global_split

    data = load_expedia_train_triplets(
        exp_cfg.EXPEDIA_TRAIN_CSV,
        max_rows=exp_cfg.EXPEDIA_MAX_ROWS,
        random_seed=exp_cfg.EXPEDIA_SPLIT_SEED,
        reward_mode=exp_cfg.EXPEDIA_REWARD_MODE,
    )
    split = get_shifted_split(
        data,
        split_mode=exp_cfg.EXPEDIA_SPLIT_MODE,
        split_config=exp_cfg.EXPEDIA_SPLIT_CONFIG,
    )
    _expedia_global_split = split
    return split


def _set_price_range(exp_cfg, data_mode, P_train=None):
    if data_mode == "expedia":
        if P_train is None or len(P_train) == 0:
            raise ValueError("P_train is required for Expedia price range.")
        p_low = float(np.percentile(P_train, exp_cfg.EXPEDIA_PRICE_LOWER_Q))
        p_high = float(np.percentile(P_train, exp_cfg.EXPEDIA_PRICE_UPPER_Q))
        if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
            p_low, p_high = float(np.min(P_train)), float(np.max(P_train))
        if p_high <= p_low:
            p_low, p_high = 1.0, 500.0
        config.PRICE_MIN = p_low
        config.PRICE_MAX = p_high
    else:
        config.PRICE_MIN = exp_cfg.PRICE_MIN
        config.PRICE_MAX = exp_cfg.PRICE_MAX


def train_models(exp_cfg, runtime, data_mode, n_train, seed, bandwidth_mode=None):
    with timer("train_models.total"):
        if data_mode == "expedia":
            split = _get_expedia_split(exp_cfg)
            tr = split["train"]
            n_eff = min(n_train, len(tr["X"]))
            X_tr_raw = tr["X"][:n_eff]
            P_tr = tr["P"][:n_eff]
            Y_tr = tr["Y"][:n_eff]
        else:
            env_train = PricingEnvironment(
                setting=exp_cfg.SETTING,
                logging_policy_type="nonlinear",
                seed=seed,
            )
            X_tr_raw, P_tr, Y_tr = env_train.sample_data(n_train)

        _set_price_range(exp_cfg, data_mode, P_tr)

        X_tr_raw = np.clip(np.nan_to_num(X_tr_raw.astype(np.float64)), -20.0, 20.0)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_tr_scaled = np.clip(X_tr_scaled, -20.0, 20.0)

        prop_model = PropensityEstimator()
        prop_model.fit(X_tr_raw, P_tr)
        Pi0_tr = prop_model.predict(X_tr_raw, P_tr)
        bw_mode = (
            bandwidth_mode
            if bandwidth_mode is not None
            else getattr(exp_cfg, "BANDWIDTH_MODE", "silverman")
        )
        h = resolve_bandwidth(P_tr, mode=bw_mode, min_bandwidth=exp_cfg.MIN_BANDWIDTH)

        input_dim = X_tr_scaled.shape[1]
        outcome_model = DifferentiableOutcomeEstimator(input_dim=input_dim)
        outcome_model.fit(
            X_tr_scaled, P_tr, Y_tr, epochs=runtime["nuisance_epochs"]
        )

        from src.estimators import ProbabilityOutcomeEstimator

        prob_outcome_model = ProbabilityOutcomeEstimator(input_dim=input_dim)
        prob_outcome_model.fit(
            X_tr_scaled, P_tr, Y_tr, epochs=runtime["nuisance_epochs"]
        )

        PolicyClass = LinearPolicy if exp_cfg.POLICY_TYPE == "linear" else MLPPolicy
        learners = {}
        # Ai2026: non-robust training (no δ); eval uses delta_test in get_optimal_alpha only.
        learners["Ai2026"] = Ai2026Learner(PolicyClass(input_dim), outcome_model, h)
        delta_train = exp_cfg.DELTA_TRAIN
        learners[f"Leung2025(δ_train={delta_train})"] = Leung2025Learner(
            PolicyClass(input_dim),
            prob_outcome_model,
            h,
            delta=delta_train,
        )
        learners[f"DRO(δ_train={delta_train})"] = CDROLearner(
            PolicyClass(input_dim),
            h,
            delta=delta_train,
            X_train=X_tr_scaled,
            P_train=P_tr,
            Y_train=Y_tr,
            n_folds=runtime["n_folds"],
            alpha_update_freq=runtime["alpha_update_freq"],
        )

        X_tensor = torch.FloatTensor(X_tr_scaled)
        P_tensor = torch.FloatTensor(P_tr).view(-1, 1)
        warm_lr = runtime["warm_start_lr"]
        warm_epochs = runtime["warm_start_epochs"]
        grad_clip = runtime["grad_clip_max_norm"]
        for learner in learners.values():
            opt = torch.optim.Adam(learner.policy.parameters(), lr=warm_lr)
            warm_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=warm_epochs, eta_min=warm_lr * 0.1
            )
            for _ in range(warm_epochs):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(learner.policy(X_tensor), P_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(learner.policy.parameters(), max_norm=grad_clip)
                opt.step()
                warm_scheduler.step()

        batch_size = min(runtime["batch_size"], max(64, n_train // 10))
        indices = np.arange(len(X_tr_scaled))

        epochs_to_use = runtime["epochs"] + max(0, (n_train - 500) // 500) * runtime["epochs_step"]

        with timer("train_models.training_loop"):
            for epoch in tqdm(range(epochs_to_use), desc="Training Epochs", leave=False):
                np.random.shuffle(indices)
                for learner in learners.values():
                    if hasattr(learner, "set_epoch"):
                        learner.set_epoch(epoch)
                for start_idx in range(0, len(indices), batch_size):
                    idx = indices[start_idx : min(start_idx + batch_size, len(indices))]
                    batch = (
                        X_tr_scaled[idx],
                        P_tr[idx],
                        Y_tr[idx],
                        Pi0_tr[idx],
                    )
                    for learner in learners.values():
                        learner.train_step(*batch)
                for learner in learners.values():
                    if hasattr(learner, "step_scheduler"):
                        learner.step_scheduler()

    return learners, scaler, prop_model, outcome_model, h


def evaluate_metrics(
    exp_cfg,
    runtime,
    data_mode,
    models,
    scaler,
    prop_model,
    outcome_model,
    n_test,
    delta_test,
    M=None,
    expedia_test_data=None,
):
    base_seed = runtime["base_seed"]
    if data_mode == "synthetic":
        results = {name: {"HatQ_DRO": 0.0} for name in models}
    else:
        results = {name: {"Hat Q_min": 0.0} for name in models}

    if data_mode == "synthetic":
        with timer("evaluate_metrics.q_dro"):
            env_base = PricingEnvironment(setting=exp_cfg.SETTING, seed=base_seed)
            X_base, P_base, Y_base = env_base.sample_data(n_test)

            X_base_scaled = np.clip(
                scaler.transform(np.clip(np.nan_to_num(X_base), -20, 20)), -5, 5
            )
            Pi0_base = prop_model.predict(X_base, P_base)

            # Same Hat Q_DRO metric for all learners; δ enters only here via delta_test
            # (Ai2026 training is non-robust; see Ai2026Learner docstring).
            for name, learner in models.items():
                with timer("evaluate_metrics.get_optimal_alpha"):
                    _, q_dro = learner.get_optimal_alpha(
                        X_base_scaled, P_base, Y_base, Pi0_base, delta_test
                    )
                results[name]["HatQ_DRO"] = q_dro
        return results

    if M is None:
        M = runtime["m_attacks"]
    n_test_attack = runtime["n_test_attack"]
    attack_logs = {name: np.full(M, np.inf, dtype=np.float32) for name in models}
    alpha_warm_start = {name: None for name in models}

    with timer("evaluate_metrics.attacks"):
        for i in tqdm(range(M), desc="Worst-case Attacks", leave=False):
            curr_seed = base_seed + 1 + i
            te = (
                expedia_test_data
                if expedia_test_data is not None
                else _get_expedia_split(exp_cfg)["test"]
            )
            n_eff = min(n_test_attack, len(te["X"]))
            rng_attack = np.random.RandomState(curr_seed)
            idx_boot = rng_attack.choice(len(te["X"]), size=n_eff, replace=True)
            X_i, P_i, Y_i = te["X"][idx_boot], te["P"][idx_boot], te["Y"][idx_boot]

            X_i_raw = np.clip(np.nan_to_num(X_i), -20, 20)
            X_i_scaled = np.clip(scaler.transform(X_i_raw), -5, 5)
            Pi0_i = prop_model.predict(X_i_raw, P_i)

            for name, learner in models.items():
                with timer("evaluate_metrics.get_optimal_alpha"):
                    alpha_star, _ = learner.get_optimal_alpha(
                        X_i_scaled,
                        P_i,
                        Y_i,
                        Pi0_i,
                        delta_test,
                        alpha_init=alpha_warm_start[name],
                    )
                    alpha_warm_start[name] = alpha_star

                weights = learner.get_worst_case_weights(
                    X_i_scaled, P_i, Y_i, Pi0_i, alpha_star
                )
                cumsum = np.cumsum(weights)
                cumsum = cumsum / cumsum[-1]
                u = np.random.rand(len(X_i))
                attack_indices = np.searchsorted(cumsum, u)

                X_attacked_raw = X_i_raw[attack_indices]
                X_attacked_scaled = X_i_scaled[attack_indices]
                with torch.no_grad():
                    pred_P = (
                        learner.policy(torch.FloatTensor(X_attacked_scaled))
                        .numpy()
                        .flatten()
                    )

                with torch.no_grad():
                    revenue = (
                        outcome_model(
                            torch.FloatTensor(X_attacked_scaled),
                            torch.FloatTensor(pred_P).view(-1, 1),
                        )
                        .numpy()
                        .flatten()
                    )

                attack_logs[name][i] = np.mean(revenue)

    for name in models:
        results[name]["Hat Q_min"] = np.min(attack_logs[name])

    return results


def print_pivot_table(
    df,
    index_col,
    value_cols=("HatQ_DRO", "Hat Q_min"),
    format_style="leung2025",
    n_repeats=None,
    save_path=None,
    model_order=None,
):
    df_mean = df.groupby([index_col, "Model"])[list(value_cols)].mean().reset_index()
    df_std = df.groupby([index_col, "Model"])[list(value_cols)].std().reset_index()
    merged = pd.merge(df_mean, df_std, on=[index_col, "Model"], suffixes=("_mean", "_std"))

    for col in value_cols:
        if format_style == "leung2025":
            if n_repeats is not None and n_repeats > 1:
                merged[col + "_se"] = merged[col + "_std"] / np.sqrt(n_repeats)
                merged[col] = merged.apply(
                    lambda x, c=col: (
                        f"{x[c+'_mean']:.2f}±{x[c+'_se']/x[c+'_mean']*100:.2f}"
                        if x[c + "_mean"] > 1e-6
                        else f"{x[c+'_mean']:.2f}±0.00"
                    ),
                    axis=1,
                )
            else:
                merged[col] = merged.apply(
                    lambda x, c=col: (
                        f"{x[c+'_mean']:.2f}±{x[c+'_std']/x[c+'_mean']*100:.2f}"
                        if x[c + "_mean"] > 1e-6
                        else f"{x[c+'_mean']:.2f}±0.00"
                    ),
                    axis=1,
                )
        else:
            merged[col] = merged.apply(
                lambda x, c=col: f"{x[c+'_mean']:.3f} ({x[c+'_std']:.3f})",
                axis=1,
            )

    if model_order is not None:
        merged["Model"] = pd.Categorical(
            merged["Model"], categories=model_order, ordered=True
        )
        merged = merged.sort_values("Model")

    pivot = merged.pivot(index=index_col, columns="Model", values=value_cols)
    print(pivot.to_string())
    if save_path:
        pivot.to_csv(save_path, sep="\t")
        print(f"\nResults saved to {save_path}")
    return pivot


def run_experiments(exp_cfg, data_mode):
    global _expedia_global_split
    _expedia_global_split = None

    runtime = resolve_runtime(exp_cfg)
    n_repeats = runtime["n_repeats"]
    train_sizes = runtime["train_sizes"]
    test_deltas = runtime["test_deltas"]

    print(f"{'=' * 60}")
    print(f"Running OPL Experiments | Repeats: {n_repeats} | Data: {data_mode}")
    print(f"Setting: {exp_cfg.SETTING} | Policy: {exp_cfg.POLICY_TYPE}")
    print(f"Train δ={exp_cfg.DELTA_TRAIN} | Train sizes: {train_sizes} | N_test={runtime['n_test']}")
    if data_mode == "expedia":
        print(f"Expedia train CSV: {exp_cfg.EXPEDIA_TRAIN_CSV}")
        print(f"Reward mode: {exp_cfg.EXPEDIA_REWARD_MODE} | Split: {exp_cfg.EXPEDIA_SPLIT_MODE}")
        print(
            f"Price quantiles: Q{exp_cfg.EXPEDIA_PRICE_LOWER_Q}~Q{exp_cfg.EXPEDIA_PRICE_UPPER_Q}"
        )
    print(
        f"Comparing: Ai2026, Leung2025 (δ_train={exp_cfg.DELTA_TRAIN}), "
        f"DRO (δ_train={exp_cfg.DELTA_TRAIN})"
    )
    print(f"{'=' * 60}")

    metric_name = "HatQ_DRO" if data_mode == "synthetic" else "Hat Q_min"
    print(f"\n>>> Experiment 1: Varying training size N | metric: {metric_name}")
    print(f"Evaluation δ values: {test_deltas}")

    res_table3 = []
    if data_mode == "synthetic":
        stats_file = f"Figure2_Statistics_HatQ_DRO_{exp_cfg.SETTING}_{exp_cfg.DATA_TAG}.csv"
    else:
        stats_file = f"Figure2_Statistics_HatQmin_{exp_cfg.SETTING}_{exp_cfg.DATA_TAG}.csv"
    stats_header_written = False
    data_by_ndelta = {}

    split_mode_label = (
        exp_cfg.EXPEDIA_SPLIT_MODE if data_mode == "expedia" else "synthetic"
    )

    for n_train in tqdm(train_sizes, desc="Training Size N", position=0):
        for r in tqdm(range(n_repeats), desc=f"N={n_train} Repeats", leave=False, position=1):
            seed = exp_cfg.SEED + n_train * 100 + r
            set_global_seed(seed)

            models, scaler, prop, outcome_model, _ = train_models(
                exp_cfg, runtime, data_mode, n_train, seed
            )
            if r == 0:
                print(
                    f"Policy price range in this run: [{config.PRICE_MIN:.3f}, {config.PRICE_MAX:.3f}]"
                )

            for delta_test in test_deltas:
                n_test_val = runtime["n_test"]
                expedia_test_data = None
                if data_mode == "expedia":
                    expedia_test_data = _get_expedia_split(exp_cfg)["test"]
                    n_test_val = min(n_test_val, len(expedia_test_data["X"]))

                metrics = evaluate_metrics(
                    exp_cfg,
                    runtime,
                    data_mode,
                    models,
                    scaler,
                    prop,
                    outcome_model,
                    n_test=n_test_val,
                    delta_test=delta_test,
                    M=runtime["m_attacks"],
                    expedia_test_data=expedia_test_data,
                )

                for m_name, vals in metrics.items():
                    data_point = {
                        "N": n_train,
                        "Delta Test": delta_test,
                        "Model": m_name,
                        "Repeat": r,
                        metric_name: vals[metric_name],
                        "DataSource": exp_cfg.DATA_TAG,
                        "SplitMode": split_mode_label,
                    }
                    res_table3.append(data_point)
                    key = (n_train, delta_test)
                    data_by_ndelta.setdefault(key, []).append(data_point)

            if r == n_repeats - 1:
                for current_delta in test_deltas:
                    key = (n_train, current_delta)
                    if key not in data_by_ndelta or not data_by_ndelta[key]:
                        continue
                    df_ndelta = pd.DataFrame(data_by_ndelta[key])

                    def build_stats(metric_name):
                        grouped = df_ndelta.groupby(["Delta Test", "N", "Model"])[
                            metric_name
                        ]
                        means = grouped.mean().reset_index()
                        stds = grouped.std().reset_index()
                        counts = grouped.count().reset_index()
                        merged = pd.merge(
                            means, stds, on=["Delta Test", "N", "Model"], suffixes=("_mean", "_std")
                        )
                        merged = pd.merge(
                            merged, counts, on=["Delta Test", "N", "Model"]
                        )
                        merged.rename(columns={metric_name: "count"}, inplace=True)

                        def compute_ci(row):
                            n_count = row["count"]
                            if n_count <= 1:
                                return row[metric_name + "_mean"], row[metric_name + "_mean"]
                            se = row[metric_name + "_std"] / np.sqrt(n_count)
                            t_val = t_dist.ppf(0.95, df=n_count - 1)
                            lower = row[metric_name + "_mean"] - t_val * se
                            upper = row[metric_name + "_mean"] + t_val * se
                            return lower, upper

                        merged[["CI_Lower", "CI_Upper"]] = merged.apply(
                            lambda row: pd.Series(compute_ci(row)), axis=1
                        )
                        output_df = merged[
                            [
                                "Delta Test",
                                "N",
                                "Model",
                                metric_name + "_mean",
                                "CI_Lower",
                                "CI_Upper",
                            ]
                        ].copy()
                        output_df.rename(
                            columns={"N": "T", metric_name + "_mean": "Mean"}, inplace=True
                        )
                        return output_df.sort_values(["Delta Test", "T", "Model"])

                    output_df = build_stats(metric_name)
                    if not stats_header_written:
                        output_df.to_csv(stats_file, index=False, mode="w")
                        stats_header_written = True
                    else:
                        output_df.to_csv(stats_file, index=False, mode="a", header=False)

    df_table3 = pd.DataFrame(res_table3)

    if data_mode == "expedia":
        expedia_results_file = (
            f"OPL_Expedia_Figure2_{exp_cfg.SETTING}_{exp_cfg.EXPEDIA_SPLIT_MODE}.csv"
        )
        df_table3.to_csv(expedia_results_file, index=False)
        print(f"Expedia results saved to {expedia_results_file}")

    print(f"\nStatistics saved incrementally to {stats_file}")

    print_performance_summary()
