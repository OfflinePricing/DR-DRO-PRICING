#!/usr/bin/env python3
"""
OPL bandwidth sensitivity experiment on synthetic data.

Evaluates HatQ_DRO under kernel bandwidth:
  - Silverman: h = 1.06 * std(P) * n^{-1/5}
  - h = 1.06 * std(P) * n^{-1/4.9}
  - h = 1.06 * std(P) * n^{-1/2.9}
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from tqdm import tqdm

from experiments.opl import config_bandwidth_sensitivity as exp_cfg
from experiments.opl.opl_common import (
    evaluate_metrics,
    resolve_runtime,
    set_global_seed,
    train_models,
)
from experiments.opl.opl_plotting import (
    FIGURE2_TITLE_SUFFIX,
    bandwidth_sensitivity_plot_path,
    build_figure2_stats,
    figure2_stats_path,
    plot_opl_figure2_from_stats,
)

RESULTS_RAW = "OPL_bandwidth_sensitivity_raw.csv"


def _bandwidth_label(mode):
    if isinstance(mode, str) and mode.lower() == "silverman":
        return "silverman"
    return str(float(mode))


def plot_bandwidth_sensitivity_figures(df_stats):
    """One Figure-2-style PDF per bandwidth (same plotting as run_main_OPL_synthetic)."""
    for bw_label in df_stats["Bandwidth"].unique():
        sub = df_stats[df_stats["Bandwidth"] == bw_label].drop(columns=["Bandwidth"])
        stats_file = figure2_stats_path(exp_cfg, suffix=bw_label)
        sub.to_csv(stats_file, index=False)
        plot_opl_figure2_from_stats(
            stats_file=stats_file,
            save_path=bandwidth_sensitivity_plot_path(bw_label),
            title_suffix=FIGURE2_TITLE_SUFFIX,
        )


def run_bandwidth_sensitivity():
    runtime = resolve_runtime(exp_cfg)
    n_repeats = runtime["n_repeats"]
    train_sizes = runtime["train_sizes"]
    test_deltas = runtime["test_deltas"]
    rows = []

    print("=" * 60)
    print("OPL Bandwidth Sensitivity (synthetic, HatQ_DRO)")
    print(f"Modes: {exp_cfg.BANDWIDTH_MODES}")
    print(f"Train sizes: {train_sizes} | Test deltas: {test_deltas} | Repeats: {n_repeats}")
    print("=" * 60)

    for bw_mode in exp_cfg.BANDWIDTH_MODES:
        bw_label = _bandwidth_label(bw_mode)
        print(f"\n>>> Bandwidth mode: {bw_label}")

        for n_train in tqdm(train_sizes, desc=f"h={bw_label} | N", position=0):
            for r in tqdm(range(n_repeats), desc=f"N={n_train} repeats", leave=False, position=1):
                seed = exp_cfg.SEED + n_train * 100 + r
                set_global_seed(seed)

                models, scaler, prop, outcome_model, h_used = train_models(
                    exp_cfg,
                    runtime,
                    data_mode="synthetic",
                    n_train=n_train,
                    seed=seed,
                    bandwidth_mode=bw_mode,
                )

                for delta_test in test_deltas:
                    metrics = evaluate_metrics(
                        exp_cfg,
                        runtime,
                        "synthetic",
                        models,
                        scaler,
                        prop,
                        outcome_model,
                        n_test=runtime["n_test"],
                        delta_test=delta_test,
                    )
                    for m_name, vals in metrics.items():
                        rows.append(
                            {
                                "Bandwidth": bw_label,
                                "h": h_used,
                                "N": n_train,
                                "Delta Test": delta_test,
                                "Model": m_name,
                                "Repeat": r,
                                "HatQ_DRO": vals["HatQ_DRO"],
                            }
                        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_RAW, index=False)
    print(f"\nRaw results saved to {RESULTS_RAW}")

    stats_parts = []
    for bw_label, g in df.groupby("Bandwidth", sort=True):
        part = build_figure2_stats(g, metric_col="HatQ_DRO")
        part.insert(0, "Bandwidth", bw_label)
        stats_parts.append(part)

    df_stats = pd.concat(stats_parts, ignore_index=True)
    combined_stats_path = figure2_stats_path(exp_cfg, suffix="all_bandwidths")
    df_stats.to_csv(combined_stats_path, index=False)
    print(f"Combined statistics saved to {combined_stats_path}")

    plot_bandwidth_sensitivity_figures(df_stats)

    return df, df_stats


if __name__ == "__main__":
    run_bandwidth_sensitivity()
