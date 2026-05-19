#!/usr/bin/env python3
"""
OPE bandwidth sensitivity experiment.

Evaluates MSE vs true R_delta under kernel bandwidth:
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

from experiments.ope import config_bandwidth_sensitivity as exp_cfg
from experiments.ope.ope_common import run_ope_experiments
from experiments.ope.ope_plotting import (
    bandwidth_sensitivity_plot_path,
    ope_stats_path,
    regenerate_ope_figure2,
)

RESULTS_RAW = "OPE_bandwidth_sensitivity_raw.csv"


def _bandwidth_label(mode):
    if isinstance(mode, str) and mode.lower() == "silverman":
        return "silverman"
    return str(float(mode))


def run_bandwidth_sensitivity():
    all_results = []

    print("=" * 60)
    print("OPE Bandwidth Sensitivity")
    print(f"Modes: {exp_cfg.BANDWIDTH_MODES}")
    print(f"δ_train: {exp_cfg.DELTA_TRAIN} | δ_test: {exp_cfg.TEST_DELTAS} | T: {exp_cfg.TRAIN_SIZES}")
    print("=" * 60)

    for bw_mode in exp_cfg.BANDWIDTH_MODES:
        bw_label = _bandwidth_label(bw_mode)
        print(f"\n>>> Bandwidth mode: {bw_label}")

        stats_file = ope_stats_path(exp_cfg, suffix=bw_label)
        results_file = f"OPE_Results_{exp_cfg.SETTING}_{bw_label}.csv"

        df, _ = run_ope_experiments(
            exp_cfg,
            bandwidth_mode=bw_mode,
            stats_file=stats_file,
            results_file=results_file,
        )
        all_results.append(df)

        regenerate_ope_figure2(
            stats_file=stats_file,
            save_path=bandwidth_sensitivity_plot_path(bw_label),
        )

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(RESULTS_RAW, index=False)
    print(f"\nCombined raw results saved to {RESULTS_RAW}")

    return df_all


if __name__ == "__main__":
    run_bandwidth_sensitivity()
