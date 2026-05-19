#!/usr/bin/env python3
"""OPL experiments on Expedia real data (evaluates Hat Q_min via worst-case attacks only)."""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from experiments.opl import config_expedia as exp_cfg
from experiments.opl.opl_common import run_experiments
from src.visualization import plot_dropl_vs_n_from_stats


def plot_expedia_figure2():
    """Plot Hat Q_min vs N from statistics CSV (same layout as synthetic Figure 2)."""
    stats_file = f"Figure2_Statistics_HatQmin_{exp_cfg.SETTING}_{exp_cfg.DATA_TAG}.csv"
    if not os.path.isfile(stats_file):
        print(f"[Visualization] Skipped: statistics file not found: {stats_file}")
        return

    df_stats = pd.read_csv(stats_file)
    save_path = f"Figure2_DROPL_vs_N_Q_min_{exp_cfg.SETTING}_hotel_type.pdf"
    plot_dropl_vs_n_from_stats(
        df_stats,
        title_suffix=" (Hat Q_min, worst-case attacks)",
        save_path=save_path,
    )


if __name__ == "__main__":
    run_experiments(exp_cfg, data_mode="expedia")
    plot_expedia_figure2()
