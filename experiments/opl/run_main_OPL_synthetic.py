#!/usr/bin/env python3
"""OPL experiments on synthetic data (evaluates HatQ_DRO only; no worst-case attacks)."""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.opl import config_synthetic as exp_cfg
from experiments.opl.opl_common import run_experiments
from experiments.opl.opl_plotting import (
    figure2_plot_path,
    figure2_stats_path,
    plot_opl_figure2_from_stats,
)


def plot_synthetic_figure2():
    """Plot Figure 2 from incremental statistics CSV (see src.visualization.py)."""
    plot_opl_figure2_from_stats(
        stats_file=figure2_stats_path(exp_cfg),
        save_path=figure2_plot_path(exp_cfg),
    )


if __name__ == "__main__":
    run_experiments(exp_cfg, data_mode="synthetic")
    plot_synthetic_figure2()
