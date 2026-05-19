#!/usr/bin/env python3
"""
OPE experiments on synthetic data.

Nuisance / cross-fitting at δ_train=0.2; R_δ estimated at δ_test ∈ {0.1, 0.2, 0.3}.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.ope import config_ope as exp_cfg
from experiments.ope.ope_common import run_ope_experiments
from experiments.ope.ope_plotting import (
    ope_figure2_path,
    ope_stats_path,
    regenerate_ope_figure2,
)


if __name__ == "__main__":
    run_ope_experiments(exp_cfg)
    regenerate_ope_figure2(
        stats_file=ope_stats_path(exp_cfg),
        save_path=ope_figure2_path(exp_cfg),
    )
