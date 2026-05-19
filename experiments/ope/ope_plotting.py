"""
OPE Figure 1 plotting via src.visualization (read stats CSV → plot → save PDF).
"""

import os

import pandas as pd

from src.visualization import plot_ope_figure1_from_stats


def ope_stats_path(exp_cfg, suffix=None):
    base = f"OPE_Statistics_{exp_cfg.SETTING}"
    if suffix is not None:
        return f"{base}_{suffix}.csv"
    return f"{base}.csv"


def ope_results_path(exp_cfg):
    return f"OPE_Results_{exp_cfg.SETTING}.csv"


def ope_figure1_path(exp_cfg, suffix=None):
    base = f"OPE_Figure1_{exp_cfg.SETTING}"
    if suffix is not None:
        return f"{base}_{suffix}.pdf"
    return f"{base}.pdf"


def bandwidth_sensitivity_plot_path(bw_label):
    return f"OPE_bandwidth_sensitivity_{bw_label}.pdf"


def _stats_for_visualization(df_stats):
    """Map OPE column names to those expected by src.visualization."""
    df = df_stats.copy()
    if "Test Delta" in df.columns and "Delta Test" not in df.columns:
        df = df.rename(columns={"Test Delta": "Delta Test"})
    return df


def regenerate_ope_figure1(stats_file, save_path):
    """Regenerate OPE Figure 1 from statistics CSV."""
    print("=" * 60)
    print("Regenerate OPE Figure 1: MSE vs T")
    print("=" * 60)

    if not os.path.isfile(stats_file):
        print(f"Error: statistics file not found: {stats_file}")
        return False

    print(f"Reading statistics file: {stats_file}")
    df_stats = pd.read_csv(stats_file)
    print(f"Shape: {df_stats.shape}")
    print(f"Columns: {df_stats.columns.tolist()}")

    print(f"\nGenerating figure: {save_path}")
    plot_ope_figure1_from_stats(_stats_for_visualization(df_stats), save_path=save_path)

    print(f"\nOPE Figure 1 saved to: {save_path}")
    return True
