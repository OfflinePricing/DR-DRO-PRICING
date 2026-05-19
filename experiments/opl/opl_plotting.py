"""
Shared OPL Figure 2 plotting (uses src.visualization.plot_dropl_vs_n_from_stats).
"""

import os

import pandas as pd
from scipy.stats import t as t_dist

from src.visualization import plot_dropl_vs_n_from_stats

FIGURE2_TITLE_SUFFIX = " (Multiple Delta Test Values)"


def figure2_stats_path(exp_cfg, suffix=None):
    """Path for HatQ_DRO statistics CSV (same layout as main synthetic OPL run)."""
    base = f"Figure2_Statistics_HatQ_DRO_{exp_cfg.SETTING}_{exp_cfg.DATA_TAG}"
    if suffix is not None:
        return f"{base}_{suffix}.csv"
    return f"{base}.csv"


def figure2_plot_path(exp_cfg, suffix=None):
    """Default Figure 2 PDF path for synthetic OPL."""
    base = f"Figure2_DROPL_vs_N_Q_DRO_{exp_cfg.SETTING}"
    if suffix is not None:
        return f"{base}_{suffix}.pdf"
    return f"{base}.pdf"


def bandwidth_sensitivity_plot_path(bw_label):
    """PDF path for bandwidth sensitivity sub-experiments."""
    return f"OPL_bandwidth_sensitivity_{bw_label}.pdf"


def build_figure2_stats(df_raw, metric_col="HatQ_DRO"):
    """
    Aggregate repeat-level results into Figure 2 statistics format.

    Returns DataFrame with columns:
    Delta Test, T, Model, Mean, CI_Lower, CI_Upper
    """
    if "Delta Test" not in df_raw.columns:
        raise ValueError("df_raw must contain a 'Delta Test' column.")

    parts = []
    for delta_test, g in df_raw.groupby("Delta Test", sort=True):
        grouped = g.groupby(["N", "Model"])[metric_col]
        means = grouped.mean().reset_index()
        stds = grouped.std().reset_index()
        counts = grouped.count().reset_index()
        merged = pd.merge(means, stds, on=["N", "Model"], suffixes=("_mean", "_std"))
        merged = pd.merge(merged, counts, on=["N", "Model"])
        merged.rename(columns={metric_col: "count"}, inplace=True)

        def compute_ci(row):
            n = row["count"]
            if n <= 1:
                return row[f"{metric_col}_mean"], row[f"{metric_col}_mean"]
            se = row[f"{metric_col}_std"] / (n ** 0.5)
            t_val = t_dist.ppf(0.95, df=n - 1)
            return (
                row[f"{metric_col}_mean"] - t_val * se,
                row[f"{metric_col}_mean"] + t_val * se,
            )

        merged[["CI_Lower", "CI_Upper"]] = merged.apply(
            lambda row: pd.Series(compute_ci(row)), axis=1
        )
        out = merged[["N", "Model", f"{metric_col}_mean", "CI_Lower", "CI_Upper"]].copy()
        out.rename(columns={f"{metric_col}_mean": "Mean"}, inplace=True)
        out.insert(0, "Delta Test", delta_test)
        out.rename(columns={"N": "T"}, inplace=True)
        parts.append(out)

    return pd.concat(parts, ignore_index=True).sort_values(
        ["Delta Test", "T", "Model"]
    )


def plot_opl_figure2_from_stats(
    stats_file,
    save_path,
    title_suffix=FIGURE2_TITLE_SUFFIX,
):
    """
    Plot Kallus Figure 2 style curves from a statistics CSV.

    Same entry point as run_main_OPL_synthetic.plot_synthetic_figure2.
    """
    if not os.path.isfile(stats_file):
        print(f"[Visualization] Skipped: statistics file not found: {stats_file}")
        return False

    df_stats = pd.read_csv(stats_file)
    plot_dropl_vs_n_from_stats(df_stats, title_suffix=title_suffix, save_path=save_path)
    return True
