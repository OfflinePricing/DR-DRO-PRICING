"""
Shared OPL Figure 3 plotting (uses src.visualization.plot_dropl_vs_n_from_stats).
"""

import os

import pandas as pd
from scipy.stats import t as t_dist

from src.visualization import plot_dropl_vs_n_from_stats

FIGURE3_TITLE_SUFFIX = " (Multiple Delta Test Values)"


def figure3_stats_path(exp_cfg, suffix=None):
    """Path for HatQ_DRO statistics CSV (same layout as main synthetic OPL run)."""
    base = f"Figure3_Statistics_HatQ_DRO_{exp_cfg.SETTING}_{exp_cfg.DATA_TAG}"
    if suffix is not None:
        return f"{base}_{suffix}.csv"
    return f"{base}.csv"


def figure3_plot_path(exp_cfg, suffix=None):
    """Default Figure 3 PDF path for synthetic OPL."""
    base = f"Figure3_DROPL_vs_N_Q_DRO_{exp_cfg.SETTING}"
    if suffix is not None:
        return f"{base}_{suffix}.pdf"
    return f"{base}.pdf"


def bandwidth_sensitivity_plot_path(bw_label):
    """PDF path for bandwidth sensitivity sub-experiments."""
    return f"OPL_bandwidth_sensitivity_{bw_label}.pdf"


def build_figure3_stats(df_raw, metric_col="HatQ_DRO"):
    """
    Aggregate repeat-level results into Figure 3 statistics format.

    Returns DataFrame with columns:
    Delta Test, T, Model, Mean, CI_Lower, CI_Upper
    """
    if "Delta Test" not in df_raw.columns:
        raise ValueError("df_raw must contain a 'Delta Test' column.")

    stats_parts = []
    for delta_test, g in df_raw.groupby("Delta Test", sort=True):
        grouped = g.groupby(["N", "Model"])[metric_col]
        means = grouped.mean().reset_index()
        stds = grouped.std().reset_index()
        counts = grouped.count().reset_index()
        merged = pd.merge(means, stds, on=["N", "Model"], suffixes=("_mean", "_std"))
        merged = pd.merge(merged, counts, on=["N", "Model"])
        merged.rename(columns={metric_col: "count"}, inplace=True)

        def compute_ci(row):
            n_count = row["count"]
            if n_count <= 1:
                return row[metric_col + "_mean"], row[metric_col + "_mean"]
            se = row[metric_col + "_std"] / (n_count ** 0.5)
            t_val = t_dist.ppf(0.95, df=n_count - 1)
            mean = row[metric_col + "_mean"]
            return mean - t_val * se, mean + t_val * se

        merged[["CI_Lower", "CI_Upper"]] = merged.apply(
            lambda row: pd.Series(compute_ci(row)), axis=1
        )
        out = merged[
            ["N", "Model", metric_col + "_mean", "CI_Lower", "CI_Upper"]
        ].copy()
        out.rename(columns={"N": "T", metric_col + "_mean": "Mean"}, inplace=True)
        out.insert(0, "Delta Test", delta_test)
        stats_parts.append(out)

    return pd.concat(stats_parts, ignore_index=True).sort_values(
        ["Delta Test", "T", "Model"]
    )


def plot_opl_figure3_from_stats(
    stats_file,
    save_path,
    title_suffix=FIGURE3_TITLE_SUFFIX,
):
    """
    Plot Kallus Figure 3 style curves from a statistics CSV.

    Same entry point as run_main_OPL_synthetic.plot_synthetic_figure3.
    """
    if not os.path.isfile(stats_file):
        print(f"[Visualization] Skipped: statistics file not found: {stats_file}")
        return False

    df_stats = pd.read_csv(stats_file)
    plot_dropl_vs_n_from_stats(
        df_stats,
        title_suffix=title_suffix,
        save_path=save_path,
    )
    return True
