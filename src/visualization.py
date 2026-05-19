
# visualization.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# 1. 启用 LaTeX 渲染（这样字体就全是 Type-1 了）
plt.rcParams['text.usetex'] = True

# --- 必须添加以下两行来规避 Type-3 字体问题 ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 为了让图片的字体和 ICML 论文正文(Times New Roman)更统一，建议加上：
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']

import numpy as np
import torch
import pandas as pd
from . import config
from scipy.stats import t

def plot_policy_heatmap(policy_ipw, policy_ai, policy_dro, setting_name="Experiment"):
    """
    Plot Policy heatmap.
    Fix Nuisance Features (X3, X4, X5) to mean (0.0).
    Vary X1 and X2.
    """
    print(f"\n[Visualization] Generating Heatmap for {setting_name}...")
    
    # 1. Prepare grid
    resolution = 100
    x_range = np.linspace(-3.0, 3.0, resolution)
    X1, X2 = np.meshgrid(x_range, x_range)
    
    # 2. Construct input Tensor
    # Dimension: [resolution*resolution, 5]
    grid_flat = np.zeros((X1.size, config.DIM), dtype=np.float32)
    grid_flat[:, 0] = X1.ravel()
    grid_flat[:, 1] = X2.ravel()
    # Nuisance features (X3, X4, X5) default to 0.0 (config.TRAIN_X_MEAN[2:])
    
    tensor = torch.FloatTensor(grid_flat)
    
    # 3. Predict
    with torch.no_grad():
        P_ipw = policy_ipw(tensor).numpy().reshape(X1.shape)
        P_ai  = policy_ai(tensor).numpy().reshape(X1.shape)
        P_dro = policy_dro(tensor).numpy().reshape(X1.shape)
        
    # 4. Plot
    # Unified color scale range
    vmin = min(P_ipw.min(), P_ai.min(), P_dro.min())
    vmax = max(P_ipw.max(), P_ai.max(), P_dro.max())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    cmap = 'viridis'
    
    # Plot 1: IPW
    im1 = axes[0].contourf(X1, X2, P_ipw, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("IPW", fontsize=14)
    axes[0].set_xlabel(r"$X_1$")
    axes[0].set_ylabel(r"$X_2$")
    
    # Plot 2: Ai 2024
    im2 = axes[1].contourf(X1, X2, P_ai, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("DD", fontsize=14)
    axes[1].set_xlabel(r"$X_1$")
    axes[1].set_yticks([])
    
    # Plot 3: DRO
    im3 = axes[2].contourf(X1, X2, P_dro, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title("DRO", fontsize=14)
    axes[2].set_xlabel(r"$X_1$")
    axes[2].set_yticks([])
    
    # Colorbar
    cbar = fig.colorbar(im3, ax=axes, shrink=0.8, location='right')
    cbar.set_label('Recommended Price', fontsize=12)
    
    filename = f"heatmap_{setting_name}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved heatmap to {filename}")
    # plt.show()

def plot_dropl_vs_n(df, metric_col='Hat Q_DRO', title_suffix='', save_path=None):
    """
    Plot similar to Kallus22 Figure 2: Distributionally Robust Value vs N
    
    Parameters:
    -----------
    df : pd.DataFrame
        Contains columns ['N', 'Model', metric_col, 'Repeat'] or ['N', 'Model', 'Delta Test', metric_col, 'Repeat']
    metric_col : str
        Metric column name to plot ('Hat Q_DRO' or 'Hat Q_min')
    title_suffix : str
        Title suffix
    save_path : str, optional
        Save path
    """
    print(f"\n[Visualization] Generating DROPL vs N plot (Kallus22 Figure 2 style)...")
    
    # Check if 'Delta Test' column exists
    has_delta_test = 'Delta Test' in df.columns
    
    if has_delta_test:
        # Group by (N, Model, Delta Test)
        grouped = df.groupby(['N', 'Model', 'Delta Test'])[metric_col]
        means = grouped.mean().reset_index()
        stds = grouped.std().reset_index()
        counts = grouped.count().reset_index()
        
        merged = pd.merge(means, stds, on=['N', 'Model', 'Delta Test'], suffixes=('_mean', '_std'))
        merged = pd.merge(merged, counts, on=['N', 'Model', 'Delta Test'])
        merged.rename(columns={metric_col: 'count'}, inplace=True)
        
        # Compute 90% confidence interval (t-distribution)
        def compute_ci(row):
            n = row['count']
            if n <= 1:
                return row[metric_col + '_mean'], row[metric_col + '_mean']
            se = row[metric_col + '_std'] / np.sqrt(n)
            t_val = t.ppf(0.95, df=n-1)
            lower = row[metric_col + '_mean'] - t_val * se
            upper = row[metric_col + '_mean'] + t_val * se
            return lower, upper
        
        merged[['ci_lower', 'ci_upper']] = merged.apply(
            lambda row: pd.Series(compute_ci(row)), axis=1
        )
        
        # Get all models and delta_test values
        all_models = df['Model'].unique()
        all_deltas = sorted(df['Delta Test'].unique())
        
        # Define model order
        preferred_order = ['Ai2024']
        for model in all_models:
            if 'Leung2025' in model:
                preferred_order.append(model)
        for model in all_models:
            if 'DRO(' in model:
                if model not in preferred_order:
                    preferred_order.append(model)
        for model in all_models:
            if model not in preferred_order:
                preferred_order.append(model)
        
        # Define colors and linestyles (for each model, not varying with delta_test)
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        model_colors = {}
        model_linestyles = {}
        for i, model in enumerate(preferred_order):
            model_colors[model] = color_palette[i % len(color_palette)]
            if 'Ai2024' in model:
                model_linestyles[model] = '--'
            elif 'Leung2025' in model:
                model_linestyles[model] = '-.'
            elif 'DRO(' in model:
                model_linestyles[model] = '-'
            else:
                model_linestyles[model] = '-'
        
        # Compute global y-axis range (for unifying y-axis across all subplots)
        all_y_values = []
        for model in preferred_order:
            for delta_test in all_deltas:
                model_delta_data = merged[
                    (merged['Model'] == model) & 
                    (merged['Delta Test'] == delta_test)
                ]
                if len(model_delta_data) > 0:
                    all_y_values.extend(model_delta_data[metric_col + '_mean'].values)
                    all_y_values.extend(model_delta_data['ci_lower'].values)
                    all_y_values.extend(model_delta_data['ci_upper'].values)
        
        y_min = min(all_y_values) if all_y_values else 0
        y_max = max(all_y_values) if all_y_values else 1
        y_margin = (y_max - y_min) * 0.1  # 10% margin
        y_lim = (y_min - y_margin, y_max + y_margin)
        
        # Create subplot layout: if multiple delta_test, create subplots; otherwise use single plot
        if len(all_deltas) > 1:
            # Multiple delta_test: create subplots (1 row, number of delta_test columns)
            fig, axes = plt.subplots(1, len(all_deltas), figsize=(5*len(all_deltas), 6), sharey=True)
            # Ensure axes is in list format
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            else:
                axes = axes.flatten() if axes.ndim > 0 else [axes]
        else:
            # Single delta_test: use single plot
            fig, ax = plt.subplots(figsize=(10, 6))
            axes = [ax]
            all_deltas = [all_deltas[0]]  # Ensure it's a list
        
        # Create subplot for each delta_test
        for idx, delta_test in enumerate(all_deltas):
            ax = axes[idx]
            
            # Plot curve for each model (under current delta_test)
            for model in preferred_order:
                model_delta_data = merged[
                    (merged['Model'] == model) & 
                    (merged['Delta Test'] == delta_test)
                ].sort_values('N')
                
                if len(model_delta_data) == 0:
                    continue
                
                x = model_delta_data['N']
                y = model_delta_data[metric_col + '_mean']
                y_lower = model_delta_data['ci_lower']
                y_upper = model_delta_data['ci_upper']
                
                # Label: map model name to display name
                if 'Ai2024' in model:
                    label = 'DR'
                elif 'Leung2025' in model:
                    label = 'DRO-IPW'
                elif 'DRO(' in model:
                    label = 'CDR$^2$O$^2$PL-CP'
                else:
                    label = model
                
                # Plot curve (using model's color and linestyle)
                ax.plot(x, y, label=label, color=model_colors[model], 
                        linestyle=model_linestyles[model], linewidth=2, marker='o', markersize=4)
                
                # Plot 90% confidence interval (shaded region)
                ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=model_colors[model])
            
            # Set subplot title and labels
            ax.set_xlabel('Training Sample Size $T$', fontsize=24)
            if idx == 0:  # Only show y-axis label on first subplot
                ax.set_ylabel('Distributionally Robust Value $R_\\delta$', fontsize=24)
            ax.set_title(f'$\\delta = {delta_test}$', fontsize=24, fontweight='bold')
            ax.legend(loc='best', fontsize=22)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(y_lim)  # Unify y-axis range
    else:
        # Original logic (group by (N, Model))
        grouped = df.groupby(['N', 'Model'])[metric_col]
        means = grouped.mean().reset_index()
        stds = grouped.std().reset_index()
        counts = grouped.count().reset_index()
        
        merged = pd.merge(means, stds, on=['N', 'Model'], suffixes=('_mean', '_std'))
        merged = pd.merge(merged, counts, on=['N', 'Model'])
        merged.rename(columns={metric_col: 'count'}, inplace=True)
        
        # Compute 90% confidence interval (t-distribution)
        def compute_ci(row):
            n = row['count']
            if n <= 1:
                return row[metric_col + '_mean'], row[metric_col + '_mean']
            se = row[metric_col + '_std'] / np.sqrt(n)
            t_val = t.ppf(0.95, df=n-1)
            lower = row[metric_col + '_mean'] - t_val * se
            upper = row[metric_col + '_mean'] + t_val * se
            return lower, upper
        
        merged[['ci_lower', 'ci_upper']] = merged.apply(
            lambda row: pd.Series(compute_ci(row)), axis=1
        )
        
        # Get all model names and define order
        all_models = df['Model'].unique()
        preferred_order = ['Ai2024']
        for model in all_models:
            if 'Leung2025' in model:
                preferred_order.append(model)
        for model in all_models:
            if 'DRO(' in model:
                if model not in preferred_order:
                    preferred_order.append(model)
        for model in all_models:
            if model not in preferred_order:
                preferred_order.append(model)
        
        # Define colors and linestyles
        colors = {}
        linestyles = {}
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, model in enumerate(preferred_order):
            colors[model] = color_palette[i % len(color_palette)]
            if 'Ai2024' in model:
                linestyles[model] = '--'
            elif 'Leung2025' in model:
                linestyles[model] = '-.'
            elif 'DRO(' in model:
                linestyles[model] = '-'
            else:
                linestyles[model] = ['--', '-.', ':', '-'][i % 4]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]  # Unified as list format for easier processing
        
        for model in preferred_order:
            model_data = merged[merged['Model'] == model].sort_values('N')
            if len(model_data) == 0:
                continue
            
            x = model_data['N']
            y = model_data[metric_col + '_mean']
            y_lower = model_data['ci_lower']
            y_upper = model_data['ci_upper']
            
            # Label: map model name to display name
            if 'Ai2024' in model:
                label = 'DR'
            elif 'Leung2025' in model:
                label = 'DRO-IPW'
            elif 'DRO(' in model:
                label = 'CDR$^2$O$^2$PL-CP'
            else:
                label = model
            
            # Plot curve
            ax.plot(x, y, label=label, color=colors.get(model, 'gray'), 
                    linestyle=linestyles.get(model, '-'), linewidth=2, marker='o', markersize=4)
            
            # Plot 90% confidence interval (shaded region)
            ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=colors.get(model, 'gray'))
        
        # Set labels and title (single subplot case)
        ax.set_xlabel('Training Sample Size $T$', fontsize=24)
        ax.set_ylabel('Distributionally Robust Value $R_\\delta$', fontsize=24)
        ax.set_title(f'Distributionally Robust Policy Learning vs N{title_suffix}', fontsize=24, fontweight='bold')
        ax.legend(loc='best', fontsize=22)
        ax.grid(True, alpha=0.3)
    
    # Overall title removed (as requested)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        filename = f"dropl_vs_n_{metric_col.lower()}{title_suffix}.pdf"
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved figure to {filename}")
    
    plt.close()

def plot_dropl_vs_n_from_stats(df_stats, title_suffix='', save_path=None):
    """
    Plot Distributionally Robust Value vs N from statistics file
    
    This function is designed to work with pre-computed statistics files that contain
    Mean, CI_Lower, and CI_Upper columns, eliminating the need for raw data with Repeat column.
    
    Parameters:
    -----------
    df_stats : pd.DataFrame
        Contains columns ['Delta Test', 'T', 'Model', 'Mean', 'CI_Lower', 'CI_Upper']
        Note: 'T' column represents training size N
    title_suffix : str
        Title suffix
    save_path : str, optional
        Save path
    """
    print(f"\n[Visualization] Generating DROPL vs N plot from statistics file...")
    
    # Check if 'Delta Test' column exists
    has_delta_test = 'Delta Test' in df_stats.columns
    
    if has_delta_test:
        # Get all models and delta_test values
        all_models = df_stats['Model'].unique()
        all_deltas = sorted(df_stats['Delta Test'].unique())
        
        # Define model order
        preferred_order = ['Ai2024']
        for model in all_models:
            if 'Leung2025' in model:
                preferred_order.append(model)
        for model in all_models:
            if 'DRO(' in model:
                if model not in preferred_order:
                    preferred_order.append(model)
        for model in all_models:
            if model not in preferred_order:
                preferred_order.append(model)
        
        # Define colors and linestyles (for each model, not varying with delta_test)
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        model_colors = {}
        model_linestyles = {}
        for i, model in enumerate(preferred_order):
            model_colors[model] = color_palette[i % len(color_palette)]
            if 'Ai2024' in model:
                model_linestyles[model] = '--'
            elif 'Leung2025' in model:
                model_linestyles[model] = '-.'
            elif 'DRO(' in model:
                model_linestyles[model] = '-'
            else:
                model_linestyles[model] = '-'
        
        # Compute global y-axis range (for unifying y-axis across all subplots)
        all_y_values = []
        for model in preferred_order:
            for delta_test in all_deltas:
                model_delta_data = df_stats[
                    (df_stats['Model'] == model) & 
                    (df_stats['Delta Test'] == delta_test)
                ]
                if len(model_delta_data) > 0:
                    all_y_values.extend(model_delta_data['Mean'].values)
                    all_y_values.extend(model_delta_data['CI_Lower'].values)
                    all_y_values.extend(model_delta_data['CI_Upper'].values)
        
        y_min = min(all_y_values) if all_y_values else 0
        y_max = max(all_y_values) if all_y_values else 1
        y_margin = (y_max - y_min) * 0.1  # 10% margin
        y_lim = (y_min - y_margin, y_max + y_margin)
        
        # Create subplot layout: if multiple delta_test, create subplots; otherwise use single plot
        if len(all_deltas) > 1:
            # Multiple delta_test: create subplots (1 row, number of delta_test columns)
            fig, axes = plt.subplots(1, len(all_deltas), figsize=(5*len(all_deltas), 6), sharey=True)
            # Ensure axes is in list format
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            else:
                axes = axes.flatten() if axes.ndim > 0 else [axes]
        else:
            # Single delta_test: use single plot
            fig, ax = plt.subplots(figsize=(10, 6))
            axes = [ax]
            all_deltas = [all_deltas[0]]  # Ensure it's a list
        
        # Create subplot for each delta_test
        for idx, delta_test in enumerate(all_deltas):
            ax = axes[idx]
            
            # Plot curve for each model (under current delta_test)
            for model in preferred_order:
                model_delta_data = df_stats[
                    (df_stats['Model'] == model) & 
                    (df_stats['Delta Test'] == delta_test)
                ].sort_values('T')
                
                if len(model_delta_data) == 0:
                    continue
                
                x = model_delta_data['T']
                y = model_delta_data['Mean']
                y_lower = model_delta_data['CI_Lower']
                y_upper = model_delta_data['CI_Upper']
                
                # Label: map model name to display name
                if 'Ai2024' in model:
                    label = 'DR'
                elif 'Leung2025' in model:
                    label = 'DRO-IPW'
                elif 'DRO(' in model:
                    label = 'CDR$^2$O$^2$PL-CP'
                else:
                    label = model
                
                # Plot curve (using model's color and linestyle)
                ax.plot(x, y, label=label, color=model_colors[model], 
                        linestyle=model_linestyles[model], linewidth=2, marker='o', markersize=4)
                
                # Plot 90% confidence interval (shaded region)
                ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=model_colors[model])
            
            # Set subplot title and labels
            ax.set_xlabel('Training Sample Size $T$', fontsize=24)
            if idx == 0:  # Only show y-axis label on first subplot
                ax.set_ylabel('Distributionally Robust Value $R_\\delta$', fontsize=24)
            ax.set_title(f'$\\delta = {delta_test}$', fontsize=24, fontweight='bold')
            ax.legend(loc='best', fontsize=22)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(y_lim)  # Unify y-axis range
    else:
        # No Delta Test column: single plot
        all_models = df_stats['Model'].unique()
        preferred_order = ['Ai2024']
        for model in all_models:
            if 'Leung2025' in model:
                preferred_order.append(model)
        for model in all_models:
            if 'DRO(' in model:
                if model not in preferred_order:
                    preferred_order.append(model)
        for model in all_models:
            if model not in preferred_order:
                preferred_order.append(model)
        
        # Define colors and linestyles
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        model_colors = {}
        model_linestyles = {}
        for i, model in enumerate(preferred_order):
            model_colors[model] = color_palette[i % len(color_palette)]
            if 'Ai2024' in model:
                model_linestyles[model] = '--'
            elif 'Leung2025' in model:
                model_linestyles[model] = '-.'
            elif 'DRO(' in model:
                model_linestyles[model] = '-'
            else:
                model_linestyles[model] = '-'
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model in preferred_order:
            model_data = df_stats[df_stats['Model'] == model].sort_values('T')
            if len(model_data) == 0:
                continue
            
            x = model_data['T']
            y = model_data['Mean']
            y_lower = model_data['CI_Lower']
            y_upper = model_data['CI_Upper']
            
            # Label: map model name to display name
            if 'Ai2024' in model:
                label = 'DR'
            elif 'Leung2025' in model:
                label = 'DRO-IPW'
            elif 'DRO(' in model:
                label = 'CDR$^2$O$^2$PL-CP'
            else:
                label = model
            
            # Plot curve
            ax.plot(x, y, label=label, color=model_colors.get(model, 'gray'), 
                    linestyle=model_linestyles.get(model, '-'), linewidth=2, marker='o', markersize=4)
            
            # Plot 90% confidence interval (shaded region)
            ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=model_colors.get(model, 'gray'))
        
        # Set labels and title
        ax.set_xlabel('Traing Sample Size $T$', fontsize=24)
        ax.set_ylabel('Distributionally Robust Value $R_\\delta$', fontsize=24)
        ax.set_title(f'Distributionally Robust Policy Learning vs N{title_suffix}', fontsize=24, fontweight='bold')
        ax.legend(loc='best', fontsize=22)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        filename = f"dropl_vs_n_from_stats{title_suffix}.pdf"
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved figure to {filename}")
    
    plt.close()

def plot_dropl_vs_delta(df, metric_col='Hat Q_DRO', title_suffix='', save_path=None):
    """
    Plot Distributionally Robust Value vs Test Delta
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing columns ['Delta Test', 'Model', metric_col, 'Repeat']
    metric_col : str
        Name of the metric column to plot ('Hat Q_DRO' or 'Hat Q_min')
    title_suffix : str
        Title suffix
    save_path : str, optional
        Save path
    """
    print(f"\n[Visualization] Generating DROPL vs Delta plot...")
    
    # Compute mean and 90% confidence interval for each (Delta Test, Model) combination
    grouped = df.groupby(['Delta Test', 'Model'])[metric_col]
    means = grouped.mean().reset_index()
    stds = grouped.std().reset_index()
    counts = grouped.count().reset_index()
    
    merged = pd.merge(means, stds, on=['Delta Test', 'Model'], suffixes=('_mean', '_std'))
    merged = pd.merge(merged, counts, on=['Delta Test', 'Model'])
    merged.rename(columns={metric_col: 'count'}, inplace=True)
    
    # Compute 90% confidence interval
    def compute_ci(row):
        n = row['count']
        if n <= 1:
            return row[metric_col + '_mean'], row[metric_col + '_mean']
        se = row[metric_col + '_std'] / np.sqrt(n)
        t_val = t.ppf(0.95, df=n-1)
        lower = row[metric_col + '_mean'] - t_val * se
        upper = row[metric_col + '_mean'] + t_val * se
        return lower, upper
    
    merged[['ci_lower', 'ci_upper']] = merged.apply(
        lambda row: pd.Series(compute_ci(row)), axis=1
    )
    
    # Get all model names and define order
    all_models = df['Model'].unique()
    preferred_order = ['Ai2024']
    for model in all_models:
        if 'Leung2025' in model:
            preferred_order.append(model)
    for model in all_models:
        if 'DRO(' in model:
            if model not in preferred_order:
                preferred_order.append(model)
    for model in all_models:
        if model not in preferred_order:
            preferred_order.append(model)
    
    # Define colors and line styles
    colors = {}
    linestyles = {}
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, model in enumerate(preferred_order):
        colors[model] = color_palette[i % len(color_palette)]
        if 'Ai2024' in model:
            linestyles[model] = '--'
        elif 'Leung2025' in model:
            linestyles[model] = '-.'
        elif 'DRO(' in model:
            linestyles[model] = '-'
        else:
            linestyles[model] = ['--', '-.', ':', '-'][i % 4]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in preferred_order:
        model_data = merged[merged['Model'] == model].sort_values('Delta Test')
        if len(model_data) == 0:
            continue
        
        x = model_data['Delta Test']
        y = model_data[metric_col + '_mean']
        y_lower = model_data['ci_lower']
        y_upper = model_data['ci_upper']
        
        # Plot curve
        ax.plot(x, y, label=model, color=colors.get(model, 'gray'), 
                linestyle=linestyles.get(model, '-'), linewidth=2, marker='o', markersize=4)
        
        # Plot 90% confidence interval
        ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=colors.get(model, 'gray'))
    
    ax.set_xlabel('Test Delta $\\delta$', fontsize=24)
    ax.set_ylabel('Distributionally Robust Value $R_\\delta$', fontsize=24)
    ax.set_title(f'Distributionally Robust Policy Learning vs Test Delta{title_suffix}', fontsize=24)
    ax.legend(loc='best', fontsize=22)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        filename = f"dropl_vs_delta_{metric_col.lower()}{title_suffix}.pdf"
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved figure to {filename}")
    
    plt.close()

def plot_behavior_vs_target_policy(behavior_policy_func, target_policies_dict, setting_name="BehaviorVsTarget"):
    """
    Plot similar to Kallus22 Figure 1: Behavior Policy vs Target Policy
    
    For continuous action space, we plot heatmap of recommended prices.
    
    Parameters:
    -----------
    behavior_policy_func : callable
        Behavior policy function, takes X (numpy array) and returns recommended prices (numpy array)
    target_policies_dict : dict
        Dictionary, key is policy name, value is policy function (torch.nn.Module or callable)
    setting_name : str
        Setting name, used for saving file
    """
    print(f"\n[Visualization] Generating Behavior vs Target Policy plot (Kallus22 Figure 1 style)...")
    
    # 1. Prepare grid
    resolution = 100
    x_range = np.linspace(-3.0, 3.0, resolution)
    X1, X2 = np.meshgrid(x_range, x_range)
    
    # 2. Construct input
    grid_flat = np.zeros((X1.size, config.DIM), dtype=np.float32)
    grid_flat[:, 0] = X1.ravel()
    grid_flat[:, 1] = X2.ravel()
    # Nuisance features (X3, X4, X5) default to 0.0
    
    # 3. Compute Behavior Policy
    P_behavior = behavior_policy_func(grid_flat).reshape(X1.shape)
    
    # 4. Compute Target Policies
    target_policies_results = {}
    tensor = torch.FloatTensor(grid_flat)
    for name, policy in target_policies_dict.items():
        with torch.no_grad():
            if isinstance(policy, torch.nn.Module):
                P_target = policy(tensor).numpy().reshape(X1.shape)
            else:
                P_target = policy(grid_flat).reshape(X1.shape)
        target_policies_results[name] = P_target
    
    # 5. Determine number of subplots: 1 (behavior) + len(target_policies)
    n_targets = len(target_policies_dict)
    n_cols = min(3, n_targets + 1)  # Maximum 3 columns
    n_rows = (n_targets + 1 + n_cols - 1) // n_cols  # Round up
    
    # 6. Plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), constrained_layout=True)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, n_cols)
    
    # Unified color scale range
    all_values = [P_behavior]
    for P_target in target_policies_results.values():
        all_values.append(P_target)
    vmin = min(p.min() for p in all_values)
    vmax = max(p.max() for p in all_values)
    cmap = 'viridis'
    
    # Plot Behavior Policy (first subplot)
    ax = axes[0, 0] if n_rows > 1 or n_cols > 1 else axes[0]
    im = ax.contourf(X1, X2, P_behavior, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Behavior Policy\n(Logging Policy)", fontsize=14)
    ax.set_xlabel(r"$X_1$", fontsize=12)
    ax.set_ylabel(r"$X_2$", fontsize=12)
    
    # Plot Target Policies
    for idx, (name, P_target) in enumerate(target_policies_results.items()):
        row = (idx + 1) // n_cols
        col = (idx + 1) % n_cols
        ax = axes[row, col] if n_rows > 1 or n_cols > 1 else axes[idx + 1]
        
        im = ax.contourf(X1, X2, P_target, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Target Policy\n({name})", fontsize=14)
        ax.set_xlabel(r"$X_1$", fontsize=12)
        if col == 0:
            ax.set_ylabel(r"$X_2$", fontsize=12)
        else:
            ax.set_yticks([])
    
    # Hide extra subplots
    total_plots = 1 + n_targets
    for idx in range(total_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, location='right')
    cbar.set_label('Recommended Price', fontsize=12)
    
    filename = f"BehaviorVsTarget_{setting_name}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved figure to {filename}")


def plot_ope_figure1(df, save_path=None):
    """
    Plot similar to Kallus22 Figure 1: MSE vs T (training sample size)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Contains columns ['Delta Test', 'T', 'Method', 'MSE', 'Repeat']
    save_path : str, optional
        Save path (default: 'OPE_Figure1.pdf')
    """
    print(f"\n[Visualization] Generating OPE Figure 1 (MSE vs T)...")
    
    # Group by (Delta Test, T, Method)
    grouped = df.groupby(['Delta Test', 'T', 'Method'])['MSE']
    means = grouped.mean().reset_index()
    stds = grouped.std().reset_index()
    counts = grouped.count().reset_index()
    
    merged = pd.merge(means, stds, on=['Delta Test', 'T', 'Method'], suffixes=('_mean', '_std'))
    merged = pd.merge(merged, counts, on=['Delta Test', 'T', 'Method'])
    merged.rename(columns={'MSE': 'count'}, inplace=True)
    
    # Compute 90% confidence interval (t-distribution)
    def compute_ci(row):
        n = row['count']
        if n <= 1:
            return row['MSE_mean'], row['MSE_mean']
        se = row['MSE_std'] / np.sqrt(n)
        t_val = t.ppf(0.95, df=n-1)
        lower = row['MSE_mean'] - t_val * se
        upper = row['MSE_mean'] + t_val * se
        return lower, upper
    
    merged[['ci_lower', 'ci_upper']] = merged.apply(
        lambda row: pd.Series(compute_ci(row)), axis=1
    )
    
    # Get all delta_test values and methods
    all_deltas = sorted(df['Delta Test'].unique())
    all_methods = df['Method'].unique()
    
    # Define method order and styles
    method_order = []
    if 'LDR2PE-CP' in all_methods:
        method_order.append('LDR2PE-CP')
    if 'DRO-IPW' in all_methods:
        method_order.append('DRO-IPW')
    for method in all_methods:
        if method not in method_order:
            method_order.append(method)
    
    # Define colors and linestyles (directly specified, referencing Figure 2 color scheme)
    method_colors = {}
    method_linestyles = {}
    method_labels = {}
    for method in method_order:
        if 'LDR2PE-CP' in method:
            method_colors[method] = '#2ca02c'  # Green
            method_linestyles[method] = '-'  # Solid line
            method_labels[method] = 'LDR$^2$O$^2$PE-CP'
        elif 'DRO-IPW' in method:
            method_colors[method] = '#ff7f0e'  # Orange
            method_linestyles[method] = '-.'  # Dash-dot line
            method_labels[method] = 'DRO-IPW'
        else:
            # Default settings for other methods
            method_colors[method] = '#1f77b4'  # Blue
            method_linestyles[method] = '--'  # Dashed line
            method_labels[method] = method
    
    # Compute global y-axis range (for unifying y-axis across all subplots)
    all_y_values = []
    for method in method_order:
        for delta_test in all_deltas:
            method_delta_data = merged[
                (merged['Method'] == method) & 
                (merged['Delta Test'] == delta_test)
            ]
            if len(method_delta_data) > 0:
                all_y_values.extend(method_delta_data['MSE_mean'].values)
                all_y_values.extend(method_delta_data['ci_lower'].values)
                all_y_values.extend(method_delta_data['ci_upper'].values)
    
    y_min = min(all_y_values) if all_y_values else 0
    y_max = max(all_y_values) if all_y_values else 1
    y_margin = (y_max - y_min) * 0.1  # 10% margin
    y_lim = (max(0, y_min - y_margin), y_max + y_margin)
    
    # Create subplot layout: if multiple delta_test, create subplots; otherwise use single plot
    if len(all_deltas) > 1:
        # Multiple delta_test: create subplots (1 row, number of delta_test columns)
        fig, axes = plt.subplots(1, len(all_deltas), figsize=(5*len(all_deltas), 6), sharey=True)
        # Ensure axes is in list format
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        else:
            axes = axes.flatten() if axes.ndim > 0 else [axes]
    else:
        # Single delta_test: use single plot
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
        all_deltas = [all_deltas[0]]  # Ensure it's a list
    
    # Create subplot for each delta_test
    for idx, delta_test in enumerate(all_deltas):
        ax = axes[idx]
        
        # Plot curve for each method (under current delta_test)
        for method in method_order:
            method_delta_data = merged[
                (merged['Method'] == method) & 
                (merged['Delta Test'] == delta_test)
            ].sort_values('T')
            
            if len(method_delta_data) == 0:
                continue
            
            x = method_delta_data['T']
            y = method_delta_data['MSE_mean']
            y_lower = method_delta_data['ci_lower']
            y_upper = method_delta_data['ci_upper']
            
            # Get label
            label = method_labels.get(method, method)
            
            # Plot curve (using method's color and linestyle, optimized for softer style)
            ax.plot(x, y, label=label, color=method_colors[method], 
                    linestyle=method_linestyles[method], linewidth=2, marker='o', 
                    markersize=4, alpha=0.8, markeredgewidth=0.5, markeredgecolor=method_colors[method])
            
            # Plot 90% confidence interval (shaded region)
            ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=method_colors[method])
        
        # Set subplot title and labels
        ax.set_xlabel('Training Sample Size $T$', fontsize=24)
        if idx == 0:  # Only show y-axis label on first subplot
            ax.set_ylabel('MSE of Estimated\nDistributionally Robust Value', fontsize=24)
        ax.set_title(f'$\\delta = {delta_test}$', fontsize=24, fontweight='bold')
        ax.legend(loc='best', fontsize=22)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_lim)  # Unify y-axis range
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = 'OPE_Figure1.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved Figure 1 to {save_path}")
    plt.close()
    plt.close()

def plot_ope_figure1_from_stats(df_stats, save_path=None):
    """
    Plot OPE Figure 1 from statistics file
    
    This function is designed to work with pre-computed statistics files that contain
    Mean_MSE, CI_Lower_MSE, and CI_Upper_MSE columns, eliminating the need for raw data with Repeat column.
    
    Parameters:
    -----------
    df_stats : pd.DataFrame
        Contains columns ['Delta Test', 'T', 'Method', 'Mean_MSE', 'CI_Lower_MSE', 'CI_Upper_MSE']
    save_path : str, optional
        Save path (default: 'OPE_Figure1.pdf')
    """
    print(f"\n[Visualization] Generating OPE Figure 1 from statistics file...")
    
    # Get all delta_test values and methods
    all_deltas = sorted(df_stats['Delta Test'].unique())
    all_methods = df_stats['Method'].unique()
    
    # Define method order and styles
    method_order = []
    if 'LDR2PE-CP' in all_methods:
        method_order.append('LDR2PE-CP')
    if 'DRO-IPW' in all_methods:
        method_order.append('DRO-IPW')
    for method in all_methods:
        if method not in method_order:
            method_order.append(method)
    
    # Define colors and linestyles (directly specified, referencing Figure 2 color scheme)
    method_colors = {}
    method_linestyles = {}
    method_labels = {}
    for method in method_order:
        if 'LDR2PE-CP' in method:
            method_colors[method] = '#2ca02c'  # Green
            method_linestyles[method] = '-'  # Solid line
            method_labels[method] = 'LDR$^2$O$^2$PE-CP'
        elif 'DRO-IPW' in method:
            method_colors[method] = '#ff7f0e'  # Orange
            method_linestyles[method] = '-.'  # Dash-dot line
            method_labels[method] = 'DRO-IPW'
        else:
            # Default settings for other methods
            method_colors[method] = '#1f77b4'  # Blue
            method_linestyles[method] = '--'  # Dashed line
            method_labels[method] = method
    
    # Compute global y-axis range (for unifying y-axis across all subplots)
    all_y_values = []
    for method in method_order:
        for delta_test in all_deltas:
            method_delta_data = df_stats[
                (df_stats['Method'] == method) & 
                (df_stats['Delta Test'] == delta_test)
            ]
            if len(method_delta_data) > 0:
                all_y_values.extend(method_delta_data['Mean_MSE'].values)
                all_y_values.extend(method_delta_data['CI_Lower_MSE'].values)
                all_y_values.extend(method_delta_data['CI_Upper_MSE'].values)
    
    y_min = min(all_y_values) if all_y_values else 0
    y_max = max(all_y_values) if all_y_values else 1
    y_margin = (y_max - y_min) * 0.1  # 10% margin
    y_lim = (max(0, y_min - y_margin), y_max + y_margin)
    
    # Create subplot layout: if multiple delta_test, create subplots; otherwise use single plot
    if len(all_deltas) > 1:
        # Multiple delta_test: create subplots (1 row, number of delta_test columns)
        fig, axes = plt.subplots(1, len(all_deltas), figsize=(5*len(all_deltas), 6), sharey=True)
        # Ensure axes is in list format
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        else:
            axes = axes.flatten() if axes.ndim > 0 else [axes]
    else:
        # Single delta_test: use single plot
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
        all_deltas = [all_deltas[0]]  # Ensure it's a list
    
    # Create subplot for each delta_test
    for idx, delta_test in enumerate(all_deltas):
        ax = axes[idx]
        
        # Plot curve for each method (under current delta_test)
        for method in method_order:
            method_delta_data = df_stats[
                (df_stats['Method'] == method) & 
                (df_stats['Delta Test'] == delta_test)
            ].sort_values('T')
            
            if len(method_delta_data) == 0:
                continue
            
            x = method_delta_data['T']
            y = method_delta_data['Mean_MSE']
            y_lower = method_delta_data['CI_Lower_MSE']
            y_upper = method_delta_data['CI_Upper_MSE']
            
            # Get label
            label = method_labels.get(method, method)
            
            # Plot curve (using method's color and linestyle, optimized for softer style)
            ax.plot(x, y, label=label, color=method_colors[method], 
                    linestyle=method_linestyles[method], linewidth=2, marker='o', 
                    markersize=4, alpha=0.8, markeredgewidth=0.5, markeredgecolor=method_colors[method])
            
            # Plot 90% confidence interval (shaded region)
            ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=method_colors[method])
        
        # Set subplot title and labels
        ax.set_xlabel('Training Sample Size $T$', fontsize=24)
        if idx == 0:  # Only show y-axis label on first subplot
            ax.set_ylabel('MSE of Estimated\nDistributionally Robust Value', fontsize=24)
        ax.set_title(f'$\\delta = {delta_test}$', fontsize=24, fontweight='bold')
        ax.legend(loc='best', fontsize=22)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_lim)  # Unify y-axis range
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = 'OPE_Figure1.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved Figure 1 to {save_path}")
    plt.close()