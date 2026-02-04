import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from plot_registry import PlotRegistry
from utils.colors import get_palette

@PlotRegistry.register(
    id="stats_forest",
    category="Stats",
    display_name="Forest Plot",
    required_columns=["label", "mean", "ci_lower", "ci_upper"],
    optional_columns=["group", "pvalue"],
    description="Forest plot for effect sizes.",
    supports_grouping=False
)
def plot_forest(df: pd.DataFrame, label: str, mean: str, ci_lower: str, ci_upper: str,
                group: Optional[str] = None, pvalue: Optional[str] = None,
                palette: str = "npg", title: str = "Forest Plot", **kwargs) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.5 + 2))
    
    # Reverse order for plotting top-down
    df = df.iloc[::-1].reset_index(drop=True)
    
    y_pos = np.arange(len(df))
    
    ax.errorbar(
        x=df[mean], y=y_pos, 
        xerr=[df[mean] - df[ci_lower], df[ci_upper] - df[mean]], 
        fmt='o', color='black', capsize=5
    )
    
    # Add vertical line at 0 or 1 depending on effect size type
    # Heuristic: if mean is around 1 (odds ratio), line at 1. If 0 (log odds), line at 0.
    # User didn't specify, we'll assume line at 0 unless values are all positive and around 1.
    ref_line = 0
    if df[mean].min() > 0 and df[mean].mean() > 0.5:
        ref_line = 1
    
    ax.axvline(ref_line, color='grey', linestyle='--', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df[label])
    
    ax.set_title(title)
    ax.set_xlabel("Effect Size (95% CI)")
    
    # Add p-values to the right if available
    if pvalue:
        # Create a twin axis or just text
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(ax.get_xlim()[1] * 1.05, i, f"p={row[pvalue]:.3g}", va='center')
            
    sns.despine()
    return fig

@PlotRegistry.register(
    id="stats_effect_significance",
    category="Stats",
    display_name="Effect vs Significance",
    required_columns=["effect_size", "pvalue"],
    optional_columns=["label", "group"],
    description="Scatter plot of effect size vs significance (like Volcano but generic).",
    supports_grouping=True
)
def plot_effect_significance(df: pd.DataFrame, effect_size: str, pvalue: str,
                             label: Optional[str] = None, group: Optional[str] = None,
                             palette: str = "npg", title: str = "Effect vs Significance", **kwargs) -> plt.Figure:
    
    # Similar to Volcano
    fig, ax = plt.subplots(figsize=(8, 8))
    
    df['neg_log10_pval'] = -np.log10(df[pvalue])
    colors = get_palette(palette)
    
    sns.scatterplot(
        data=df, x=effect_size, y='neg_log10_pval', hue=group,
        palette=colors if group else None,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel("Effect Size")
    ax.set_ylabel("-log10 P-value")
    sns.despine()
    return fig
