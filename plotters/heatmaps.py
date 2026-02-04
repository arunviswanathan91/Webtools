import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from plot_registry import PlotRegistry
from utils.colors import get_palette

@PlotRegistry.register(
    id="heatmap_expression",
    category="Heatmaps",
    display_name="Expression Heatmap",
    required_columns=["gene", "sample_id", "expression_value"],
    optional_columns=["group", "zscore_group"],
    description="Clustered heatmap of gene expression.",
    supports_grouping=True
)
def plot_expression_heatmap(df: pd.DataFrame, gene: str, sample_id: str, expression_value: str,
                            group: Optional[str] = None, zscore_group: Optional[str] = None,
                            palette: str = "npg", title: str = "Expression Heatmap", **kwargs) -> plt.Figure:
    
    # Pivot to wide format: Index=Gene, Columns=Sample
    # If duplicates exist (e.g. multiple probes per gene), we need to aggregate.
    # We'll assume mean aggregation for safety.
    wide_df = df.pivot_table(index=gene, columns=sample_id, values=expression_value, aggfunc='mean')
    
    # Fill NAs
    wide_df = wide_df.fillna(0)
    
    # Z-score normalization (standard scaling) per gene (row)
    # 0 = rows, 1 = columns. standard_scale=0 means standardizing rows. z_score=0 means z-score rows.
    # sns.clustermap 'z_score' parameter takes 0 or 1.
    
    # Create column colors if group is provided
    col_colors = None
    if group:
        # Create a mapping from sample_id to group
        sample_to_group = df.drop_duplicates(subset=[sample_id]).set_index(sample_id)[group]
        # Realign to wide_df columns
        sample_to_group = sample_to_group.reindex(wide_df.columns)
        
        # Map groups to colors
        unique_groups = sample_to_group.unique()
        colors = get_palette(palette)
        group_color_map = {g: colors[i % len(colors)] for i, g in enumerate(unique_groups)}
        col_colors = sample_to_group.map(group_color_map)
    
    # Clustermap creates its own Figure
    g = sns.clustermap(
        wide_df,
        z_score=0, # Z-score along rows (genes)
        cmap="RdBu_r", # Diverging colormap
        center=0,
        col_colors=col_colors,
        figsize=(10, 10),
        dendrogram_ratio=(.1, .2)
    )
    
    g.fig.suptitle(title, y=1.02)
    return g.fig

@PlotRegistry.register(
    id="heatmap_correlation",
    category="Heatmaps",
    display_name="Correlation Heatmap",
    required_columns=["var1", "var2", "correlation"],
    optional_columns=[],
    description="Heatmap showing correlation between variables.",
    supports_grouping=False
)
def plot_correlation_heatmap(df: pd.DataFrame, var1: str, var2: str, correlation: str,
                             palette: str = "npg", title: str = "Correlation Heatmap", **kwargs) -> plt.Figure:
    
    wide_df = df.pivot_table(index=var1, columns=var2, values=correlation)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sns.heatmap(
        wide_df,
        cmap="RdBu_r",
        center=0,
        annot=True, # Show values
        fmt=".2f",
        square=True,
        ax=ax
    )
    
    ax.set_title(title)
    return fig

@PlotRegistry.register(
    id="heatmap_signature",
    category="Heatmaps",
    display_name="Signature Heatmap",
    required_columns=["signature", "sample_id", "score"],
    optional_columns=["group"],
    description="Heatmap of pathway/signature scores per sample.",
    supports_grouping=True
)
def plot_signature_heatmap(df: pd.DataFrame, signature: str, sample_id: str, score: str,
                           group: Optional[str] = None, palette: str = "npg", 
                           title: str = "Signature Scores", **kwargs) -> plt.Figure:
    
    # Similar to expression heatmap but usually no z-score needed if scores are already comparable across signatures
    # Or maybe z-score across samples for each signature?
    
    wide_df = df.pivot_table(index=signature, columns=sample_id, values=score, aggfunc='mean')
    
    col_colors = None
    if group:
        sample_to_group = df.drop_duplicates(subset=[sample_id]).set_index(sample_id)[group]
        sample_to_group = sample_to_group.reindex(wide_df.columns)
        unique_groups = sample_to_group.unique()
        colors = get_palette(palette)
        group_color_map = {g: colors[i % len(colors)] for i, g in enumerate(unique_groups)}
        col_colors = sample_to_group.map(group_color_map)
        
    g = sns.clustermap(
        wide_df,
        cmap="viridis",
        col_colors=col_colors,
        figsize=(10, 8),
        dendrogram_ratio=(.1, .1)
        # No z-score by default for scores
    )
    
    g.fig.suptitle(title, y=1.02)
    return g.fig
