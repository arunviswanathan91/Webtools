import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional
from plot_registry import PlotRegistry
from utils.colors import get_palette

@PlotRegistry.register(
    id="immune_cell_fraction",
    category="Immune",
    display_name="Cell Fraction",
    required_columns=["sample_id", "cell_type", "fraction"],
    optional_columns=["group"],
    description="Bar plot of cell type fractions per sample.",
    supports_grouping=True
)
def plot_cell_fraction(df: pd.DataFrame, sample_id: str, cell_type: str, fraction: str,
                       group: Optional[str] = None, palette: str = "npg", 
                       title: str = "Cell Fraction", **kwargs) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_palette(palette)
    
    # Standard stacked bar plot
    # Pivot
    pivot_df = df.pivot_table(index=sample_id, columns=cell_type, values=fraction)
    pivot_df = pivot_df.fillna(0)
    
    # Normalize to 100% just in case
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)
    
    # Sort by group if provided
    if group:
        group_map = df.drop_duplicates(subset=[sample_id]).set_index(sample_id)[group]
        pivot_df['group'] = pivot_df.index.map(group_map)
        pivot_df = pivot_df.sort_values(by='group')
        pivot_df = pivot_df.drop('group', axis=1)
        
    pivot_df.plot(kind='bar', stacked=True, ax=ax, color=colors if len(colors) >= len(pivot_df.columns) else None)
    
    ax.set_title(title)
    ax.set_ylabel("Fraction")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    sns.despine()
    plt.tight_layout()
    return fig

@PlotRegistry.register(
    id="immune_stacked_composition",
    category="Immune",
    display_name="Stacked Composition",
    required_columns=["sample_id", "component", "value"],
    optional_columns=["group"],
    description="Stacked bar plot of composition.",
    supports_grouping=True
)
def plot_stacked_composition(df: pd.DataFrame, sample_id: str, component: str, value: str,
                             group: Optional[str] = None, palette: str = "npg", 
                             title: str = "Composition", **kwargs) -> plt.Figure:
    # Same implementation as Cell Fraction essentially
    return plot_cell_fraction(df, sample_id, component, value, group, palette, title, **kwargs)

@PlotRegistry.register(
    id="immune_dysfunction",
    category="Immune",
    display_name="Dysfunction Heatmap",
    required_columns=["sample_id", "gene", "expression"],
    optional_columns=["group"],
    description="Heatmap of dysfunction markers.",
    supports_grouping=True
)
def plot_dysfunction_heatmap(df: pd.DataFrame, sample_id: str, gene: str, expression: str,
                             group: Optional[str] = None, palette: str = "npg", 
                             title: str = "Dysfunction Markers", **kwargs) -> plt.Figure:
    
    # Reuse heatmap logic but maybe with specific defaults for dysfunction
    from plotters.heatmaps import plot_expression_heatmap
    return plot_expression_heatmap(df, gene, sample_id, expression, group=group, palette=palette, title=title)
