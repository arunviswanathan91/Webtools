import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List
from plot_registry import PlotRegistry
from utils.colors import get_palette

@PlotRegistry.register(
    id="pathway_enrichment_dot",
    category="Pathways",
    display_name="Enrichment Dotplot",
    required_columns=["pathway", "enrichment_score", "pvalue", "count"],
    optional_columns=["gene_ratio"],
    description="Dotplot for pathway enrichment analysis.",
    supports_grouping=False
)
def plot_enrichment_dot(df: pd.DataFrame, pathway: str, enrichment_score: str, 
                        pvalue: str, count: str, gene_ratio: Optional[str] = None,
                        palette: str = "npg", title: str = "Pathway Enrichment", **kwargs) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Sorting: usually by pvalue or score
    # Top 20 pathways
    df_sorted = df.sort_values(by=pvalue, ascending=True).head(20)
    # Reverse so top is at top of plot
    df_sorted = df_sorted.iloc[::-1]
    
    # Calculate -log10 pvalue if needed or use pvalue for color
    # Usually dot plot: Size = Count, Color = Pvalue (or adjusted)
    
    sc = ax.scatter(
        x=df_sorted[enrichment_score], 
        y=df_sorted[pathway], 
        s=df_sorted[count] * 5, # Scale size
        c=df_sorted[pvalue],
        cmap="Reds_r", # Lower p-value is redder/darker
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5
    )
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("P-value")
    
    # Size Legend - manual
    # This is tricky in matplotlib without seaborn scatterplot size handling
    # But basic scatter is fine.
    
    ax.set_title(title)
    ax.set_xlabel("Enrichment Score")
    sns.despine()
    return fig

@PlotRegistry.register(
    id="pathway_enrichment_bar",
    category="Pathways",
    display_name="Enrichment Barplot",
    required_columns=["pathway", "pvalue"],
    optional_columns=["count", "enrichment_score"],
    description="Barplot of enriched pathways.",
    supports_grouping=False
)
def plot_enrichment_bar(df: pd.DataFrame, pathway: str, pvalue: str, 
                        count: Optional[str] = None, enrichment_score: Optional[str] = None,
                        palette: str = "npg", title: str = "Pathway Enrichment", **kwargs) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Top 20
    df['neg_log10_pval'] = -np.log10(df[pvalue])
    df_sorted = df.sort_values(by='neg_log10_pval', ascending=False).head(20)
    
    sns.barplot(
        data=df_sorted, x='neg_log10_pval', y=pathway,
        palette="Reds_r",
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel("-log10 P-value")
    sns.despine()
    return fig

@PlotRegistry.register(
    id="pathway_radar",
    category="Pathways",
    display_name="Radar Plot",
    required_columns=["pathway", "score", "group"],
    optional_columns=[],
    description="Radar/Spider plot comparing pathway scores across groups.",
    supports_grouping=True
)
def plot_radar(df: pd.DataFrame, pathway: str, score: str, group: str,
               palette: str = "npg", title: str = "Pathway Radar", **kwargs) -> plt.Figure:
    
    # This requires specific aggregation: one score per pathway per group
    # If duplicates, aggregate
    
    pivot_df = df.pivot_table(index=group, columns=pathway, values=score)
    pivot_df = pivot_df.fillna(0)
    
    # Number of variables
    categories = list(pivot_df.columns)
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    colors = get_palette(palette)
    
    for i, (idx, row) in enumerate(pivot_df.iterrows()):
        values = row.tolist()
        values += values[:1] # Close the loop
        
        color = colors[i % len(colors)]
        
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=str(idx), color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    ax.set_title(title, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig
