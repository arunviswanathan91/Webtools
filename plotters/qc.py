import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional
from plot_registry import PlotRegistry
from utils.colors import get_palette

@PlotRegistry.register(
    id="qc_library_size",
    category="QC",
    display_name="Library Size",
    required_columns=["sample_id", "library_size"],
    optional_columns=["group"],
    description="Bar plot of library sizes (total counts) per sample.",
    supports_grouping=True
)
def plot_library_size(df: pd.DataFrame, sample_id: str, library_size: str, 
                      group: Optional[str] = None, palette: str = "npg", 
                      title: str = "Library Size per Sample", **kwargs) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_palette(palette)
    
    # Sort by library size for better visualization
    df_sorted = df.sort_values(by=library_size, ascending=False)
    
    sns.barplot(
        data=df_sorted, x=sample_id, y=library_size, hue=group,
        palette=colors if group else None,
        ax=ax
    )
    
    # Rotate x labels if many samples
    if len(df) > 10:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
    ax.set_title(title)
    ax.set_ylabel("Total Counts")
    sns.despine()
    plt.tight_layout()
    return fig

@PlotRegistry.register(
    id="qc_gene_detection",
    category="QC",
    display_name="Gene Detection Rate",
    required_columns=["sample_id", "detected_genes"],
    optional_columns=["group"],
    description="Number of genes detected per sample.",
    supports_grouping=True
)
def plot_gene_detection(df: pd.DataFrame, sample_id: str, detected_genes: str, 
                        group: Optional[str] = None, palette: str = "npg", 
                        title: str = "Detected Genes per Sample", **kwargs) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_palette(palette)
    
    df_sorted = df.sort_values(by=detected_genes, ascending=False)
    
    sns.barplot(
        data=df_sorted, x=sample_id, y=detected_genes, hue=group,
        palette=colors if group else None,
        ax=ax
    )
    
    if len(df) > 10:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
    ax.set_title(title)
    ax.set_ylabel("Number of Genes")
    sns.despine()
    plt.tight_layout()
    return fig

@PlotRegistry.register(
    id="qc_mito_content",
    category="QC",
    display_name="Mitochondrial Content",
    required_columns=["sample_id", "mito_percent"],
    optional_columns=["group", "threshold"],
    description="Percentage of mitochondrial reads per sample.",
    supports_grouping=True
)
def plot_mito_content(df: pd.DataFrame, sample_id: str, mito_percent: str, 
                      group: Optional[str] = None, palette: str = "npg", 
                      title: str = "Mitochondrial Content", threshold: float = 5.0, 
                      **kwargs) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_palette(palette)
    
    # Violin plot is often better for mito content if many samples, but box/scatter is standard QC
    # Here we use box plot + strip plot hybrid if supported, or just box
    # If samples are distinct, bar might be better. 
    # Let's stick to bar for individual samples, or box if grouped by condition.
    # The requirement is "per sample" usually in QC steps.
    
    # If many samples, a distribution (violin/box) across groups is better. 
    # If few samples, bar plot. 
    # Let's interpret "sample_id" as unique. 
    
    if len(df) > 20 and group:
        # Grouped distribution
        sns.violinplot(data=df, x=group, y=mito_percent, palette=colors, ax=ax)
        sns.stripplot(data=df, x=group, y=mito_percent, color="black", size=3, alpha=0.5, ax=ax)
    else:
        # Individual bars
        sns.barplot(
            data=df, x=sample_id, y=mito_percent, hue=group, 
            palette=colors if group else None, ax=ax
        )
        if len(df) > 10:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
    # Add threshold line
    if threshold:
        ax.axhline(float(threshold), color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold}%)')
        ax.legend()

    ax.set_title(title)
    ax.set_ylabel("Mitochondrial %")
    sns.despine()
    plt.tight_layout()
    return fig
