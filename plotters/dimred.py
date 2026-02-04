import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional
from plot_registry import PlotRegistry
from utils.colors import get_palette

@PlotRegistry.register(
    id="dimred_pca",
    category="Dimensionality Reduction",
    display_name="PCA Plot",
    required_columns=["PC1", "PC2", "sample_id"],
    optional_columns=["group", "batch", "label"],
    description="Principal Component Analysis scatter plot.",
    supports_grouping=True
)
def plot_pca(df: pd.DataFrame, PC1: str, PC2: str, sample_id: str,
             group: Optional[str] = None, batch: Optional[str] = None, 
             label: Optional[str] = None, palette: str = "npg", 
             title: str = "PCA Plot", **kwargs) -> plt.Figure:
    """
    Generates a publication-ready PCA plot.
    """
    return _generic_dimred_plot(df, x=PC1, y=PC2, group=group, shape=batch, 
                                label=label, palette=palette, title=title, 
                                xlabel=kwargs.get("xlabel", "PC1"), 
                                ylabel=kwargs.get("ylabel", "PC2"))

@PlotRegistry.register(
    id="dimred_umap",
    category="Dimensionality Reduction",
    display_name="UMAP Plot",
    required_columns=["UMAP1", "UMAP2", "sample_id"],
    optional_columns=["group", "batch", "label"],
    description="Uniform Manifold Approximation and Projection plot.",
    supports_grouping=True
)
def plot_umap(df: pd.DataFrame, UMAP1: str, UMAP2: str, sample_id: str,
              group: Optional[str] = None, batch: Optional[str] = None,
              label: Optional[str] = None, palette: str = "npg", 
              title: str = "UMAP Plot", **kwargs) -> plt.Figure:
    return _generic_dimred_plot(df, x=UMAP1, y=UMAP2, group=group, shape=batch,
                                label=label, palette=palette, title=title,
                                xlabel="UMAP 1", ylabel="UMAP 2")

@PlotRegistry.register(
    id="dimred_tsne",
    category="Dimensionality Reduction",
    display_name="t-SNE Plot",
    required_columns=["tSNE1", "tSNE2", "sample_id"],
    optional_columns=["group", "batch", "label"],
    description="t-Distributed Stochastic Neighbor Embedding plot.",
    supports_grouping=True
)
def plot_tsne(df: pd.DataFrame, tSNE1: str, tSNE2: str, sample_id: str,
              group: Optional[str] = None, batch: Optional[str] = None,
              label: Optional[str] = None, palette: str = "npg", 
              title: str = "t-SNE Plot", **kwargs) -> plt.Figure:
    return _generic_dimred_plot(df, x=tSNE1, y=tSNE2, group=group, shape=batch, 
                                label=label, palette=palette, title=title,
                                xlabel="t-SNE 1", ylabel="t-SNE 2")

def _generic_dimred_plot(df: pd.DataFrame, x: str, y: str, group: Optional[str] = None, 
                         shape: Optional[str] = None, label: Optional[str] = None, 
                         palette: str = "npg", title: str = "", 
                         xlabel: str = "", ylabel: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = get_palette(palette)
    
    sns.scatterplot(
        data=df, x=x, y=y, hue=group, style=shape,
        palette=colors if group else None,
        s=80, alpha=0.8, edgecolor="black", linewidth=0.5, ax=ax
    )
    
    # Label top points if label column provided? 
    # For now, let's keep it clean or just handle legend
    if group:
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    sns.despine()
    plt.tight_layout()
    return fig
