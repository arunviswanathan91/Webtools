import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from plot_registry import PlotRegistry
from utils.colors import get_palette

@PlotRegistry.register(
    id="de_volcano",
    category="Differential Expression",
    display_name="Volcano Plot",
    required_columns=["gene", "log2FC", "pvalue"],
    optional_columns=["padj", "significant", "label"],
    description="Scatter plot of log2 Fold Change vs -log10 P-value.",
    supports_grouping=False
)
def plot_volcano(df: pd.DataFrame, gene: str, log2FC: str, pvalue: str,
                 padj: Optional[str] = None, significant: Optional[str] = None,
                 label: Optional[str] = None, palette: str = "npg", 
                 title: str = "Volcano Plot", 
                 pvalue_threshold: float = 0.05, logfc_threshold: float = 1.0,
                 **kwargs) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate -log10 pvalue
    df['neg_log10_pval'] = -np.log10(df[pvalue])
    
    # Determine significance if not provided
    if significant is None:
        df['sig_status'] = 'Not Significant'
        df.loc[(df[pvalue] < pvalue_threshold) & (df[log2FC] > logfc_threshold), 'sig_status'] = 'Up'
        df.loc[(df[pvalue] < pvalue_threshold) & (df[log2FC] < -logfc_threshold), 'sig_status'] = 'Down'
    else:
        df['sig_status'] = df[significant]
        
    # Custom color palette for Up/Down
    # Usually: Up=Red, Down=Blue, NS=Grey
    volcano_palette = {"Up": "#CE3D32", "Down": "#466983", "Not Significant": "#B0B0B0"}
    
    sns.scatterplot(
        data=df, x=log2FC, y='neg_log10_pval', hue='sig_status',
        palette=volcano_palette, alpha=0.7, s=40,
        linewidth=0, ax=ax
    )
    
    # Add threshold lines
    ax.axhline(-np.log10(pvalue_threshold), color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(logfc_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(-logfc_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Label top genes
    # If label column exists, label top significant genes
    if label and significant is None: # Auto-labeling logic if no pre-defined labels
        # Pick top 5 up and top 5 down by pvalue
        up_genes = df[df['sig_status'] == 'Up'].nsmallest(5, pvalue)
        down_genes = df[df['sig_status'] == 'Down'].nsmallest(5, pvalue)
        
        texts = []
        for _, row in pd.concat([up_genes, down_genes]).iterrows():
            texts.append(ax.text(row[log2FC], row['neg_log10_pval'], row[gene], fontsize=8))
            
        # Note: adjustText is great here but likely not in std lib, skipping for now
    
    ax.set_title(title)
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10 P-value")
    sns.despine()
    
    # Clean up legend
    try:
        ax.legend(title="Significance")
    except:
        pass
        
    return fig

@PlotRegistry.register(
    id="de_ma_plot",
    category="Differential Expression",
    display_name="MA Plot",
    required_columns=["gene", "log2FC", "baseMean"],
    optional_columns=["padj", "significant"],
    description="Scatter plot of log mean expression vs log2 Fold Change.",
    supports_grouping=False
)
def plot_ma(df: pd.DataFrame, gene: str, log2FC: str, baseMean: str,
            padj: Optional[str] = None, significant: Optional[str] = None,
            palette: str = "npg", title: str = "MA Plot", 
            padj_threshold: float = 0.05, **kwargs) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Log10 baseMean for x-axis if not already logged (usually MA plot uses log scale axis)
    # But usually baseMean is raw counts.
    
    if significant is None:
        if padj:
             df['sig_status'] = df[padj].apply(lambda x: 'Significant' if x < padj_threshold else 'NS')
        else:
            df['sig_status'] = 'Unknown'
    else:
        df['sig_status'] = df[significant]
        
    ma_palette = {"Significant": "#CE3D32", "NS": "#B0B0B0", "Unknown": "grey"}

    sns.scatterplot(
        data=df, x=baseMean, y=log2FC, hue='sig_status',
        palette=ma_palette, alpha=0.6, s=20, linewidth=0, ax=ax
    )
    
    ax.set_xscale("log")
    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    
    ax.set_title(title)
    ax.set_xlabel("Mean Expression")
    ax.set_ylabel("log2 Fold Change")
    sns.despine()
    return fig

@PlotRegistry.register(
    id="de_ranked_genes",
    category="Differential Expression",
    display_name="Ranked Genes",
    required_columns=["gene", "rank_metric"],
    optional_columns=["highlight"],
    description="Waterfals plot of ranked genes.",
    supports_grouping=False
)
def plot_ranked_genes(df: pd.DataFrame, gene: str, rank_metric: str,
                      highlight: Optional[str] = None, palette: str = "npg",
                      title: str = "Ranked Genes", **kwargs) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = get_palette(palette)
    
    # Sort
    df_sorted = df.sort_values(by=rank_metric, ascending=False).reset_index(drop=True)
    df_sorted['rank'] = df_sorted.index
    
    # Scatter all points
    ax.scatter(df_sorted['rank'], df_sorted[rank_metric], color=colors[0], s=5, alpha=0.5)
    
    # Highlight specific genes if requested (not yet implemented in detail)
    # This acts as the backbone for GSEA plots essentially.
    
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Metric")
    sns.despine()
    return fig
