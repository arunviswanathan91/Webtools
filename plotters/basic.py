import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, List
from plot_registry import PlotRegistry
from utils.colors import get_palette

# Set global style defaults
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.2)

@PlotRegistry.register(
    id="basic_scatter",
    category="Basic",
    display_name="Scatter Plot",
    required_columns=["x", "y"],
    optional_columns=["color", "size", "label"],
    description="Standard scatter plot for comparing two continuous variables.",
    supports_grouping=True
)
def plot_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, 
                 size: Optional[str] = None, label: Optional[str] = None, 
                 palette: str = "npg", title: str = "", **kwargs) -> plt.Figure:
    """
    Generates a publication-ready scatter plot.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Handle color palette
    colors = get_palette(palette)
    
    sns.scatterplot(
        data=df, x=x, y=y, hue=color, size=size,
        palette=colors if color else None,
        edgecolor="black", linewidth=0.5,
        alpha=0.8, ax=ax
    )
    
    if label and 'label' in df.columns:
        # Simple labeling for top points could be added here, 
        # but for basic scatter we might just let it be.
        pass

    ax.set_title(title if title else f"{x} vs {y}")
    sns.despine()
    return fig

@PlotRegistry.register(
    id="basic_bar",
    category="Basic",
    display_name="Bar Chart",
    required_columns=["x", "y"],
    optional_columns=["group"],
    description="Bar chart for comparing categories.",
    supports_grouping=True
)
def plot_bar(df: pd.DataFrame, x: str, y: str, group: Optional[str] = None, 
             palette: str = "npg", title: str = "", **kwargs) -> plt.Figure:
    """
    Generates a publication-ready bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = get_palette(palette)
    
    sns.barplot(
        data=df, x=x, y=y, hue=group,
        palette=colors if group else None,
        capsize=0.1, errwidth=1.5,
        ax=ax, edgecolor="black"
    )
    
    ax.set_title(title)
    sns.despine()
    return fig

@PlotRegistry.register(
    id="basic_box",
    category="Basic",
    display_name="Box Plot",
    required_columns=["x", "y"],
    optional_columns=["group"],
    description="Box plot for visualizing distribution statistics.",
    supports_grouping=True
)
def plot_box(df: pd.DataFrame, x: str, y: str, group: Optional[str] = None, 
             palette: str = "npg", title: str = "", **kwargs) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = get_palette(palette)
    
    sns.boxplot(
        data=df, x=x, y=y, hue=group,
        palette=colors if group else None,
        width=0.6, ax=ax, flierprops={"marker": "o", "markersize": 3}
    )
    
    ax.set_title(title)
    sns.despine()
    return fig

@PlotRegistry.register(
    id="basic_violin",
    category="Basic",
    display_name="Violin Plot",
    required_columns=["x", "y"],
    optional_columns=["group"],
    description="Violin plot for visualizing distribution density.",
    supports_grouping=True
)
def plot_violin(df: pd.DataFrame, x: str, y: str, group: Optional[str] = None, 
                palette: str = "npg", title: str = "", **kwargs) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = get_palette(palette)
    
    sns.violinplot(
        data=df, x=x, y=y, hue=group,
        palette=colors if group else None,
        ax=ax, inner="box", linewidth=1
    )
    
    ax.set_title(title)
    sns.despine()
    return fig

@PlotRegistry.register(
    id="basic_histogram",
    category="Basic",
    display_name="Histogram",
    required_columns=["x"],
    optional_columns=["group"],
    description="Histogram for distribution of a single variable.",
    supports_grouping=True
)
def plot_histogram(df: pd.DataFrame, x: str, group: Optional[str] = None, 
                   palette: str = "npg", title: str = "", bins: int = 30, **kwargs) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = get_palette(palette)
    
    sns.histplot(
        data=df, x=x, hue=group,
        palette=colors if group else None,
        bins=bins, kde=True, ax=ax, element="step"
    )
    
    ax.set_title(title)
    sns.despine()
    return fig
