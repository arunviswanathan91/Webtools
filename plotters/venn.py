import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from plot_registry import PlotRegistry
from utils.colors import get_palette

try:
    from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
    HAS_VENN = True
except ImportError:
    HAS_VENN = False

@PlotRegistry.register(
    id="basic_venn",
    category="Basic",
    display_name="Venn Diagram",
    required_columns=["item", "set"],
    optional_columns=[],
    description="Venn diagram for 2 or 3 sets. Data should be in long format (one column for items, one for set names).",
    supports_grouping=False
)
def plot_venn(df: pd.DataFrame, item: str, set: str, palette: str = "npg", 
              title: str = "Venn Diagram", **kwargs) -> plt.Figure:
    
    if not HAS_VENN:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "matplotlib-venn not installed", ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = get_palette(palette)
    
    # Get unique sets
    unique_sets = sorted(df[set].dropna().unique())
    num_sets = len(unique_sets)
    
    if num_sets < 2 or num_sets > 3:
        ax.text(0.5, 0.5, f"Venn functionality supports 2 or 3 sets. Found {num_sets}: {unique_sets}", 
                ha='center', va='center', wrap=True)
        return fig
        
    # Create sets
    sets_dict = {}
    for s_name in unique_sets:
        sets_dict[s_name] = set(df[df[set] == s_name][item].dropna().astype(str))
    
    set_values = list(sets_dict.values())
    set_labels = list(sets_dict.keys())
    
    # Transparency
    alpha = kwargs.get("alpha", 0.5)
    
    if num_sets == 2:
        v = venn2(
            subsets=set_values,
            set_labels=set_labels,
            set_colors=[colors[i % len(colors)] for i in range(num_sets)],
            alpha=alpha,
            ax=ax
        )
        # Outline circles
        venn2_circles(subsets=set_values, linewidth=1, ax=ax)
        
    elif num_sets == 3:
        v = venn3(
            subsets=set_values,
            set_labels=set_labels,
            set_colors=[colors[i % len(colors)] for i in range(num_sets)],
            alpha=alpha,
            ax=ax
        )
        venn3_circles(subsets=set_values, linewidth=1, ax=ax)
        
    ax.set_title(title)
    return fig
