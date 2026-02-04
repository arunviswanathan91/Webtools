import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional
from plot_registry import PlotRegistry
from utils.colors import get_palette

try:
    from lifelines import KaplanMeierFitter
except ImportError:
    KaplanMeierFitter = None

@PlotRegistry.register(
    id="clinical_km",
    category="Clinical",
    display_name="Kaplan-Meier Curve",
    required_columns=["time", "event"],
    optional_columns=["group"],
    description="Survival analysis curve.",
    supports_grouping=True
)
def plot_kaplan_meier(df: pd.DataFrame, time: str, event: str,
                      group: Optional[str] = None, palette: str = "npg", 
                      title: str = "Survival Analysis", **kwargs) -> plt.Figure:
    
    if KaplanMeierFitter is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "lifelines package not installed", ha='center', va='center')
        return fig
        
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = get_palette(palette)
    
    kmf = KaplanMeierFitter()
    
    if group:
        groups = df[group].unique()
        for i, g in enumerate(groups):
            mask = df[group] == g
            color = colors[i % len(colors)]
            kmf.fit(df.loc[mask, time], df.loc[mask, event], label=str(g))
            kmf.plot_survival_function(ax=ax, ci_show=False, color=color) # ci_show=True often messy with many groups
    else:
        kmf.fit(df[time], df[event], label="All")
        kmf.plot_survival_function(ax=ax)
        
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    sns.despine()
    return fig

@PlotRegistry.register(
    id="clinical_roc",
    category="Clinical",
    display_name="ROC Curve",
    required_columns=["fpr", "tpr"],
    optional_columns=["group", "auc"],
    description="Receiver Operating Characteristic curve.",
    supports_grouping=True
)
def plot_roc(df: pd.DataFrame, fpr: str, tpr: str,
             group: Optional[str] = None, auc: Optional[str] = None,
             palette: str = "npg", title: str = "ROC Curve", **kwargs) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = get_palette(palette)
    
    # If grouping, expected that df has multiple rows per group tracing the curve
    if group:
        # We need to iterate groups
        groups = df[group].unique()
        for i, g in enumerate(groups):
            sub_df = df[df[group] == g].sort_values(by=fpr)
            label = str(g)
            if auc and not sub_df[auc].isna().all():
                 # Use first AUC value found for group
                 auc_val = sub_df[auc].iloc[0]
                 label += f" (AUC={auc_val:.2f})"
                 
            ax.plot(sub_df[fpr], sub_df[tpr], label=label, color=colors[i % len(colors)])
    else:
        df_sorted = df.sort_values(by=fpr)
        label = "ROC"
        if auc:
            auc_val = df_sorted[auc].iloc[0]
            label += f" (AUC={auc_val:.2f})"
        ax.plot(df_sorted[fpr], df_sorted[tpr], label=label, color=colors[0])
        
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    sns.despine()
    return fig

@PlotRegistry.register(
    id="clinical_cox_forest",
    category="Clinical",
    display_name="Cox Forest Plot",
    required_columns=["variable", "coef", "exp_coef", "se_coef", "p"],
    optional_columns=["group"],
    description="Forest plot for Cox Proportional Hazards model.",
    supports_grouping=False
)
def plot_cox_forest(df: pd.DataFrame, variable: str, coef: str, exp_coef: str, se_coef: str, p: str,
                    group: Optional[str] = None, palette: str = "npg", 
                    title: str = "Cox Regression Results", **kwargs) -> plt.Figure:
    
    # Calculate CI for exp_coef (Hazard Ratio)
    # HR typically: exp(coef +/- 1.96*se)
    # But usually exp_coef is provided. CI lower = exp(coef - 1.96*se), upper = exp(coef + 1.96*se)
    
    df['ci_lower'] = np.exp(df[coef] - 1.96 * df[se_coef])
    df['ci_upper'] = np.exp(df[coef] + 1.96 * df[se_coef])
    
    # Use standard forest plot
    from plotters.stats import plot_forest
    return plot_forest(df, label=variable, mean=exp_coef, ci_lower='ci_lower', ci_upper='ci_upper', 
                       pvalue=p, palette=palette, title=title)
