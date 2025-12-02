# streamlit_freedman_lane_app.py
"""
Generic Streamlit app for Freedman-Lane permutation adjustment and statistical analysis.

Features:
- Works with ANY expression/feature data and clinical metadata
- Flexible data format support (wide or long)
- Configurable Freedman-Lane permutation tests
- Multiple effect size measures and bootstrap options
- Comprehensive progress tracking
- Parallel processing support

Usage:
$ pip install -r requirements.txt
$ streamlit run streamlit_freedman_lane_app.py

Requirements:
streamlit
pandas
numpy
statsmodels
joblib
scipy
"""

import io
import os
import math
import time
import tempfile
import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy import stats

warnings.filterwarnings("ignore")

# ---------- Configuration ----------

class AnalysisConfig:
    """Configuration container for analysis parameters"""
    def __init__(self):
        self.n_perm = 2000
        self.n_boot = 1000
        self.min_n_per_group = 8
        self.workers = 1
        self.alpha = 0.05
        self.bootstrap_ci_method = 'percentile'  # or 'bca'
        self.effect_size_measures = ['cohens_d']  # can add 'hedges_g', 'glass_delta'
        self.seed = 42
        self.show_progress = True


# ---------- Utilities ----------

def smart_id_extraction(sample_id: Any, delimiter: str = '-') -> str:
    """
    Flexible ID extraction that handles various formats.
    Can handle semicolon-separated, hyphen-separated, underscore-separated IDs.
    """
    if pd.isna(sample_id):
        return ""
    s = str(sample_id).strip()
    
    # Handle semicolon-separated (take first)
    if ';' in s:
        s = s.split(';')[0].strip()
    
    # Handle delimiter-based extraction
    if delimiter in s:
        parts = s.split(delimiter)
        # Keep first 2 parts if available
        if len(parts) >= 2:
            return f"{parts[0]}{delimiter}{parts[1]}"
    
    return s


def detect_numeric_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """
    Detect numeric columns that could be features.
    More robust than checking for specific patterns.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    exclude_set = set(exclude_cols)
    # Common clinical/metadata column patterns to exclude
    exclude_patterns = ['id', 'sample', 'patient', 'subject', 'group', 'class', 'label']
    
    numeric_cols = []
    for col in df.columns:
        if col in exclude_set:
            continue
        
        # Check if column name suggests it's metadata
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in exclude_patterns):
            continue
        
        # Check if numeric
        if np.issubdtype(df[col].dtype, np.number):
            numeric_cols.append(col)
    
    return numeric_cols


def infer_data_format(df: pd.DataFrame) -> str:
    """
    Infer whether data is in wide or long format.
    Long format typically has columns like: sample_id, feature, value
    Wide format has: sample_id, feature1, feature2, ..., featureN
    """
    # Check for common long format indicators
    col_names_lower = [c.lower() for c in df.columns]
    
    long_indicators = ['feature', 'gene', 'protein', 'metabolite', 'variable']
    value_indicators = ['value', 'expression', 'abundance', 'intensity', 'z']
    
    has_feature_col = any(ind in col_names_lower for ind in long_indicators)
    has_value_col = any(ind in col_names_lower for ind in value_indicators)
    
    if has_feature_col and has_value_col:
        return 'long'
    
    # If many numeric columns, likely wide format
    numeric_cols = detect_numeric_columns(df)
    if len(numeric_cols) > 5:
        return 'wide'
    
    return 'unknown'


# ---------- Statistical Functions ----------

def compute_cohens_d(a, b):
    """Compute Cohen's d effect size"""
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    
    n1, n2 = len(a), len(b)
    m1, m2 = np.nanmean(a), np.nanmean(b)
    v1 = np.nanvar(a, ddof=1) if n1 > 1 else 0.0
    v2 = np.nanvar(b, ddof=1) if n2 > 1 else 0.0
    
    # Pooled standard deviation
    denom = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2) if (n1 + n2 - 2) > 0 else 0.0
    sd_pooled = math.sqrt(denom) if denom > 0 else 0.0
    
    if sd_pooled == 0:
        return np.nan
    
    return (m2 - m1) / sd_pooled


def compute_hedges_g(a, b):
    """Compute Hedges' g (bias-corrected Cohen's d)"""
    d = compute_cohens_d(a, b)
    if np.isnan(d):
        return np.nan
    
    n1, n2 = len(a), len(b)
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return d * correction


def bootstrap_effect_size(resid_vals, group_labels, effect_measure='cohens_d', 
                          n_boot=1000, ci_level=0.95, seed=42, 
                          progress_callback=None):
    """
    Bootstrap effect size estimation with confidence intervals.
    
    Parameters:
    -----------
    resid_vals : array-like
        Residual values
    group_labels : array-like
        Group labels
    effect_measure : str
        'cohens_d' or 'hedges_g'
    n_boot : int
        Number of bootstrap iterations
    ci_level : float
        Confidence interval level (default 0.95 for 95% CI)
    seed : int
        Random seed
    progress_callback : callable
        Optional callback for progress updates
    """
    resid_vals = np.asarray(resid_vals)
    labels = np.asarray(group_labels)
    
    # Get unique groups
    uniq = np.unique(labels[~pd.isna(labels)])
    if len(uniq) != 2:
        return {
            "effect_size": np.nan, 
            "ci_low": np.nan, 
            "ci_high": np.nan, 
            "n_boot": 0,
            "measure": effect_measure
        }
    
    a_idx = np.where(labels == uniq[0])[0]
    b_idx = np.where(labels == uniq[1])[0]
    n1, n2 = len(a_idx), len(b_idx)
    
    if n1 < 2 or n2 < 2:
        return {
            "effect_size": np.nan, 
            "ci_low": np.nan, 
            "ci_high": np.nan, 
            "n_boot": 0,
            "measure": effect_measure
        }
    
    rng = np.random.RandomState(seed)
    boot_effects = []
    
    # Choose effect size function
    if effect_measure == 'hedges_g':
        effect_func = compute_hedges_g
    else:
        effect_func = compute_cohens_d
    
    # Bootstrap loop with progress tracking
    for i in range(n_boot):
        # Resample within each group
        boot_a_idx = rng.choice(a_idx, size=n1, replace=True)
        boot_b_idx = rng.choice(b_idx, size=n2, replace=True)
        
        boot_a = resid_vals[boot_a_idx]
        boot_b = resid_vals[boot_b_idx]
        
        effect = effect_func(boot_a, boot_b)
        if np.isfinite(effect):
            boot_effects.append(effect)
        
        # Progress callback
        if progress_callback and i % max(1, n_boot // 20) == 0:
            progress_callback(i / n_boot)
    
    boot_effects = np.array(boot_effects)
    
    if boot_effects.size < max(50, int(0.1 * n_boot)):
        # Not enough valid bootstrap samples
        obs_effect = effect_func(resid_vals[a_idx], resid_vals[b_idx])
        return {
            "effect_size": float(obs_effect), 
            "ci_low": np.nan, 
            "ci_high": np.nan, 
            "n_boot": int(boot_effects.size),
            "measure": effect_measure
        }
    
    # Compute observed effect
    obs_effect = effect_func(resid_vals[a_idx], resid_vals[b_idx])
    
    # Compute confidence intervals
    alpha = 1 - ci_level
    ci_low = np.percentile(boot_effects, 100 * alpha / 2)
    ci_high = np.percentile(boot_effects, 100 * (1 - alpha / 2))
    
    return {
        "effect_size": float(obs_effect),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_boot": int(len(boot_effects)),
        "measure": effect_measure
    }


def fit_covariate_model(df, outcome_col, covariates):
    """
    Fit model with only covariates to get residuals.
    Returns fitted values and residuals.
    """
    if not covariates or outcome_col not in df.columns:
        # No covariates: return mean as fitted
        mean_val = df[outcome_col].mean()
        fitted = np.repeat(mean_val, len(df))
        resid = df[outcome_col].values - fitted
        return fitted, resid
    
    # Build formula
    terms = []
    for c in covariates:
        if c not in df.columns:
            continue
        if df[c].dtype == 'object' or df[c].nunique() < 10:
            terms.append(f"C({c})")
        else:
            terms.append(c)
    
    if not terms:
        mean_val = df[outcome_col].mean()
        fitted = np.repeat(mean_val, len(df))
        resid = df[outcome_col].values - fitted
        return fitted, resid
    
    formula = f"{outcome_col} ~ " + " + ".join(terms)
    
    try:
        # Fit model on complete cases
        complete_df = df.dropna(subset=[outcome_col] + covariates)
        model = smf.ols(formula=formula, data=complete_df).fit()
        
        # Predict for all rows
        try:
            fitted = model.predict(df)
            resid = df[outcome_col].values - fitted
        except Exception:
            # Handle prediction errors by using mean imputation
            fitted = np.repeat(np.nan, len(df))
            complete_mask = df[covariates].notna().all(axis=1)
            fitted[complete_mask] = model.fittedvalues.values
            mean_fitted = np.nanmean(fitted)
            fitted = np.nan_to_num(fitted, nan=mean_fitted)
            resid = df[outcome_col].values - fitted
        
        return fitted, resid
    
    except Exception as e:
        # Fallback to mean
        mean_val = df[outcome_col].mean()
        fitted = np.repeat(mean_val, len(df))
        resid = df[outcome_col].values - fitted
        return fitted, resid


def run_freedman_lane(df: pd.DataFrame, outcome_col: str, group_col: str, 
                      covariates: List[str], groupA: Any, groupB: Any,
                      n_perm: int = 2000, seed: int = 42,
                      progress_callback=None) -> Dict[str, Any]:
    """
    Freedman-Lane permutation test for group differences adjusted for covariates.
    
    The Freedman-Lane procedure:
    1. Fit reduced model (covariates only) to get residuals
    2. Fit full model (group + covariates) to get observed test statistic
    3. Permute residuals, add to fitted values from reduced model
    4. Refit full model on permuted data to get null distribution
    5. Compute p-value
    """
    df = df.copy().reset_index(drop=True)
    
    # Step 1: Fit reduced model (covariates only)
    fitted_reduced, residuals = fit_covariate_model(df, outcome_col, covariates)
    
    # Step 2: Fit full model (group + covariates)
    terms_full = [f"C({group_col})"]
    for c in covariates:
        if c in df.columns:
            if df[c].dtype == 'object' or df[c].nunique() < 10:
                terms_full.append(f"C({c})")
            else:
                terms_full.append(c)
    
    formula_full = f"{outcome_col} ~ " + " + ".join(terms_full)
    
    try:
        # Fit on complete cases
        full_df = df.dropna(subset=[outcome_col, group_col])
        model_full = smf.ols(formula=formula_full, data=full_df).fit()
        
        # Compute mean values for covariates (for prediction)
        cov_means = {}
        for c in covariates:
            if c in full_df.columns:
                if full_df[c].dtype.kind in 'biufc':
                    cov_means[c] = float(full_df[c].mean())
                else:
                    mode_val = full_df[c].mode()
                    cov_means[c] = mode_val.iloc[0] if len(mode_val) > 0 else full_df[c].dropna().iloc[0]
        
        # Predict for both groups
        row_A = {group_col: groupA, **cov_means}
        row_B = {group_col: groupB, **cov_means}
        
        pred_A = float(model_full.predict(pd.DataFrame([row_A]))[0])
        pred_B = float(model_full.predict(pd.DataFrame([row_B]))[0])
        obs_stat = pred_B - pred_A
        
    except Exception as e:
        # Fallback: simple difference of means
        try:
            vals = df.dropna(subset=[outcome_col, group_col])
            mean_A = vals[vals[group_col] == groupA][outcome_col].mean()
            mean_B = vals[vals[group_col] == groupB][outcome_col].mean()
            obs_stat = mean_B - mean_A
        except Exception:
            return {
                "obs_stat": np.nan,
                "p_value": np.nan,
                "perm_dist": np.array([]),
                "n_perm_valid": 0
            }
    
    # Step 3-4: Permutation loop
    rng = np.random.RandomState(seed)
    perm_stats = []
    n = len(df)
    
    for i in range(n_perm):
        # Permute residuals
        perm_idx = rng.permutation(n)
        perm_resid = residuals[perm_idx]
        
        # Add to fitted values from reduced model
        Y_perm = fitted_reduced + perm_resid
        
        # Create permuted dataframe
        df_perm = df.copy()
        df_perm[outcome_col] = Y_perm
        
        # Refit full model
        try:
            model_perm = smf.ols(
                formula=formula_full, 
                data=df_perm.dropna(subset=[outcome_col, group_col])
            ).fit()
            
            pred_A_perm = float(model_perm.predict(pd.DataFrame([row_A]))[0])
            pred_B_perm = float(model_perm.predict(pd.DataFrame([row_B]))[0])
            stat_perm = pred_B_perm - pred_A_perm
            
            if np.isfinite(stat_perm):
                perm_stats.append(stat_perm)
        except Exception:
            continue
        
        # Progress callback
        if progress_callback and i % max(1, n_perm // 20) == 0:
            progress_callback(i / n_perm)
    
    perm_stats = np.array(perm_stats)
    
    # Step 5: Compute p-value
    if perm_stats.size == 0 or not np.isfinite(obs_stat):
        return {
            "obs_stat": obs_stat,
            "p_value": np.nan,
            "perm_dist": perm_stats,
            "n_perm_valid": 0
        }
    
    # Two-tailed p-value
    p_value = (np.sum(np.abs(perm_stats) >= abs(obs_stat)) + 1) / (len(perm_stats) + 1)
    
    return {
        "obs_stat": float(obs_stat),
        "p_value": float(p_value),
        "perm_dist": perm_stats,
        "n_perm_valid": int(len(perm_stats))
    }


def analyze_single_feature(feature_name: str, df_feature: pd.DataFrame, 
                           outcome_col: str, group_col: str, covariates: List[str],
                           groupA: Any, groupB: Any, comparison_name: str,
                           config: AnalysisConfig,
                           progress_callback=None) -> Optional[Dict[str, Any]]:
    """
    Analyze a single feature: run Freedman-Lane test and bootstrap effect sizes.
    """
    try:
        # Filter to relevant groups
        df_sub = df_feature[df_feature[group_col].isin([groupA, groupB])].copy()
        
        nA = int((df_sub[group_col] == groupA).sum())
        nB = int((df_sub[group_col] == groupB).sum())
        
        if nA < config.min_n_per_group or nB < config.min_n_per_group:
            return None
        
        # Get residuals for effect size calculation
        fitted, residuals = fit_covariate_model(df_sub, outcome_col, covariates)
        
        # Run Freedman-Lane permutation test
        perm_result = run_freedman_lane(
            df_sub, outcome_col, group_col, covariates,
            groupA, groupB, n_perm=config.n_perm, seed=config.seed,
            progress_callback=lambda p: progress_callback('perm', p) if progress_callback else None
        )
        
        # Bootstrap effect sizes
        effect_results = {}
        for measure in config.effect_size_measures:
            boot_result = bootstrap_effect_size(
                residuals, df_sub[group_col].values,
                effect_measure=measure, n_boot=config.n_boot,
                ci_level=1-config.alpha, seed=config.seed,
                progress_callback=lambda p: progress_callback('boot', p) if progress_callback else None
            )
            effect_results[measure] = boot_result
        
        # Compile results
        result = {
            'feature': feature_name,
            'comparison': comparison_name,
            'groupA': str(groupA),
            'groupB': str(groupB),
            'nA': nA,
            'nB': nB,
            'obs_diff': perm_result['obs_stat'],
            'p_perm': perm_result['p_value'],
            'n_perm_valid': perm_result['n_perm_valid'],
            'covariates_used': ','.join(covariates) if covariates else 'none'
        }
        
        # Add effect size results
        for measure, res in effect_results.items():
            result[f'{measure}'] = res['effect_size']
            result[f'{measure}_ci_low'] = res['ci_low']
            result[f'{measure}_ci_high'] = res['ci_high']
            result[f'{measure}_n_boot'] = res['n_boot']
        
        return result
        
    except Exception as e:
        return {
            'feature': feature_name,
            'comparison': comparison_name,
            'error': str(e)[:200]
        }


# ---------- Streamlit App ----------

def main():
    st.set_page_config(
        page_title='Generic Freedman-Lane Analysis',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    st.title('🧬 Freedman-Lane Permutation Analysis')
    st.markdown("""
    ### Generic statistical analysis for expression/feature data with covariate adjustment
    
    This app implements the **Freedman-Lane permutation procedure** for testing group differences 
    while adjusting for covariates, plus bootstrap estimation of effect sizes.
    
    **Supports:**
    - Any type of expression/feature data (genes, proteins, metabolites, signatures, etc.)
    - Wide or long data formats
    - Multiple group comparisons
    - Flexible covariate adjustment
    - Parallel processing for speed
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header('⚙️ Configuration')
        
        st.subheader('Analysis Parameters')
        config = AnalysisConfig()
        
        config.n_perm = st.number_input(
            'Number of permutations',
            min_value=100, max_value=10000, value=2000, step=100,
            help='More permutations = more accurate p-values but slower'
        )
        
        config.n_boot = st.number_input(
            'Number of bootstrap iterations',
            min_value=100, max_value=10000, value=1000, step=100,
            help='For confidence intervals around effect sizes'
        )
        
        config.min_n_per_group = st.number_input(
            'Minimum samples per group',
            min_value=2, max_value=50, value=8, step=1,
            help='Features with fewer samples will be skipped'
        )
        
        config.alpha = st.number_input(
            'Significance level (α)',
            min_value=0.001, max_value=0.2, value=0.05, step=0.01,
            help='For FDR correction and confidence intervals'
        )
        
        config.workers = st.number_input(
            'Parallel workers',
            min_value=1, max_value=16, value=4, step=1,
            help='Number of CPU cores to use'
        )
        
        st.subheader('Effect Size Measures')
        effect_options = st.multiselect(
            'Select effect size measures',
            options=['cohens_d', 'hedges_g'],
            default=['cohens_d'],
            help="Cohen's d is standard; Hedges' g applies bias correction for small samples"
        )
        config.effect_size_measures = effect_options if effect_options else ['cohens_d']
        
        config.seed = st.number_input(
            'Random seed',
            min_value=0, max_value=9999, value=42, step=1,
            help='For reproducibility'
        )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(['📁 Data Upload', '🔧 Configure Analysis', '📊 Results'])
    
    with tab1:
        st.header('Data Upload')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Expression/Feature Data')
            uploaded_features = st.file_uploader(
                'Upload feature data (CSV)',
                type=['csv'],
                help='Can be wide format (samples × features) or long format (sample, feature, value)'
            )
            
            if uploaded_features:
                try:
                    features_df = pd.read_csv(uploaded_features)
                    st.success(f'✓ Loaded {features_df.shape[0]} rows × {features_df.shape[1]} columns')
                    
                    # Infer format
                    inferred_format = infer_data_format(features_df)
                    st.info(f'Inferred format: **{inferred_format}**')
                    
                    with st.expander('Preview data'):
                        st.dataframe(features_df.head(20))
                    
                    # Store in session state
                    st.session_state['features_df'] = features_df
                    st.session_state['inferred_format'] = inferred_format
                    
                except Exception as e:
                    st.error(f'Error loading file: {e}')
        
        with col2:
            st.subheader('Clinical/Metadata (Optional)')
            uploaded_clinical = st.file_uploader(
                'Upload clinical data (CSV)',
                type=['csv'],
                help='Optional if feature data already contains clinical columns'
            )
            
            if uploaded_clinical:
                try:
                    clinical_df = pd.read_csv(uploaded_clinical)
                    st.success(f'✓ Loaded {clinical_df.shape[0]} rows × {clinical_df.shape[1]} columns')
                    
                    with st.expander('Preview clinical data'):
                        st.dataframe(clinical_df.head(20))
                    
                    st.session_state['clinical_df'] = clinical_df
                    
                except Exception as e:
                    st.error(f'Error loading file: {e}')
            else:
                st.session_state['clinical_df'] = None
    
    with tab2:
        st.header('Configure Analysis')
        
        if 'features_df' not in st.session_state:
            st.warning('⚠️ Please upload feature data first')
            st.stop()
        
        features_df = st.session_state['features_df']
        clinical_df = st.session_state.get('clinical_df', None)
        inferred_format = st.session_state.get('inferred_format', 'unknown')
        
        # Data format selection
        st.subheader('1️⃣ Data Format')
        format_choice = st.radio(
            'Confirm data format',
            options=['wide', 'long', 'auto-detect'],
            index=0 if inferred_format == 'wide' else (1 if inferred_format == 'long' else 2),
            horizontal=True
        )
        
        if format_choice == 'auto-detect':
            format_choice = inferred_format
        
        is_long_format = (format_choice == 'long')
        
        # Column selection
        st.subheader('2️⃣ Column Identification')
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_id_col = st.selectbox(
                'Sample ID column',
                options=features_df.columns.tolist(),
                help='Column containing sample identifiers'
            )
            
            if is_long_format:
                feature_col = st.selectbox(
                    'Feature name column',
                    options=[c for c in features_df.columns if c != sample_id_col],
                    help='Column containing feature names (genes, proteins, etc.)'
                )
                
                value_col = st.selectbox(
                    'Value column',
                    options=[c for c in features_df.columns if c not in [sample_id_col, feature_col]],
                    help='Column containing expression/abundance values'
                )
        
        with col2:
            if not is_long_format:
                # Wide format: detect or select feature columns
                detected_features = detect_numeric_columns(
                    features_df,
                    exclude_cols=[sample_id_col]
                )
                
                st.write(f'Detected {len(detected_features)} numeric feature columns')
                
                use_all_features = st.checkbox(
                    'Use all detected numeric columns as features',
                    value=True
                )
                
                if not use_all_features:
                    selected_features = st.multiselect(
                        'Select feature columns',
                        options=detected_features,
                        default=detected_features[:min(10, len(detected_features))],
                        help='Choose specific columns to analyze'
                    )
                else:
                    selected_features = detected_features
                
                if len(selected_features) == 0:
                    st.error('No feature columns selected!')
                    st.stop()
        
        # ID extraction options
        st.subheader('3️⃣ ID Processing')
        extract_base_id = st.checkbox(
            'Extract base ID from sample identifiers',
            value=True,
            help='Useful when sample IDs have suffixes (e.g., SAMPLE-01-A → SAMPLE-01)'
        )
        
        if extract_base_id:
            id_delimiter = st.text_input(
                'ID delimiter for extraction',
                value='-',
                help='Character used to split IDs'
            )
        else:
            id_delimiter = None
        
        # Merge data
        st.subheader('4️⃣ Data Merging')
        
        with st.spinner('Preparing data...'):
            # Convert to long format if needed
            if is_long_format:
                df_long = features_df.rename(columns={
                    sample_id_col: 'sample_id_original',
                    feature_col: 'feature',
                    value_col: 'value'
                }).copy()
            else:
                # Melt wide to long
                df_long = features_df.melt(
                    id_vars=[sample_id_col],
                    value_vars=selected_features,
                    var_name='feature',
                    value_name='value'
                ).rename(columns={sample_id_col: 'sample_id_original'})
            
            # Extract base IDs if requested
            if extract_base_id:
                df_long['sample_id'] = df_long['sample_id_original'].apply(
                    lambda x: smart_id_extraction(x, delimiter=id_delimiter)
                )
            else:
                df_long['sample_id'] = df_long['sample_id_original']
            
            # Merge with clinical data if provided
            if clinical_df is not None:
                clinical_id_col = st.selectbox(
                    'Clinical data ID column',
                    options=clinical_df.columns.tolist()
                )
                
                # Extract base IDs from clinical if requested
                if extract_base_id:
                    clinical_df['sample_id'] = clinical_df[clinical_id_col].apply(
                        lambda x: smart_id_extraction(x, delimiter=id_delimiter)
                    )
                else:
                    clinical_df['sample_id'] = clinical_df[clinical_id_col]
                
                # Merge
                merged_df = df_long.merge(
                    clinical_df,
                    on='sample_id',
                    how='left',
                    suffixes=('', '_clinical')
                )
                
                st.success(f'✓ Merged data: {merged_df["sample_id"].nunique()} unique samples, {merged_df["feature"].nunique()} features')
            else:
                merged_df = df_long
                st.info('No clinical data merged')
            
            # Store merged data
            st.session_state['merged_df'] = merged_df
            
            with st.expander('Preview merged data'):
                st.dataframe(merged_df.head(50))
        
        # Model configuration
        st.subheader('5️⃣ Statistical Model')
        
        # Identify available clinical columns
        exclude_cols = ['sample_id', 'sample_id_original', 'feature', 'value']
        clinical_cols = [c for c in merged_df.columns if c not in exclude_cols]
        
        if not clinical_cols:
            st.error('No clinical/grouping columns available!')
            st.stop()
        
        col1, col2 = st.columns(2)
        
        with col1:
            group_col = st.selectbox(
                'Group column',
                options=clinical_cols,
                help='Categorical variable for group comparisons'
            )
            
            if group_col:
                groups_available = sorted([g for g in merged_df[group_col].dropna().unique()])
                st.write(f'Groups found: {", ".join(map(str, groups_available))}')
        
        with col2:
            covariate_cols = st.multiselect(
                'Covariates to adjust for',
                options=[c for c in clinical_cols if c != group_col],
                help='Variables to control for (age, sex, batch, etc.)'
            )
        
        # Comparison definition
        st.subheader('6️⃣ Group Comparisons')
        
        comparison_mode = st.radio(
            'Comparison mode',
            options=['All pairwise', 'Specific pairs'],
            horizontal=True
        )
        
        comparisons = []
        
        if comparison_mode == 'All pairwise':
            if len(groups_available) < 2:
                st.error('Need at least 2 groups for comparisons')
                st.stop()
            
            # Generate all pairwise comparisons
            for i in range(len(groups_available)):
                for j in range(i + 1, len(groups_available)):
                    gA, gB = groups_available[i], groups_available[j]
                    comp_name = f'{gB}_vs_{gA}'
                    comparisons.append((gA, gB, comp_name))
            
            st.info(f'Will perform {len(comparisons)} pairwise comparisons')
        
        else:
            st.write('Define specific comparisons (one per line):')
            comparison_text = st.text_area(
                'Comparisons',
                value='',
                height=150,
                help='Format: groupA, groupB, comparison_name (one per line)'
            )
            
            if comparison_text.strip():
                for line in comparison_text.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        gA, gB = parts[0], parts[1]
                        comp_name = parts[2] if len(parts) >= 3 else f'{gB}_vs_{gA}'
                        comparisons.append((gA, gB, comp_name))
        
        if not comparisons:
            st.warning('No comparisons defined')
            st.stop()
        
        # Store configuration
        st.session_state['config'] = config
        st.session_state['group_col'] = group_col
        st.session_state['covariate_cols'] = covariate_cols
        st.session_state['comparisons'] = comparisons
        
        st.success(f'✓ Configuration complete: {len(comparisons)} comparisons, {len(covariate_cols)} covariates')
    
    with tab3:
        st.header('Run Analysis & View Results')
        
        if 'merged_df' not in st.session_state or 'comparisons' not in st.session_state:
            st.warning('⚠️ Please configure analysis first')
            st.stop()
        
        if st.button('🚀 Run Analysis', type='primary'):
            merged_df = st.session_state['merged_df']
            config = st.session_state['config']
            group_col = st.session_state['group_col']
            covariate_cols = st.session_state['covariate_cols']
            comparisons = st.session_state['comparisons']
            
            # Prepare jobs
            features = sorted(merged_df['feature'].unique())
            jobs = []
            
            for comp in comparisons:
                gA, gB, comp_name = comp
                for feat in features:
                    df_feat = merged_df[merged_df['feature'] == feat].copy()
                    
                    # Check sample sizes
                    nA = (df_feat[group_col] == gA).sum()
                    nB = (df_feat[group_col] == gB).sum()
                    
                    if nA >= config.min_n_per_group and nB >= config.min_n_per_group:
                        jobs.append((feat, df_feat, gA, gB, comp_name))
            
            if len(jobs) == 0:
                st.error('No valid jobs to run. Check your group sizes.')
                st.stop()
            
            st.info(f'Running {len(jobs)} analyses ({len(features)} features × {len(comparisons)} comparisons)')
            
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                overall_progress = st.progress(0)
                status_text = st.empty()
                detail_text = st.empty()
            
            # Run analysis
            start_time = time.time()
            results = []
            
            def run_job_wrapper(job_data):
                feat, df_feat, gA, gB, comp_name = job_data
                
                # Simple progress tracking (can't update Streamlit from worker threads directly)
                result = analyze_single_feature(
                    feat, df_feat, 'value', group_col, covariate_cols,
                    gA, gB, comp_name, config, progress_callback=None
                )
                return result
            
            # Run in parallel or serial
            if config.workers == 1:
                # Serial execution with progress
                for idx, job in enumerate(jobs):
                    result = run_job_wrapper(job)
                    if result is not None:
                        results.append(result)
                    
                    # Update progress
                    progress = (idx + 1) / len(jobs)
                    overall_progress.progress(progress)
                    status_text.text(f'Progress: {idx + 1}/{len(jobs)} jobs completed ({progress*100:.1f}%)')
                    
                    if idx % 10 == 0:
                        detail_text.text(f'Processing: {job[0]} ({job[4]})')
            
            else:
                # Parallel execution
                status_text.text(f'Running in parallel with {config.workers} workers...')
                
                parallel = Parallel(n_jobs=config.workers, backend='loky', verbose=0)
                results_raw = parallel(
                    delayed(run_job_wrapper)(job) for job in jobs
                )
                
                # Filter out None results
                results = [r for r in results_raw if r is not None]
                overall_progress.progress(1.0)
            
            elapsed = time.time() - start_time
            st.success(f'✓ Analysis complete in {elapsed:.1f} seconds')
            
            if not results:
                st.error('No valid results produced')
                st.stop()
            
            # Create results dataframe
            df_results = pd.DataFrame(results)
            
            # Apply FDR correction per comparison
            try:
                from statsmodels.stats.multitest import fdrcorrection
                
                df_results['p_adj'] = np.nan
                df_results['significant'] = False
                
                for comp in df_results['comparison'].unique():
                    mask = df_results['comparison'] == comp
                    pvals = df_results.loc[mask, 'p_perm'].values
                    
                    if len(pvals) > 0:
                        rejected, qvals = fdrcorrection(pvals, alpha=config.alpha)
                        df_results.loc[mask, 'p_adj'] = qvals
                        df_results.loc[mask, 'significant'] = rejected
                
                st.info(f'FDR correction applied (α = {config.alpha})')
            
            except ImportError:
                st.warning('statsmodels not available for FDR correction')
            
            # Store results
            st.session_state['df_results'] = df_results
            
            # Display results
            st.subheader('Results Summary')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Total Features', len(df_results['feature'].unique()))
            with col2:
                st.metric('Total Tests', len(df_results))
            with col3:
                n_sig = df_results['significant'].sum() if 'significant' in df_results.columns else 0
                st.metric('Significant (FDR)', n_sig)
            
            # Results table
            st.subheader('Detailed Results')
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                show_significant_only = st.checkbox('Show significant only', value=False)
            with col2:
                sort_by = st.selectbox(
                    'Sort by',
                    options=['p_perm', 'p_adj', 'cohens_d', 'feature'],
                    index=0
                )
            
            # Filter and sort
            df_display = df_results.copy()
            if show_significant_only and 'significant' in df_display.columns:
                df_display = df_display[df_display['significant']]
            
            if sort_by in df_display.columns:
                ascending = sort_by not in ['cohens_d', 'hedges_g']
                df_display = df_display.sort_values(sort_by, ascending=ascending)
            
            # Display
            st.dataframe(
                df_display,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='📥 Download Full Results (CSV)',
                data=csv_data,
                file_name=f'freedman_lane_results_{time.strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
            
            # Quick visualization
            st.subheader('Quick Visualization')
            
            if 'p_adj' in df_results.columns and 'cohens_d' in df_results.columns:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Volcano-like plot
                df_plot = df_results.dropna(subset=['p_adj', 'cohens_d'])
                
                # Color by significance
                colors = ['red' if sig else 'gray' 
                         for sig in df_plot.get('significant', [False]*len(df_plot))]
                
                ax.scatter(
                    df_plot['cohens_d'],
                    -np.log10(df_plot['p_adj'] + 1e-300),
                    c=colors,
                    alpha=0.5,
                    s=20
                )
                
                ax.axhline(-np.log10(config.alpha), color='blue', linestyle='--', 
                          label=f'FDR = {config.alpha}')
                ax.set_xlabel("Cohen's d (Effect Size)")
                ax.set_ylabel('-log10(FDR-adjusted p-value)')
                ax.set_title('Effect Size vs. Significance')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
        
        # Show previous results if available
        elif 'df_results' in st.session_state:
            st.info('Showing previous results. Click "Run Analysis" to recompute.')
            
            df_results = st.session_state['df_results']
            
            st.subheader('Previous Results')
            st.dataframe(df_results, use_container_width=True, height=400)
            
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='📥 Download Results (CSV)',
                data=csv_data,
                file_name=f'freedman_lane_results.csv',
                mime='text/csv'
            )


if __name__ == '__main__':
    main()
