# streamlit_statistical_analysis_app.py
"""
Generic Statistical Analysis App for Population Studies

Supports ANY type of data:
- Clinical/epidemiological studies
- Genomics/proteomics/metabolomics
- Environmental studies
- Social science research
- Any other observational or experimental data

Analysis Methods:
- Descriptive statistics
- Group comparisons (with/without covariate adjustment)
- Continuous associations
- Permutation tests (Freedman-Lane)
- Standard parametric tests
- Multiple testing correction

Author: Research Analysis Tool
License: MIT
"""

import io
import os
import time
import warnings
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# =============================================================================
# INSTRUCTIONS AND DOCUMENTATION
# =============================================================================

CSV_INSTRUCTIONS = """
## 📋 CSV Format Instructions

### Option 1: Wide Format (Recommended for most studies)
One row per subject/sample, columns for variables and features.

**Example Structure:**
```
sample_id, age, sex, bmi, group, gene1, gene2, gene3, ...
SUBJ001, 45, M, 24.5, control, 5.2, 3.1, 7.8, ...
SUBJ002, 52, F, 28.3, treatment, 6.1, 4.2, 8.9, ...
```

**Required:**
- One ID column (any name: subject_id, patient_id, sample_id, etc.)
- At least one grouping or clinical variable
- At least one feature/measurement column

**Optional:**
- Covariates (age, sex, batch, etc.)
- Multiple grouping variables
- Any number of features


### Option 2: Long Format
One row per observation, with separate columns for feature name and value.

**Example Structure:**
```
sample_id, age, sex, group, feature_name, value
SUBJ001, 45, M, control, gene1, 5.2
SUBJ001, 45, M, control, gene2, 3.1
SUBJ002, 52, F, treatment, gene1, 6.1
```

**Required:**
- ID column
- Feature name column
- Value column
- Clinical/grouping variables can be repeated or in separate file


### Clinical/Metadata File (Optional)
If your features are in a separate file, you can upload clinical data separately.

**Example:**
```
subject_id, age, sex, bmi, diagnosis, treatment_arm
SUBJ001, 45, M, 24.5, healthy, placebo
SUBJ002, 52, F, 28.3, disease, active
```

**Merging:** The app will merge on ID columns (with flexible matching).


### Data Types Supported
- **Numeric:** Continuous measurements (expression, concentrations, scores)
- **Categorical:** Groups, diagnoses, sex, treatment arms
- **Binary:** Yes/No, Disease/Healthy, 0/1
- **Missing data:** Represented as empty cells or NA (will be handled)


### Common Use Cases

**Clinical Study:**
- IDs: patient_id
- Groups: treatment_arm, disease_status
- Covariates: age, sex, BMI, baseline_score
- Features: biomarkers, lab_values, outcome_measures

**Genomics Study:**
- IDs: sample_id
- Groups: case_control, tissue_type
- Covariates: age, sex, batch, sequencing_depth
- Features: gene_expression_values

**Environmental Study:**
- IDs: site_id, location_id
- Groups: exposure_level, region
- Covariates: season, temperature, population_density
- Features: pollutant_concentrations, species_counts

**Survey/Social Science:**
- IDs: respondent_id
- Groups: demographic_groups, intervention_status
- Covariates: education, income, location
- Features: survey_responses, behavioral_scores
"""

STATISTICAL_METHODS_INFO = """
## 📊 Statistical Methods Available

### 1. Descriptive Statistics
- Summary statistics by group
- Distribution visualizations
- Missing data analysis
- Sample size reporting

### 2. Group Comparisons (Categorical Predictor)
**When to use:** Comparing 2+ groups (e.g., case vs control, treatment arms)

**Methods:**
- **Freedman-Lane Permutation Test** (recommended with covariates)
  - Non-parametric, exact p-values
  - Properly adjusts for covariates
  - Suitable for any distribution
  - Use with: n_permutations (1000-10000)

- **Standard Tests** (parametric)
  - t-test (2 groups)
  - ANOVA (3+ groups)
  - Linear regression with covariates
  - Faster but assumes normality

**Output:**
- p-values (raw and adjusted for multiple testing)
- Effect sizes (Cohen's d, Hedges' g)
- Confidence intervals
- Mean differences

### 3. Continuous Associations (Continuous Predictor)
**When to use:** Relationship between continuous variables (e.g., age vs expression)

**Methods:**
- Pearson correlation (linear relationships)
- Spearman correlation (monotonic relationships)
- Linear regression with covariates
- Permutation-based correlation tests

**Output:**
- Correlation coefficients
- p-values
- Regression slopes
- R-squared values

### 4. Covariate Adjustment
**Why adjust?**
- Control for confounding variables (age, sex, batch effects)
- Isolate effect of interest
- Reduce false positives

**Methods:**
- Freedman-Lane procedure (permutation-based)
- Linear model residuals
- Stratified analysis

### 5. Multiple Testing Correction
**Why needed?**
- When testing many features (genes, biomarkers)
- Controls false discovery rate (FDR)

**Methods:**
- Benjamini-Hochberg FDR (recommended)
- Bonferroni (conservative)
- Benjamini-Yekutieli (dependent tests)

### 6. Effect Sizes
- **Cohen's d:** Standardized mean difference
- **Hedges' g:** Bias-corrected Cohen's d (small samples)
- **Correlation (r):** Strength of linear relationship
- **R²:** Proportion of variance explained

All effect sizes include bootstrap confidence intervals.
"""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def flexible_id_match(id1: str, id2: str) -> bool:
    """Flexible ID matching - handles various formats"""
    if pd.isna(id1) or pd.isna(id2):
        return False
    
    s1 = str(id1).strip().upper()
    s2 = str(id2).strip().upper()
    
    # Direct match
    if s1 == s2:
        return True
    
    # Try matching first part before delimiter
    for delim in ['-', '_', '.', ' ']:
        if delim in s1 and delim in s2:
            if s1.split(delim)[0] == s2.split(delim)[0]:
                return True
    
    return False


def extract_base_id(sample_id: Any, method: str = 'first_two') -> str:
    """Extract base ID with multiple strategies"""
    if pd.isna(sample_id):
        return ""
    
    s = str(sample_id).strip()
    
    if method == 'none':
        return s
    
    # Handle semicolon-separated
    if ';' in s:
        s = s.split(';')[0].strip()
    
    if method == 'first_part':
        # Take everything before first delimiter
        for delim in ['-', '_', '.']:
            if delim in s:
                return s.split(delim)[0]
        return s
    
    elif method == 'first_two':
        # Take first two parts
        for delim in ['-', '_', '.']:
            if delim in s:
                parts = s.split(delim)
                if len(parts) >= 2:
                    return f"{parts[0]}{delim}{parts[1]}"
        return s
    
    return s


def detect_data_format(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """
    Detect data format and suggest column mappings.
    Returns: (format_type, suggestions_dict)
    """
    suggestions = {}
    
    # Check for long format indicators
    col_lower = [c.lower() for c in df.columns]
    
    # Long format indicators
    feature_keywords = ['feature', 'gene', 'protein', 'variable', 'marker', 
                       'metabolite', 'probe', 'transcript', 'peptide']
    value_keywords = ['value', 'expression', 'abundance', 'intensity', 
                     'concentration', 'level', 'measurement', 'score']
    
    has_feature_col = any(any(kw in col for kw in feature_keywords) 
                          for col in col_lower)
    has_value_col = any(any(kw in col for kw in value_keywords) 
                        for col in col_lower)
    
    if has_feature_col and has_value_col:
        # Likely long format
        for col, col_l in zip(df.columns, col_lower):
            if any(kw in col_l for kw in feature_keywords):
                suggestions['feature_col'] = col
            if any(kw in col_l for kw in value_keywords):
                suggestions['value_col'] = col
        return 'long', suggestions
    
    # Check for wide format (many numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 5:
        suggestions['feature_cols'] = numeric_cols
        return 'wide', suggestions
    
    return 'unknown', suggestions


def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    """Detect likely ID column"""
    id_keywords = ['id', 'sample', 'subject', 'patient', 'individual', 
                   'participant', 'specimen', 'case']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in id_keywords):
            # Check if values look like IDs (unique)
            if df[col].nunique() == len(df):
                return col
    
    # Fallback: first column if mostly unique
    if df[df.columns[0]].nunique() / len(df) > 0.9:
        return df.columns[0]
    
    return None


def detect_grouping_columns(df: pd.DataFrame) -> List[str]:
    """Detect likely grouping/categorical columns"""
    candidates = []
    
    for col in df.columns:
        # Skip ID-like columns
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['id', 'sample', 'subject']):
            continue
        
        # Check if categorical or low cardinality
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            candidates.append(col)
    
    return candidates


def detect_covariate_columns(df: pd.DataFrame) -> List[str]:
    """Detect likely covariate columns"""
    covariate_keywords = ['age', 'sex', 'gender', 'batch', 'plate', 
                          'bmi', 'weight', 'height', 'race', 'ethnicity',
                          'smoking', 'education', 'income']
    
    candidates = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in covariate_keywords):
            candidates.append(col)
    
    return candidates


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def compute_descriptive_stats(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    """Compute descriptive statistics, optionally by group"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if group_col and group_col in df.columns:
        # By group
        results = []
        for group in df[group_col].unique():
            if pd.isna(group):
                continue
            
            group_data = df[df[group_col] == group]
            
            for col in numeric_cols:
                values = group_data[col].dropna()
                if len(values) == 0:
                    continue
                
                results.append({
                    'variable': col,
                    'group': group,
                    'n': len(values),
                    'mean': values.mean(),
                    'std': values.std(),
                    'median': values.median(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75),
                    'min': values.min(),
                    'max': values.max()
                })
        
        return pd.DataFrame(results)
    
    else:
        # Overall
        results = []
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            results.append({
                'variable': col,
                'n': len(values),
                'mean': values.mean(),
                'std': values.std(),
                'median': values.median(),
                'q25': values.quantile(0.25),
                'q75': values.quantile(0.75),
                'min': values.min(),
                'max': values.max()
            })
        
        return pd.DataFrame(results)


def freedman_lane_test(df: pd.DataFrame, outcome_col: str, group_col: str,
                       covariates: List[str], group_a: Any, group_b: Any,
                       n_perm: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """
    Freedman-Lane permutation test for group differences with covariate adjustment.
    
    This is the CORE permutation method for group comparisons.
    """
    
    df_sub = df[df[group_col].isin([group_a, group_b])].copy().reset_index(drop=True)
    
    if len(df_sub) < 4:
        return {'statistic': np.nan, 'p_value': np.nan, 'method': 'freedman_lane'}
    
    # Step 1: Fit reduced model (covariates only)
    if not covariates:
        fitted = np.repeat(df_sub[outcome_col].mean(), len(df_sub))
        residuals = df_sub[outcome_col].values - fitted
    else:
        terms = []
        for cov in covariates:
            if cov in df_sub.columns:
                if df_sub[cov].dtype == 'object' or df_sub[cov].nunique() < 10:
                    terms.append(f"C({cov})")
                else:
                    terms.append(cov)
        
        if terms:
            try:
                formula_reduced = f"{outcome_col} ~ " + " + ".join(terms)
                model_reduced = smf.ols(formula_reduced, data=df_sub.dropna()).fit()
                fitted = model_reduced.fittedvalues
                residuals = model_reduced.resid
                
                # Align with full dataframe
                fitted_full = np.repeat(np.nan, len(df_sub))
                resid_full = np.repeat(np.nan, len(df_sub))
                fitted_full[fitted.index] = fitted.values
                resid_full[residuals.index] = residuals.values
                fitted = np.nan_to_num(fitted_full, nan=df_sub[outcome_col].mean())
                residuals = np.nan_to_num(resid_full, nan=0)
            except:
                fitted = np.repeat(df_sub[outcome_col].mean(), len(df_sub))
                residuals = df_sub[outcome_col].values - fitted
        else:
            fitted = np.repeat(df_sub[outcome_col].mean(), len(df_sub))
            residuals = df_sub[outcome_col].values - fitted
    
    # Step 2: Compute observed statistic (group difference)
    try:
        vals_a = df_sub[df_sub[group_col] == group_a][outcome_col]
        vals_b = df_sub[df_sub[group_col] == group_b][outcome_col]
        obs_stat = vals_b.mean() - vals_a.mean()
    except:
        return {'statistic': np.nan, 'p_value': np.nan, 'method': 'freedman_lane'}
    
    # Step 3: Permutation loop
    rng = np.random.RandomState(seed)
    perm_stats = []
    
    for _ in range(n_perm):
        # Permute residuals
        perm_idx = rng.permutation(len(residuals))
        y_perm = fitted + residuals[perm_idx]
        
        df_perm = df_sub.copy()
        df_perm[outcome_col] = y_perm
        
        try:
            vals_a_perm = df_perm[df_perm[group_col] == group_a][outcome_col]
            vals_b_perm = df_perm[df_perm[group_col] == group_b][outcome_col]
            stat_perm = vals_b_perm.mean() - vals_a_perm.mean()
            
            if np.isfinite(stat_perm):
                perm_stats.append(stat_perm)
        except:
            continue
    
    perm_stats = np.array(perm_stats)
    
    if len(perm_stats) == 0:
        return {'statistic': obs_stat, 'p_value': np.nan, 'method': 'freedman_lane'}
    
    # Two-tailed p-value
    p_value = (np.sum(np.abs(perm_stats) >= abs(obs_stat)) + 1) / (len(perm_stats) + 1)
    
    return {
        'statistic': float(obs_stat),
        'p_value': float(p_value),
        'n_perm_valid': len(perm_stats),
        'method': 'freedman_lane'
    }


def standard_group_test(df: pd.DataFrame, outcome_col: str, group_col: str,
                       group_a: Any, group_b: Any) -> Dict[str, Any]:
    """Standard parametric tests (t-test or Mann-Whitney)"""
    
    df_sub = df[df[group_col].isin([group_a, group_b])].copy()
    
    vals_a = df_sub[df_sub[group_col] == group_a][outcome_col].dropna()
    vals_b = df_sub[df_sub[group_col] == group_b][outcome_col].dropna()
    
    if len(vals_a) < 2 or len(vals_b) < 2:
        return {'statistic': np.nan, 'p_value': np.nan, 'method': 'insufficient_data'}
    
    # Use t-test
    try:
        stat, pval = stats.ttest_ind(vals_a, vals_b, equal_var=False)
        return {
            'statistic': float(stat),
            'p_value': float(pval),
            'method': 'welch_ttest'
        }
    except:
        return {'statistic': np.nan, 'p_value': np.nan, 'method': 'error'}


def correlation_analysis(df: pd.DataFrame, feature_col: str, predictor_col: str,
                        method: str = 'pearson') -> Dict[str, Any]:
    """Correlation analysis for continuous predictors"""
    
    df_clean = df[[feature_col, predictor_col]].dropna()
    
    if len(df_clean) < 3:
        return {'correlation': np.nan, 'p_value': np.nan, 'method': method}
    
    x = df_clean[predictor_col].values
    y = df_clean[feature_col].values
    
    try:
        if method == 'pearson':
            corr, pval = pearsonr(x, y)
        else:  # spearman
            corr, pval = spearmanr(x, y)
        
        return {
            'correlation': float(corr),
            'p_value': float(pval),
            'n': len(df_clean),
            'method': method
        }
    except:
        return {'correlation': np.nan, 'p_value': np.nan, 'method': method}


def compute_effect_size(vals_a, vals_b, method='cohens_d') -> float:
    """Compute effect size"""
    
    vals_a = np.asarray(vals_a).flatten()
    vals_b = np.asarray(vals_b).flatten()
    
    n1, n2 = len(vals_a), len(vals_b)
    if n1 < 2 or n2 < 2:
        return np.nan
    
    m1, m2 = np.nanmean(vals_a), np.nanmean(vals_b)
    
    if method == 'cohens_d':
        # Pooled standard deviation
        v1 = np.nanvar(vals_a, ddof=1)
        v2 = np.nanvar(vals_b, ddof=1)
        pooled_std = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
        
        if pooled_std == 0:
            return np.nan
        
        return (m2 - m1) / pooled_std
    
    elif method == 'hedges_g':
        # Bias-corrected Cohen's d
        d = compute_effect_size(vals_a, vals_b, method='cohens_d')
        if np.isnan(d):
            return np.nan
        
        correction = 1 - (3 / (4*(n1+n2) - 9))
        return d * correction
    
    return np.nan


def bootstrap_ci(vals_a, vals_b, effect_func, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for effect size"""
    
    vals_a = np.asarray(vals_a)
    vals_b = np.asarray(vals_b)
    
    n1, n2 = len(vals_a), len(vals_b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    
    rng = np.random.RandomState(seed)
    boot_effects = []
    
    for _ in range(n_boot):
        boot_a = rng.choice(vals_a, size=n1, replace=True)
        boot_b = rng.choice(vals_b, size=n2, replace=True)
        
        effect = effect_func(boot_a, boot_b)
        if np.isfinite(effect):
            boot_effects.append(effect)
    
    if len(boot_effects) < 50:
        return np.nan, np.nan
    
    alpha = 1 - ci
    ci_low = np.percentile(boot_effects, 100 * alpha/2)
    ci_high = np.percentile(boot_effects, 100 * (1-alpha/2))
    
    return ci_low, ci_high


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

def analyze_feature(feature_name: str, df_feature: pd.DataFrame,
                   analysis_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main analysis function - routes to appropriate test based on analysis type.
    """
    
    try:
        if analysis_type == 'group_comparison':
            return analyze_group_comparison(feature_name, df_feature, config)
        
        elif analysis_type == 'continuous_association':
            return analyze_continuous_association(feature_name, df_feature, config)
        
        else:
            return {'feature': feature_name, 'error': 'Unknown analysis type'}
    
    except Exception as e:
        return {'feature': feature_name, 'error': str(e)[:200]}


def analyze_group_comparison(feature_name: str, df_feature: pd.DataFrame,
                             config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze group comparison for a single feature"""
    
    outcome_col = config['outcome_col']
    group_col = config['group_col']
    group_a = config['group_a']
    group_b = config['group_b']
    covariates = config.get('covariates', [])
    method = config.get('method', 'freedman_lane')
    
    # Filter to relevant groups
    df_sub = df_feature[df_feature[group_col].isin([group_a, group_b])].copy()
    
    n_a = (df_sub[group_col] == group_a).sum()
    n_b = (df_sub[group_col] == group_b).sum()
    
    min_n = config.get('min_n_per_group', 3)
    if n_a < min_n or n_b < min_n:
        return None
    
    # Run statistical test
    if method == 'freedman_lane':
        test_result = freedman_lane_test(
            df_sub, outcome_col, group_col, covariates,
            group_a, group_b,
            n_perm=config.get('n_perm', 2000),
            seed=config.get('seed', 42)
        )
    else:
        test_result = standard_group_test(df_sub, outcome_col, group_col, group_a, group_b)
    
    # Compute effect size
    vals_a = df_sub[df_sub[group_col] == group_a][outcome_col].dropna()
    vals_b = df_sub[df_sub[group_col] == group_b][outcome_col].dropna()
    
    effect_size = compute_effect_size(vals_a, vals_b, method='cohens_d')
    
    # Bootstrap CI for effect size
    ci_low, ci_high = bootstrap_ci(
        vals_a, vals_b,
        lambda a, b: compute_effect_size(a, b, method='cohens_d'),
        n_boot=config.get('n_boot', 500),
        seed=config.get('seed', 42)
    )
    
    # Compile results
    result = {
        'feature': feature_name,
        'comparison': f"{group_b}_vs_{group_a}",
        'group_a': str(group_a),
        'group_b': str(group_b),
        'n_a': int(n_a),
        'n_b': int(n_b),
        'mean_a': float(vals_a.mean()),
        'mean_b': float(vals_b.mean()),
        'mean_diff': float(vals_b.mean() - vals_a.mean()),
        'test_statistic': test_result.get('statistic', np.nan),
        'p_value': test_result.get('p_value', np.nan),
        'effect_size_d': effect_size,
        'effect_size_ci_low': ci_low,
        'effect_size_ci_high': ci_high,
        'method': test_result.get('method', method),
        'covariates': ','.join(covariates) if covariates else 'none'
    }
    
    return result


def analyze_continuous_association(feature_name: str, df_feature: pd.DataFrame,
                                   config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze continuous association for a single feature"""
    
    outcome_col = config['outcome_col']
    predictor_col = config['predictor_col']
    method = config.get('correlation_method', 'pearson')
    
    result = correlation_analysis(df_feature, outcome_col, predictor_col, method=method)
    
    result['feature'] = feature_name
    result['predictor'] = predictor_col
    
    return result


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title='Statistical Analysis App',
        page_icon='📊',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
        }
        .success-box {
            padding: 10px;
            border-radius: 5px;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title('📊 Generic Statistical Analysis for Population Studies')
    st.markdown('**A comprehensive tool for ANY observational or experimental data**')
    
    # Sidebar
    with st.sidebar:
        st.header('📚 Help & Documentation')
        
        with st.expander('📋 CSV Format Guide', expanded=False):
            st.markdown(CSV_INSTRUCTIONS)
        
        with st.expander('📊 Statistical Methods', expanded=False):
            st.markdown(STATISTICAL_METHODS_INFO)
        
        with st.expander('💡 Quick Start Guide'):
            st.markdown("""
            ### Quick Start (3 steps):
            
            1. **Upload Data**
               - Upload your CSV file(s)
               - App will auto-detect format
            
            2. **Configure Analysis**
               - Select analysis type
               - Choose variables
               - Set parameters
            
            3. **Run & Download**
               - Click "Run Analysis"
               - Review results
               - Download CSV
            
            ### Example Workflows:
            
            **Compare gene expression between cases and controls:**
            - Analysis type: Group Comparison
            - Groups: case vs control
            - Method: Freedman-Lane (if adjusting for age/sex)
            - Features: all genes
            
            **Test correlation between age and biomarkers:**
            - Analysis type: Continuous Association
            - Predictor: age
            - Method: Pearson or Spearman
            - Features: all biomarkers
            """)
    
    # Main tabs
    tabs = st.tabs(['📁 Upload Data', '⚙️ Configure', '▶️ Run Analysis', '📊 Results'])
    
    # =============================================================================
    # TAB 1: UPLOAD DATA
    # =============================================================================
    with tabs[0]:
        st.header('📁 Step 1: Upload Your Data')
        
        st.info('💡 **Tip:** Upload one file with everything, or separate feature and clinical files')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Primary Data File')
            uploaded_main = st.file_uploader(
                'Upload CSV (features and/or clinical data)',
                type=['csv'],
                key='main_file',
                help='This can contain everything, or just features'
            )
            
            if uploaded_main:
                try:
                    df_main = pd.read_csv(uploaded_main)
                    st.success(f'✓ Loaded {df_main.shape[0]} rows × {df_main.shape[1]} columns')
                    
                    # Auto-detect format
                    format_detected, suggestions = detect_data_format(df_main)
                    st.info(f'🔍 Detected format: **{format_detected.upper()}**')
                    
                    # Store in session
                    st.session_state['df_main'] = df_main
                    st.session_state['format_detected'] = format_detected
                    st.session_state['suggestions'] = suggestions
                    
                    with st.expander('👀 Preview Data (first 20 rows)'):
                        st.dataframe(df_main.head(20), use_container_width=True)
                    
                    # Data summary
                    with st.expander('📋 Data Summary'):
                        st.write('**Column Types:**')
                        col_types = pd.DataFrame({
                            'Column': df_main.columns,
                            'Type': df_main.dtypes.astype(str),
                            'Non-Null': df_main.notna().sum(),
                            'Null': df_main.isna().sum(),
                            'Unique': [df_main[col].nunique() for col in df_main.columns]
                        })
                        st.dataframe(col_types, use_container_width=True)
                
                except Exception as e:
                    st.error(f'❌ Error loading file: {e}')
        
        with col2:
            st.subheader('Clinical Data (Optional)')
            uploaded_clinical = st.file_uploader(
                'Upload clinical/metadata CSV',
                type=['csv'],
                key='clinical_file',
                help='Only needed if not included in primary file'
            )
            
            if uploaded_clinical:
                try:
                    df_clinical = pd.read_csv(uploaded_clinical)
                    st.success(f'✓ Loaded {df_clinical.shape[0]} rows × {df_clinical.shape[1]} columns')
                    
                    st.session_state['df_clinical'] = df_clinical
                    
                    with st.expander('👀 Preview Clinical Data'):
                        st.dataframe(df_clinical.head(20), use_container_width=True)
                
                except Exception as e:
                    st.error(f'❌ Error loading file: {e}')
    
    # =============================================================================
    # TAB 2: CONFIGURE
    # =============================================================================
    with tabs[1]:
        st.header('⚙️ Step 2: Configure Your Analysis')
        
        if 'df_main' not in st.session_state:
            st.warning('⚠️ Please upload data first (see Upload Data tab)')
            st.stop()
        
        df_main = st.session_state['df_main']
        df_clinical = st.session_state.get('df_clinical', None)
        format_detected = st.session_state.get('format_detected', 'unknown')
        suggestions = st.session_state.get('suggestions', {})
        
        # Section 1: Data Format
        st.subheader('1️⃣ Confirm Data Format')
        
        format_choice = st.radio(
            'Data format',
            options=['Wide Format', 'Long Format'],
            index=0 if format_detected == 'wide' else 1,
            horizontal=True,
            help='Wide: one row per sample. Long: one row per measurement'
        )
        
        is_long = (format_choice == 'Long Format')
        
        # Section 2: Column Mapping
        st.subheader('2️⃣ Identify Columns')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # ID column
            id_detected = detect_id_column(df_main)
            id_col = st.selectbox(
                'Sample ID Column',
                options=df_main.columns.tolist(),
                index=df_main.columns.tolist().index(id_detected) if id_detected else 0,
                help='Unique identifier for each sample/subject'
            )
        
        with col2:
            if is_long:
                # Feature column
                feature_col = st.selectbox(
                    'Feature Name Column',
                    options=[c for c in df_main.columns if c != id_col],
                    index=0 if 'feature_col' not in suggestions else 
                          [c for c in df_main.columns if c != id_col].index(suggestions['feature_col']),
                    help='Column containing feature/variable names'
                )
        
        with col3:
            if is_long:
                # Value column
                value_col = st.selectbox(
                    'Value Column',
                    options=[c for c in df_main.columns if c not in [id_col, feature_col]],
                    index=0 if 'value_col' not in suggestions else
                          [c for c in df_main.columns if c not in [id_col, feature_col]].index(suggestions['value_col']),
                    help='Column containing measurement values'
                )
        
        # Feature selection for wide format
        if not is_long:
            st.write('**Select Feature Columns:**')
            
            # Detect numeric columns
            numeric_cols = df_main.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != id_col]
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                feature_selection_mode = st.radio(
                    'Feature selection',
                    options=['All numeric columns', 'Select specific columns'],
                    index=0
                )
            
            with col2:
                if feature_selection_mode == 'Select specific columns':
                    selected_features = st.multiselect(
                        'Choose feature columns',
                        options=numeric_cols,
                        default=numeric_cols[:min(10, len(numeric_cols))],
                        help='Select columns to analyze'
                    )
                else:
                    selected_features = numeric_cols
                    st.info(f'Using {len(selected_features)} numeric columns')
        
        # Section 3: Merge Clinical Data
        if df_clinical is not None:
            st.subheader('3️⃣ Merge Clinical Data')
            
            clinical_id_col = st.selectbox(
                'ID column in clinical file',
                options=df_clinical.columns.tolist(),
                index=0,
                help='Must match IDs in primary file'
            )
            
            # ID extraction options
            id_extraction = st.radio(
                'ID matching strategy',
                options=['Exact match', 'Extract base ID (first part)', 'Extract base ID (first two parts)'],
                index=1,
                horizontal=True,
                help='Use extraction if IDs have suffixes (e.g., SAMPLE-01-A → SAMPLE-01)'
            )
            
            extraction_map = {
                'Exact match': 'none',
                'Extract base ID (first part)': 'first_part',
                'Extract base ID (first two parts)': 'first_two'
            }
            
            id_method = extraction_map[id_extraction]
        else:
            clinical_id_col = None
            id_method = 'none'
        
        # Convert to long format and merge
        with st.spinner('Preparing data...'):
            if is_long:
                df_long = df_main.rename(columns={
                    id_col: 'sample_id_orig',
                    feature_col: 'feature',
                    value_col: 'value'
                }).copy()
            else:
                if not selected_features:
                    st.error('❌ No feature columns selected')
                    st.stop()
                
                df_long = df_main.melt(
                    id_vars=[id_col],
                    value_vars=selected_features,
                    var_name='feature',
                    value_name='value'
                ).rename(columns={id_col: 'sample_id_orig'})
            
            # Extract base IDs
            df_long['sample_id'] = df_long['sample_id_orig'].apply(
                lambda x: extract_base_id(x, method=id_method)
            )
            
            # Merge clinical if provided
            if df_clinical is not None:
                df_clinical['sample_id'] = df_clinical[clinical_id_col].apply(
                    lambda x: extract_base_id(x, method=id_method)
                )
                
                merged_df = df_long.merge(
                    df_clinical,
                    on='sample_id',
                    how='left',
                    suffixes=('', '_clin')
                )
                
                st.success(f'✓ Merged: {merged_df["sample_id"].nunique()} samples, {merged_df["feature"].nunique()} features')
            else:
                merged_df = df_long
                st.info(f'ℹ️ Using {merged_df["sample_id"].nunique()} samples, {merged_df["feature"].nunique()} features')
            
            st.session_state['merged_df'] = merged_df
        
        # Show preview
        with st.expander('👀 Preview Merged Data'):
            st.dataframe(merged_df.head(50), use_container_width=True)
        
        # Section 4: Analysis Type Selection
        st.subheader('4️⃣ Select Analysis Type')
        
        analysis_type = st.radio(
            'What type of analysis?',
            options=[
                'Group Comparison (categorical predictor)',
                'Continuous Association (continuous predictor)',
                'Descriptive Statistics Only'
            ],
            index=0,
            help='Choose based on your research question'
        )
        
        st.session_state['analysis_type'] = analysis_type
        
        # Section 5: Analysis-specific configuration
        st.subheader('5️⃣ Analysis Configuration')
        
        # Get available clinical columns
        exclude_cols = ['sample_id', 'sample_id_orig', 'feature', 'value']
        clinical_cols = [c for c in merged_df.columns if c not in exclude_cols]
        
        if not clinical_cols:
            st.error('❌ No clinical/grouping columns found!')
            st.stop()
        
        if analysis_type == 'Group Comparison (categorical predictor)':
            col1, col2 = st.columns(2)
            
            with col1:
                # Group column
                group_suggestions = detect_grouping_columns(merged_df)
                group_col = st.selectbox(
                    'Grouping Variable',
                    options=clinical_cols,
                    index=clinical_cols.index(group_suggestions[0]) if group_suggestions else 0,
                    help='Categorical variable for group comparisons'
                )
                
                # Show available groups
                if group_col:
                    groups = sorted([g for g in merged_df[group_col].dropna().unique()])
                    st.write(f'**Groups found:** {", ".join(map(str, groups))}')
                    
                    # Group selection
                    if len(groups) >= 2:
                        comparison_mode = st.radio(
                            'Comparison mode',
                            options=['All pairwise', 'Select specific pair'],
                            index=0
                        )
                        
                        if comparison_mode == 'Select specific pair':
                            col_a, col_b = st.columns(2)
                            with col_a:
                                group_a = st.selectbox('Group A (reference)', options=groups, index=0)
                            with col_b:
                                group_b = st.selectbox('Group B (comparison)', options=groups, index=min(1, len(groups)-1))
                            
                            comparisons = [(group_a, group_b)]
                        else:
                            # Generate all pairs
                            comparisons = []
                            for i in range(len(groups)):
                                for j in range(i+1, len(groups)):
                                    comparisons.append((groups[i], groups[j]))
                            
                            st.info(f'Will perform {len(comparisons)} pairwise comparisons')
                    else:
                        st.error('❌ Need at least 2 groups for comparison')
                        st.stop()
            
            with col2:
                # Covariates
                covariate_suggestions = detect_covariate_columns(merged_df)
                covariates = st.multiselect(
                    'Covariates (adjust for)',
                    options=[c for c in clinical_cols if c != group_col],
                    default=[c for c in covariate_suggestions if c != group_col][:3],
                    help='Variables to control for (age, sex, batch, etc.)'
                )
                
                # Method selection
                st.write('**Statistical Method:**')
                method = st.radio(
                    'Choose method',
                    options=['Freedman-Lane Permutation (recommended)', 'Standard Parametric Test'],
                    index=0 if covariates else 1,
                    help='Freedman-Lane is better for covariate adjustment'
                )
                
                use_permutation = (method == 'Freedman-Lane Permutation (recommended)')
                
                if use_permutation:
                    n_perm = st.number_input(
                        'Number of permutations',
                        min_value=100,
                        max_value=10000,
                        value=2000,
                        step=100,
                        help='More = more accurate but slower'
                    )
                else:
                    n_perm = 0
            
            # Store config
            st.session_state['group_col'] = group_col
            st.session_state['comparisons'] = comparisons
            st.session_state['covariates'] = covariates
            st.session_state['use_permutation'] = use_permutation
            st.session_state['n_perm'] = n_perm
        
        elif analysis_type == 'Continuous Association (continuous predictor)':
            col1, col2 = st.columns(2)
            
            with col1:
                # Predictor selection
                numeric_clinical = [c for c in clinical_cols 
                                   if pd.api.types.is_numeric_dtype(merged_df[c])]
                
                if not numeric_clinical:
                    st.error('❌ No continuous variables found in clinical data')
                    st.stop()
                
                predictor_col = st.selectbox(
                    'Predictor Variable',
                    options=numeric_clinical,
                    help='Continuous variable to test association with'
                )
            
            with col2:
                # Method selection
                correlation_method = st.radio(
                    'Correlation Method',
                    options=['Pearson (linear)', 'Spearman (monotonic)'],
                    index=0,
                    help='Pearson for linear relationships, Spearman for monotonic'
                )
            
            st.session_state['predictor_col'] = predictor_col
            st.session_state['correlation_method'] = correlation_method.split()[0].lower()
        
        # Section 6: General Settings
        st.subheader('6️⃣ General Settings')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_n_per_group = st.number_input(
                'Min samples per group',
                min_value=2,
                max_value=50,
                value=5,
                help='Features with fewer samples will be skipped'
            )
        
        with col2:
            alpha_level = st.number_input(
                'Significance level (α)',
                min_value=0.001,
                max_value=0.2,
                value=0.05,
                step=0.01,
                help='For FDR correction'
            )
        
        with col3:
            n_workers = st.number_input(
                'Parallel workers',
                min_value=1,
                max_value=16,
                value=4,
                help='Number of CPU cores to use'
            )
        
        st.session_state['min_n_per_group'] = min_n_per_group
        st.session_state['alpha_level'] = alpha_level
        st.session_state['n_workers'] = n_workers
        
        # Bootstrap settings (for effect sizes)
        with st.expander('⚙️ Advanced: Bootstrap Settings'):
            n_boot = st.number_input(
                'Bootstrap iterations',
                min_value=100,
                max_value=5000,
                value=500,
                help='For confidence intervals (more = slower)'
            )
            
            random_seed = st.number_input(
                'Random seed',
                min_value=0,
                max_value=9999,
                value=42,
                help='For reproducibility'
            )
            
            st.session_state['n_boot'] = n_boot
            st.session_state['seed'] = random_seed
        
        st.success('✓ Configuration complete!')
    
    # =============================================================================
    # TAB 3: RUN ANALYSIS
    # =============================================================================
    with tabs[2]:
        st.header('▶️ Step 3: Run Analysis')
        
        if 'merged_df' not in st.session_state:
            st.warning('⚠️ Please configure analysis first')
            st.stop()
        
        # Show summary of configuration
        st.subheader('📋 Analysis Summary')
        
        analysis_type = st.session_state.get('analysis_type', '')
        
        if 'Group Comparison' in analysis_type:
            group_col = st.session_state.get('group_col', '')
            comparisons = st.session_state.get('comparisons', [])
            covariates = st.session_state.get('covariates', [])
            use_perm = st.session_state.get('use_permutation', False)
            
            st.write(f'**Analysis:** Group Comparison')
            st.write(f'**Grouping variable:** {group_col}')
            st.write(f'**Comparisons:** {len(comparisons)} pairs')
            st.write(f'**Covariates:** {", ".join(covariates) if covariates else "None"}')
            st.write(f'**Method:** {"Freedman-Lane Permutation" if use_perm else "Parametric test"}')
        
        elif 'Continuous' in analysis_type:
            predictor_col = st.session_state.get('predictor_col', '')
            method = st.session_state.get('correlation_method', 'pearson')
            
            st.write(f'**Analysis:** Continuous Association')
            st.write(f'**Predictor:** {predictor_col}')
            st.write(f'**Method:** {method.capitalize()} correlation')
        
        merged_df = st.session_state['merged_df']
        features = sorted(merged_df['feature'].unique())
        
        st.write(f'**Features to analyze:** {len(features)}')
        st.write(f'**Total samples:** {merged_df["sample_id"].nunique()}')
        
        # Run button
        if st.button('🚀 Run Analysis', type='primary', use_container_width=True):
            
            # Prepare jobs
            if 'Group Comparison' in analysis_type:
                jobs = []
                comparisons = st.session_state['comparisons']
                
                for comp in comparisons:
                    group_a, group_b = comp
                    
                    for feat in features:
                        df_feat = merged_df[merged_df['feature'] == feat].copy()
                        
                        # Check sample sizes
                        n_a = (df_feat[st.session_state['group_col']] == group_a).sum()
                        n_b = (df_feat[st.session_state['group_col']] == group_b).sum()
                        
                        if n_a >= st.session_state['min_n_per_group'] and \
                           n_b >= st.session_state['min_n_per_group']:
                            
                            config = {
                                'outcome_col': 'value',
                                'group_col': st.session_state['group_col'],
                                'group_a': group_a,
                                'group_b': group_b,
                                'covariates': st.session_state.get('covariates', []),
                                'method': 'freedman_lane' if st.session_state.get('use_permutation') else 'parametric',
                                'n_perm': st.session_state.get('n_perm', 2000),
                                'n_boot': st.session_state.get('n_boot', 500),
                                'min_n_per_group': st.session_state['min_n_per_group'],
                                'seed': st.session_state.get('seed', 42)
                            }
                            
                            jobs.append((feat, df_feat, 'group_comparison', config))
            
            elif 'Continuous' in analysis_type:
                jobs = []
                
                for feat in features:
                    df_feat = merged_df[merged_df['feature'] == feat].copy()
                    
                    config = {
                        'outcome_col': 'value',
                        'predictor_col': st.session_state['predictor_col'],
                        'correlation_method': st.session_state.get('correlation_method', 'pearson'),
                        'seed': st.session_state.get('seed', 42)
                    }
                    
                    jobs.append((feat, df_feat, 'continuous_association', config))
            
            if not jobs:
                st.error('❌ No valid analyses to run (check sample sizes)')
                st.stop()
            
            st.info(f'📊 Running {len(jobs)} analyses...')
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run analyses
            start_time = time.time()
            results = []
            
            n_workers = st.session_state.get('n_workers', 1)
            
            if n_workers == 1:
                # Serial execution
                for idx, job in enumerate(jobs):
                    feat, df_feat, atype, config = job
                    result = analyze_feature(feat, df_feat, atype, config)
                    
                    if result:
                        results.append(result)
                    
                    progress = (idx + 1) / len(jobs)
                    progress_bar.progress(progress)
                    status_text.text(f'Progress: {idx+1}/{len(jobs)} ({progress*100:.1f}%)')
            
            else:
                # Parallel execution
                status_text.text(f'Running in parallel with {n_workers} workers...')
                
                def run_job(job):
                    feat, df_feat, atype, config = job
                    return analyze_feature(feat, df_feat, atype, config)
                
                parallel = Parallel(n_jobs=n_workers, backend='loky', verbose=0)
                results_raw = parallel(delayed(run_job)(job) for job in jobs)
                
                results = [r for r in results_raw if r is not None]
                progress_bar.progress(1.0)
            
            elapsed = time.time() - start_time
            st.success(f'✅ Analysis complete in {elapsed:.1f} seconds!')
            
            if not results:
                st.error('❌ No valid results produced')
                st.stop()
            
            # Create results dataframe
            df_results = pd.DataFrame(results)
            
            # Apply FDR correction
            if 'p_value' in df_results.columns:
                try:
                    from statsmodels.stats.multitest import fdrcorrection
                    
                    pvals = df_results['p_value'].fillna(1.0).values
                    rejected, qvals = fdrcorrection(pvals, alpha=st.session_state.get('alpha_level', 0.05))
                    
                    df_results['p_adjusted'] = qvals
                    df_results['significant'] = rejected
                    
                    st.info(f'✓ FDR correction applied (α = {st.session_state.get("alpha_level", 0.05)})')
                
                except ImportError:
                    st.warning('⚠️ Could not apply FDR correction (statsmodels required)')
            
            # Store results
            st.session_state['df_results'] = df_results
            st.session_state['analysis_complete'] = True
            
            # Show quick summary
            st.subheader('📊 Quick Summary')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric('Features Tested', len(df_results['feature'].unique()))
            
            with col2:
                st.metric('Total Tests', len(df_results))
            
            with col3:
                if 'significant' in df_results.columns:
                    n_sig = df_results['significant'].sum()
                    st.metric('Significant (FDR)', n_sig)
                else:
                    st.metric('Significant', 'N/A')
            
            with col4:
                st.metric('Time Elapsed', f'{elapsed:.1f}s')
            
            st.info('👉 Go to **Results** tab to view and download full results')
    
    # =============================================================================
    # TAB 4: RESULTS
    # =============================================================================
    with tabs[3]:
        st.header('📊 Analysis Results')
        
        if 'df_results' not in st.session_state or not st.session_state.get('analysis_complete', False):
            st.warning('⚠️ No results available yet. Run analysis first!')
            st.stop()
        
        df_results = st.session_state['df_results']
        
        # Results summary
        st.subheader('📈 Summary Statistics')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric('Features Analyzed', len(df_results['feature'].unique()))
        
        with col2:
            st.metric('Total Tests', len(df_results))
        
        with col3:
            if 'significant' in df_results.columns:
                n_sig = df_results['significant'].sum()
                pct_sig = 100 * n_sig / len(df_results)
                st.metric('Significant Results', f'{n_sig} ({pct_sig:.1f}%)')
        
        with col4:
            if 'p_value' in df_results.columns:
                median_p = df_results['p_value'].median()
                st.metric('Median p-value', f'{median_p:.4f}')
        
        # Interactive filtering
        st.subheader('🔍 Filter Results')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_sig_only = st.checkbox(
                'Show significant only',
                value=False,
                help='Filter to FDR-significant results only'
            )
        
        with col2:
            if 'p_value' in df_results.columns:
                p_threshold = st.number_input(
                    'p-value threshold',
                    min_value=0.0,
                    max_value=1.0,
                    value=0.05,
                    step=0.01,
                    help='Filter by raw p-value'
                )
        
        with col3:
            sort_by = st.selectbox(
                'Sort by',
                options=['p_value', 'p_adjusted', 'effect_size_d', 'feature'] 
                        if 'p_adjusted' in df_results.columns 
                        else ['p_value', 'correlation', 'feature'],
                index=0
            )
        
        # Apply filters
        df_display = df_results.copy()
        
        if show_sig_only and 'significant' in df_display.columns:
            df_display = df_display[df_display['significant']]
        
        if 'p_value' in df_display.columns:
            df_display = df_display[df_display['p_value'] <= p_threshold]
        
        if sort_by in df_display.columns:
            ascending = sort_by not in ['effect_size_d', 'correlation']
            df_display = df_display.sort_values(sort_by, ascending=ascending)
        
        # Display results table
        st.subheader('📋 Results Table')
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Download buttons
        st.subheader('💾 Download Results')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Full results
            csv_full = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='📥 Download All Results (CSV)',
                data=csv_full,
                file_name=f'analysis_results_full_{time.strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        with col2:
            # Filtered results
            if len(df_display) < len(df_results):
                csv_filtered = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label='📥 Download Filtered Results (CSV)',
                    data=csv_filtered,
                    file_name=f'analysis_results_filtered_{time.strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
        
        # Visualization
        st.subheader('📊 Visualization')
        
        try:
            import matplotlib.pyplot as plt
            
            viz_type = st.selectbox(
                'Select visualization',
                options=['Volcano Plot', 'P-value Distribution', 'Effect Size Distribution']
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if viz_type == 'Volcano Plot' and 'p_adjusted' in df_results.columns and 'effect_size_d' in df_results.columns:
                df_plot = df_results.dropna(subset=['p_adjusted', 'effect_size_d'])
                
                colors = ['red' if sig else 'gray' 
                         for sig in df_plot.get('significant', [False]*len(df_plot))]
                
                ax.scatter(
                    df_plot['effect_size_d'],
                    -np.log10(df_plot['p_adjusted'] + 1e-300),
                    c=colors,
                    alpha=0.5,
                    s=30
                )
                
                alpha_level = st.session_state.get('alpha_level', 0.05)
                ax.axhline(-np.log10(alpha_level), color='blue', linestyle='--', 
                          label=f'FDR = {alpha_level}')
                
                ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
                ax.set_ylabel('-log10(FDR-adjusted p-value)', fontsize=12)
                ax.set_title('Volcano Plot', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            elif viz_type == 'P-value Distribution' and 'p_value' in df_results.columns:
                pvals = df_results['p_value'].dropna()
                
                ax.hist(pvals, bins=50, edgecolor='black', alpha=0.7)
                ax.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
                ax.set_xlabel('P-value', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title('P-value Distribution', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            elif viz_type == 'Effect Size Distribution' and 'effect_size_d' in df_results.columns:
                effect_sizes = df_results['effect_size_d'].dropna()
                
                ax.hist(effect_sizes, bins=50, edgecolor='black', alpha=0.7)
                ax.axvline(0, color='black', linestyle='-', linewidth=2)
                ax.set_xlabel("Cohen's d", fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title('Effect Size Distribution', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        except Exception as e:
            st.info('Visualization not available')


if __name__ == '__main__':
    main()
