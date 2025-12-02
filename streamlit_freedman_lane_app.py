# streamlit_freedman_lane_app.py
"""
Streamlit app for Freedman-Lane permutation adjustment and Cohen's d bootstrapping.

Features:
- Upload clinical and feature CSVs (or a single CSV with both clinical and features).
- Detects wide or long format; supports melting wide -> long using a feature-pattern or manual columns.
- Choose ID columns, group column (e.g., bmi_group), covariates to adjust for (AGE, SEX, etc.).
- Define pairwise comparisons or automatic pairwise across group levels.
- Runs Freedman-Lane permutation test (explicit pair) and bootstrap Cohen's d on residuals.
- Parallel processing (joblib) with configurable workers.
- Shows results, basic plots, and allows CSV download.

Usage:
$ pip install -r requirements.txt
$ streamlit run streamlit_freedman_lane_app.py

Requirements (example):
streamlit
pandas
numpy
statsmodels
joblib
scipy

This file is intended to be committed to GitHub and deployed on Streamlit Cloud or any other host.
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

# ---------- Utilities ----------

def extract_base_sample_id(sample_id: Any) -> str:
    """Extracts a base ID from various sample id formats.
    Examples: C3L-00017-02 -> C3L-00017 ; semicolon-separated -> take first.
    """
    if pd.isna(sample_id):
        return ""
    s = str(sample_id)
    if ';' in s:
        s = s.split(';')[0]
    parts = s.split('-')
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return s


def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    """Heuristic: columns containing '||' indicate celltype||signature style wide features.
    Falls back to numeric columns excluding the id/clinical columns.
    """
    cols = [c for c in df.columns if '||' in c]
    if cols:
        return cols
    # otherwise numeric columns (exclude typical clinical names)
    exclude = {'sample_id', 'sampleId', 'sample', 'id', 'bmi_group', 'BMI', 'AGE', 'SEX'}
    numeric_cols = [c for c, t in df.dtypes.items() if (np.issubdtype(t, np.number) and c not in exclude)]
    return numeric_cols


# ---------- Statistical helpers ----------

def compute_cohens_d(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    n1, n2 = len(a), len(b)
    m1, m2 = np.nanmean(a), np.nanmean(b)
    v1 = np.nanvar(a, ddof=1) if n1 > 1 else 0.0
    v2 = np.nanvar(b, ddof=1) if n2 > 1 else 0.0
    denom = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2) if (n1 + n2 - 2) > 0 else 0.0
    sd_pooled = math.sqrt(denom) if denom > 0 else 0.0
    if sd_pooled == 0:
        return np.nan
    return (m2 - m1) / sd_pooled


def bootstrap_cohens_d(resid_vals, group_labels, n_boot=1000, seed=42):
    resid_vals = np.asarray(resid_vals)
    labels = np.asarray(group_labels)
    uniq = np.unique(labels[~pd.isna(labels)])
    if len(uniq) != 2:
        return {"d": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0}
    a_idx = np.where(labels == uniq[0])[0]
    b_idx = np.where(labels == uniq[1])[0]
    n1, n2 = len(a_idx), len(b_idx)
    if n1 < 2 or n2 < 2:
        return {"d": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0}

    rng = np.random.RandomState(seed)
    # Draw bootstrap samples of indices within each group
    idx_a = rng.randint(0, n1, size=(n_boot, n1))
    idx_b = rng.randint(0, n2, size=(n_boot, n2))

    arr_a = resid_vals[a_idx][idx_a]
    arr_b = resid_vals[b_idx][idx_b]

    mean_a = arr_a.mean(axis=1)
    mean_b = arr_b.mean(axis=1)
    var_a = arr_a.var(axis=1, ddof=1)
    var_b = arr_b.var(axis=1, ddof=1)

    denom = ((n1 - 1) * var_a + (n2 - 1) * var_b) / (n1 + n2 - 2)
    sd_pooled = np.sqrt(np.maximum(denom, 1e-20))
    boot_ds = (mean_b - mean_a) / sd_pooled
    boot_ds = boot_ds[np.isfinite(boot_ds)]

    if boot_ds.size < max(50, int(0.1 * n_boot)):
        d0 = compute_cohens_d(resid_vals[labels == uniq[0]], resid_vals[labels == uniq[1]])
        return {"d": float(d0), "ci_low": np.nan, "ci_high": np.nan, "n_boot": int(boot_ds.size)}

    d0 = compute_cohens_d(resid_vals[labels == uniq[0]], resid_vals[labels == uniq[1]])
    ci_low = np.percentile(boot_ds, 2.5)
    ci_high = np.percentile(boot_ds, 97.5)
    return {"d": float(d0), "ci_low": float(ci_low), "ci_high": float(ci_high), "n_boot": int(len(boot_ds))}


# Freedman-Lane permutation with explicit groups
def run_freedman_lane(feature_df: pd.DataFrame, group_col: str, covariates: List[str],
                      groupA: Optional[Any] = None, groupB: Optional[Any] = None,
                      n_perm: int = 2000, rng_seed: int = 42) -> Dict[str, Any]:
    df = feature_df.copy().reset_index(drop=True)

    # Fit covariates-only model to get residuals (or global mean if no covariates)
    if not covariates:
        fitted = np.repeat(df["Z"].mean(), len(df))
        resid = df["Z"].values - fitted
    else:
        terms = []
        for c in covariates:
            if c in df.columns:
                if df[c].dtype == "object" or df[c].nunique() < 10:
                    terms.append(f"C({c})")
                else:
                    terms.append(c)
        if not terms:
            fitted = np.repeat(df["Z"].mean(), len(df))
            resid = df["Z"].values - fitted
        else:
            try:
                modr = smf.ols(formula="Z ~ " + " + ".join(terms), data=df.dropna(subset=["Z"] + covariates)).fit()
                try:
                    fitted = modr.predict(df)
                    resid = df["Z"].values - fitted
                except Exception:
                    # if index mismatch, align fittedvalues
                    complete_mask = df[covariates].notna().all(axis=1)
                    fitted = np.repeat(np.nan, len(df))
                    fitted[complete_mask] = modr.fittedvalues.reindex(df[complete_mask].index).values
                    resid = df["Z"].values - np.nan_to_num(fitted, nan=df["Z"].mean())
            except Exception:
                fitted = np.repeat(df["Z"].mean(), len(df))
                resid = df["Z"].values - fitted

    # Fit full model including group
    try:
        terms_full = [f"C({group_col})"]
        if covariates:
            for c in covariates:
                if c in df.columns:
                    if df[c].dtype == "object" or df[c].nunique() < 10:
                        terms_full.append(f"C({c})")
                    else:
                        terms_full.append(c)
        formula_full = "Z ~ " + " + ".join(terms_full)
        full_df_for_fit = df.dropna(subset=["Z"])
        modf = smf.ols(formula=formula_full, data=full_df_for_fit).fit()

        # compute covariate mean/mode for prediction rows
        cov_mean_vals = {}
        for c in covariates or []:
            if c in full_df_for_fit.columns:
                if full_df_for_fit[c].dtype.kind in "biufc":
                    cov_mean_vals[c] = float(full_df_for_fit[c].mean())
                else:
                    cov_mean_vals[c] = full_df_for_fit[c].mode().iloc[0] if not full_df_for_fit[c].mode().empty else full_df_for_fit[c].dropna().iloc[0]

        groups = df[group_col].dropna().unique()
        gvals = sorted([g for g in groups if pd.notna(g)])
        if len(gvals) < 2 and (groupA is None or groupB is None):
            return {"obs": np.nan, "p_perm": np.nan, "perm_dist": np.array([])}

        # pick explicit pairs if provided, else first two sorted
        g1, g2 = (groupA, groupB) if (groupA is not None and groupB is not None) else (gvals[0], gvals[1])

        # ensure present
        vals = df.dropna(subset=["Z", group_col])
        present = set(vals[group_col].unique())
        if g1 not in present or g2 not in present:
            # fallback: pick first two available
            available = sorted(list(present))
            if len(available) < 2:
                return {"obs": np.nan, "p_perm": np.nan, "perm_dist": np.array([])}
            g1, g2 = available[0], available[1]

        row1 = {group_col: g1}
        row2 = {group_col: g2}
        for k, v in cov_mean_vals.items():
            row1[k] = v
            row2[k] = v

        pred1 = float(modf.predict(pd.DataFrame([row1]))[0])
        pred2 = float(modf.predict(pd.DataFrame([row2]))[0])
        obs_stat = pred2 - pred1
    except Exception:
        try:
            vals = df.dropna(subset=["Z", group_col])
            grp_sorted = sorted(vals[group_col].unique())
            obs_stat = vals[vals[group_col]==grp_sorted[1]]["Z"].mean() - vals[vals[group_col]==grp_sorted[0]]["Z"].mean()
        except Exception:
            obs_stat = np.nan

    rng = np.random.RandomState(rng_seed)
    perm_stats = []
    n = len(df)

    # permutation loop
    for i in range(n_perm):
        perm_idx = rng.permutation(n)
        perm_resid = resid[perm_idx]
        Y_perm = fitted + perm_resid
        df_perm = df.copy()
        df_perm["Z"] = Y_perm
        try:
            modp = smf.ols(formula=formula_full, data=df_perm.dropna(subset=["Z"] + (covariates or []))).fit()
            pred1_p = float(modp.predict(pd.DataFrame([row1]))[0])
            pred2_p = float(modp.predict(pd.DataFrame([row2]))[0])
            stat_p = pred2_p - pred1_p
            if np.isfinite(stat_p):
                perm_stats.append(stat_p)
        except Exception:
            continue

    perm_stats = np.array(perm_stats)
    if perm_stats.size == 0 or not np.isfinite(obs_stat):
        return {"obs": obs_stat, "p_perm": np.nan, "perm_dist": perm_stats}

    p_perm = (np.sum(np.abs(perm_stats) >= abs(obs_stat)) + 1) / (len(perm_stats) + 1)
    return {"obs": float(obs_stat), "p_perm": float(p_perm), "perm_dist": perm_stats}


# analyze a single feature (long-format DF with 'Z')
def analyze_feature(feat_name: str, df_feat: pd.DataFrame, model_name: str, covariates: List[str],
                    comp: Tuple[Any, Any, str], min_n: int = 8, n_perm: int = 2000, n_boot: int = 1000,
                    rng_seed: int = 42) -> Optional[Dict[str, Any]]:
    try:
        gA, gB, comp_name = comp
        sub = df_feat[df_feat['bmi_group'].isin([gA, gB])].copy()
        nA = int((sub['bmi_group'] == gA).sum())
        nB = int((sub['bmi_group'] == gB).sum())
        if nA < min_n or nB < min_n:
            return None

        df_for_test = sub.copy().reset_index(drop=True)

        # create residuals adjusted for covariates
        if not covariates:
            fitted_vals = np.repeat(df_for_test['Z'].mean(), len(df_for_test))
            resid_vals = df_for_test['Z'].values - fitted_vals
        else:
            terms = []
            for c in covariates:
                if c in df_for_test.columns:
                    if df_for_test[c].dtype == 'object' or df_for_test[c].nunique() < 10:
                        terms.append(f"C({c})")
                    else:
                        terms.append(c)
            if not terms:
                fitted_vals = np.repeat(df_for_test['Z'].mean(), len(df_for_test))
                resid_vals = df_for_test['Z'].values - fitted_vals
            else:
                formula_r = 'Z ~ ' + ' + '.join(terms)
                try:
                    modr = smf.ols(formula=formula_r, data=df_for_test.dropna(subset=['Z'] + covariates)).fit()
                    try:
                        preds = modr.predict(df_for_test)
                        fitted_vals = preds
                        resid_vals = df_for_test['Z'].values - preds
                    except Exception:
                        complete = df_for_test[covariates].notna().all(axis=1)
                        fitted_vals = np.repeat(np.nan, len(df_for_test))
                        fitted_vals[complete] = modr.fittedvalues.reindex(df_for_test[complete].index).values
                        mean_f = np.nanmean(fitted_vals)
                        fitted_vals = np.nan_to_num(fitted_vals, nan=mean_f)
                        resid_vals = df_for_test['Z'].values - fitted_vals
                except Exception:
                    fitted_vals = np.repeat(df_for_test['Z'].mean(), len(df_for_test))
                    resid_vals = df_for_test['Z'].values - fitted_vals

        perm_res = run_freedman_lane(df_for_test, 'bmi_group', covariates, groupA=gA, groupB=gB, n_perm=n_perm, rng_seed=rng_seed)
        obs_diff = perm_res.get('obs', np.nan)
        p_perm = perm_res.get('p_perm', np.nan)

        d_boot = bootstrap_cohens_d(resid_vals, df_for_test['bmi_group'].values, n_boot=n_boot, seed=rng_seed)
        d_val = d_boot.get('d', np.nan)
        d_lo = d_boot.get('ci_low', np.nan)
        d_hi = d_boot.get('ci_high', np.nan)
        boot_n = d_boot.get('n_boot', 0)

        return {
            'feature': feat_name,
            'CellType': df_for_test.get('CellType', pd.Series([''])).iloc[0] if 'CellType' in df_for_test.columns else '',
            'Signature': df_for_test.get('Signature', pd.Series([''])).iloc[0] if 'Signature' in df_for_test.columns else '',
            'model': model_name,
            'covariates_used': str(covariates),
            'comparison': comp[2],
            'groupA': comp[0],
            'groupB': comp[1],
            'nA': nA,
            'nB': nB,
            'obs_diff': obs_diff,
            'p_perm': p_perm,
            'cohens_d': d_val,
            'd_ci_low': d_lo,
            'd_ci_high': d_hi,
            'boot_n': boot_n
        }
    except Exception as e:
        return {'feature': feat_name, 'error': str(e)[:200]}


# ---------- Streamlit UI / App ----------

def app():
    st.set_page_config(page_title='Freedman-Lane Permutation (Streamlit)', layout='wide')
    st.title('Freedman–Lane permutation + Cohen\'s d bootstrap')

    st.markdown(
        """
        Upload your data (feature matrix and clinical metadata). App supports:
        - Wide-format features (one column per feature, rows = samples) + clinical file, OR
        - Single long-format file with columns: sample id, feature, Z, and clinical columns.

        The app will compute Freedman–Lane permutation tests for pairwise group comparisons and bootstrap Cohen's d on residuals after adjusting for covariates.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        uploaded_features = st.file_uploader('Upload feature CSV (wide or long). If combined with clinical columns, leave clinical upload blank.', type=['csv'])
        uploaded_clinical = st.file_uploader('(Optional) Upload clinical CSV (if features file does not contain clinical columns)', type=['csv'])

    with col2:
        st.info('Hints: feature columns often contain "||" in our pipeline. If using a long file, ensure columns include sample id, feature, Z.')
        st.write('Example workflow: upload wide features CSV, upload clinical CSV, choose ID columns and feature columns.')

    if not uploaded_features:
        st.stop()

    # Load feature file
    try:
        features_df = pd.read_csv(uploaded_features)
    except Exception as e:
        st.error(f'Failed to read features CSV: {e}')
        st.stop()

    # Load clinical if provided
    clinical_df = None
    if uploaded_clinical:
        try:
            clinical_df = pd.read_csv(uploaded_clinical)
        except Exception as e:
            st.error(f'Failed to read clinical CSV: {e}')
            st.stop()

    st.sidebar.header('Analysis options')
    format_type = st.sidebar.selectbox('Feature file format', ['auto-detect', 'wide', 'long'], index=0)

    # ID selection
    st.sidebar.subheader('ID & feature selection')
    sample_id_col = st.sidebar.selectbox('Sample ID column (features file)', options=features_df.columns.tolist(), index=0)

    # If long, user must pick feature and Z columns
    if format_type == 'long' or (format_type == 'auto-detect' and ('feature' in features_df.columns and 'Z' in features_df.columns)):
        is_long = True
    else:
        is_long = False

    if is_long:
        feature_col = st.sidebar.selectbox('Feature column (long file)', options=[c for c in features_df.columns if c != sample_id_col])
        z_col = st.sidebar.selectbox('Z / value column (long file)', options=[c for c in features_df.columns if c not in [sample_id_col, feature_col]])
    else:
        detected = detect_feature_columns(features_df)
        st.sidebar.write(f'Heuristic-detected feature columns (first 20): {detected[:20]}')
        feature_cols = st.sidebar.multiselect('Feature columns (wide) to include', options=features_df.columns.tolist(), default=detected[:min(50, len(detected))])

    # Clinical merging
    clinical_id_col = None
    if clinical_df is not None:
        clinical_id_col = st.sidebar.selectbox('Clinical file ID column', options=clinical_df.columns.tolist(), index=0)

    # Determine clinical columns available
    merged_df = None
    if is_long:
        # Use features_df (long) as main
        df_long = features_df.rename(columns={sample_id_col: 'sample_id_full', feature_col: 'feature', z_col: 'Z'})
        df_long['base_sample_id'] = df_long['sample_id_full'].apply(extract_base_sample_id)
        if clinical_df is not None:
            clinical_df['base_sample_id'] = clinical_df[clinical_id_col].apply(extract_base_sample_id)
            clinical_subset = clinical_df.copy()
            merged_df = df_long.merge(clinical_subset, on='base_sample_id', how='left')
        else:
            merged_df = df_long.copy()
    else:
        # wide -> melt
        features_df = features_df.rename(columns={sample_id_col: 'sample_id_full'})
        features_df['base_sample_id'] = features_df['sample_id_full'].apply(extract_base_sample_id)
        if clinical_df is not None:
            clinical_df['base_sample_id'] = clinical_df[clinical_id_col].apply(extract_base_sample_id)
            clinical_subset = clinical_df.copy()
            # select feature_cols
            if not feature_cols:
                st.error('No feature columns selected for wide file.')
                st.stop()
            df_long = features_df.melt(id_vars=['sample_id_full', 'base_sample_id'], value_vars=feature_cols, var_name='feature', value_name='Z')
            merged_df = df_long.merge(clinical_subset, on='base_sample_id', how='left')
        else:
            if not feature_cols:
                st.error('No feature columns selected for wide file and no clinical file provided.')
                st.stop()
            df_long = features_df.melt(id_vars=['sample_id_full', 'base_sample_id'], value_vars=feature_cols, var_name='feature', value_name='Z')
            merged_df = df_long.copy()

    st.write('Merged data preview')
    st.dataframe(merged_df.head())

    # Identify candidate clinical columns
    candidate_clin_cols = [c for c in merged_df.columns if c not in ['sample_id_full', 'base_sample_id', 'feature', 'Z']]
    st.sidebar.subheader('Model & covariates')
    group_col = st.sidebar.selectbox('Group column (categorical)', options=candidate_clin_cols, index=0 if candidate_clin_cols else None)
    covariates = st.sidebar.multiselect('Covariates to adjust for (optional)', options=candidate_clin_cols, default=[]) if candidate_clin_cols else []

    # Allow user to create bmi_group from BMI
    if 'BMI' in candidate_clin_cols and 'bmi_group' not in merged_df.columns:
        if st.sidebar.checkbox('Create bmi_group from BMI column'):
            merged_df['BMI'] = pd.to_numeric(merged_df['BMI'], errors='coerce')
            merged_df['bmi_group'] = pd.cut(merged_df['BMI'], bins=[0,18.5,25,30,100], labels=['underweight','normal_weight','over_weight','obese'], include_lowest=True)
            st.success('bmi_group created')
            if group_col is None:
                group_col = 'bmi_group'

    st.sidebar.subheader('Comparisons & thresholds')
    auto_pairwise = st.sidebar.checkbox('Auto pairwise across group levels', value=True)

    manual_pairs_txt = st.sidebar.text_area('Manual comparisons (one per line, format: groupA,groupB,label)', value='')

    if not auto_pairwise and not manual_pairs_txt.strip():
        st.sidebar.warning('Either enable auto pairwise or specify manual comparisons')

    # parameters
    N_PERM = st.sidebar.number_input('Number of permutations (N_PERM)', value=2000, min_value=100, step=100)
    N_BOOT = st.sidebar.number_input('Number of bootstrap iterations (N_BOOT)', value=1000, min_value=100, step=100)
    MIN_N_PER_GROUP = st.sidebar.number_input('Min observations per group (rows)', value=8, min_value=2, step=1)
    WORKERS = st.sidebar.number_input('Parallel workers (joblib)', value=1, min_value=1, step=1)

    if st.button('Run analysis'):
        # prepare comparisons
        available_levels = sorted([g for g in merged_df[group_col].dropna().unique()]) if group_col in merged_df.columns else []
        comparisons = []
        if auto_pairwise and len(available_levels) >= 2:
            for i in range(len(available_levels)):
                for j in range(i+1, len(available_levels)):
                    comparisons.append((available_levels[i], available_levels[j], f"{available_levels[j]}_vs_{available_levels[i]}"))
        if manual_pairs_txt.strip():
            for line in manual_pairs_txt.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    comparisons.append((parts[0], parts[1], parts[2]))
                elif len(parts) == 2:
                    comparisons.append((parts[0], parts[1], f"{parts[1]}_vs_{parts[0]}"))

        if not comparisons:
            st.error('No comparisons defined. Please enable auto pairwise or provide manual comparisons.')
            st.stop()

        st.info(f'Running {len(comparisons)} comparisons; N_PERM={N_PERM}, N_BOOT={N_BOOT}, workers={WORKERS}')

        # prepare list of features
        features = sorted(merged_df['feature'].unique())
        jobs = []
        for comp in comparisons:
            gA, gB, cname = comp
            for feat in features:
                df_feat = merged_df[merged_df['feature'] == feat]
                nA = int((df_feat[group_col] == gA).sum())
                nB = int((df_feat[group_col] == gB).sum())
                if nA >= MIN_N_PER_GROUP and nB >= MIN_N_PER_GROUP:
                    jobs.append((feat, df_feat, 'freedman_lane_model', covariates, (gA, gB, cname)))

        st.write(f'Jobs to run: {len(jobs)} (features x comparisons passing min counts)')
        if len(jobs) == 0:
            st.error('No jobs to run - adjust MIN_N_PER_GROUP or check groups')
            st.stop()

        progress_bar = st.progress(0)
        results = []

        # chunk and parallelize
        cpu = int(WORKERS)
        # ensure no oversubscription
        cpu = max(1, min(cpu, len(jobs)))

        def run_job(j):
            feat, df_feat, model_name, covs, comp = j
            return analyze_feature(feat, df_feat, model_name, covs, comp, min_n=MIN_N_PER_GROUP, n_perm=int(N_PERM), n_boot=int(N_BOOT), rng_seed=42)

        if cpu == 1:
            for idx, job in enumerate(jobs):
                res = run_job(job)
                if res is not None:
                    results.append(res)
                progress_bar.progress((idx+1)/len(jobs))
        else:
            # run in parallel and update progress in batches
            parallel = Parallel(n_jobs=cpu, backend='loky')
            batch_size = max(1, len(jobs)//(cpu*2))
            out = parallel(delayed(run_job)(job) for job in jobs)
            for idx, r in enumerate(out):
                if r is not None:
                    results.append(r)
                if idx % max(1, int(len(out)/100)) == 0:
                    progress_bar.progress((idx+1)/len(out))
            progress_bar.progress(1.0)

        if not results:
            st.error('No results produced (all jobs filtered or errors).')
            st.stop()

        df_res = pd.DataFrame(results)
        # numeric coercion and FDR per comparison
        df_res['p_perm'] = pd.to_numeric(df_res.get('p_perm', pd.Series([1]*len(df_res))), errors='coerce').fillna(1.0)
        df_res['p_adj'] = np.nan
        df_res['significant_fdr'] = False

        # group by comparison
        groups = df_res.groupby(['model', 'comparison']).groups
        try:
            from statsmodels.stats.multitest import fdrcorrection
            for (model, comp), idxs in groups.items():
                pvals = df_res.loc[idxs, 'p_perm'].astype(float).values
                if pvals.size == 0:
                    continue
                rej, qvals = fdrcorrection(pvals, alpha=0.05, method='indep')
                df_res.loc[idxs, 'p_adj'] = qvals
                df_res.loc[idxs, 'significant_fdr'] = rej
        except Exception:
            st.warning('fdrcorrection not applied')

        st.subheader('Results (top 200 rows)')
        st.dataframe(df_res.head(200))

        # allow download
        csv_bytes = df_res.to_csv(index=False).encode('utf-8')
        st.download_button('Download results CSV', data=csv_bytes, file_name='freedman_lane_results.csv', mime='text/csv')

        st.success('Analysis complete')


if __name__ == '__main__':
    app()
