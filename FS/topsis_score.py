import os
import sys
import argparse
import numpy as np
import pandas as pd

def infer_column_rules(cols):
    abs_keywords = ['pearson', 'spearman', 'covariance', 'covar', 't_stat', 't-stat', 'tstat']
    cost_keywords = ['pvalue', 'p_value', 'p-value', '_p', 'pval', 'p_val']
    benefit_explicit = {'mi_score', 'chi2_score', 'f_score', 'rf_importance', 'xgb_importance'}
    
    abs_cols, cost_cols = set(), set()
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in abs_keywords): abs_cols.add(c)
        if any(k in cl for k in cost_keywords): cost_cols.add(c)
        if c in benefit_explicit: cost_cols.discard(c)
            
    return abs_cols, cost_cols

def minmax_normalize(df, cost_cols=set(), eps=1e-12):
    df_norm = df.copy().astype(float)
    for col in df.columns:
        col_vals = df_norm[col].values
        minv, maxv = np.nanmin(col_vals), np.nanmax(col_vals)
        if np.isclose(maxv, minv):
            df_norm[col] = 0.0
        else:
            if col in cost_cols:
                df_norm[col] = (maxv - col_vals) / (maxv - minv)
            else:
                df_norm[col] = (col_vals - minv) / (maxv - minv)
    return df_norm.fillna(0.0)

def entropy_weights(df_norm, eps=1e-12):
    m, n = df_norm.shape
    P = df_norm.values.astype(float)
    col_sums = P.sum(axis=0)
    col_sums[col_sums == 0] = eps
    P = P / col_sums
    
    with np.errstate(divide='ignore', invalid='ignore'):
        logP = np.where(P > 0, np.log(P), 0.0)
        
    k = 1.0 / np.log(m) if m > 1 else 1.0
    e = -k * (P * logP).sum(axis=0)
    d = 1 - e

    # Apply stability bias term (SAM-FS approach)
    lam = np.mean(d)
    var_j = np.var(df_norm.values.astype(float), axis=0)
    d_adjusted = d + lam * var_j
    
    if np.allclose(d_adjusted, 0):
        w = np.ones_like(d_adjusted) / len(d_adjusted)
    else:
        w = d_adjusted / d_adjusted.sum()
        
    return pd.Series(w, index=df_norm.columns)

def topsis(df_norm, weights):
    V = df_norm.values * weights.values[np.newaxis, :]
    v_pos, v_neg = V.max(axis=0), V.min(axis=0)
    
    d_pos = np.sqrt(((V - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - v_neg) ** 2).sum(axis=1))
    
    den = d_pos + d_neg
    den[den == 0] = 1e-12
    C = d_neg / den
    
    return pd.Series(C, index=df_norm.index), pd.Series(d_pos, index=df_norm.index), pd.Series(d_neg, index=df_norm.index)

def run(input_path, output_dir=None, save_intermediate=True):
    output_dir = output_dir or os.path.dirname(input_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path, index_col=0)
    if df.shape[0] == 0: raise ValueError("Empty input file: " + input_path)

    abs_cols, cost_cols = infer_column_rules(df.columns)
    df_proc = df.copy()
    
    for c in df_proc.columns:
        df_proc[c] = pd.to_numeric(df_proc[c], errors='coerce')
        if c in abs_cols: df_proc[c] = df_proc[c].abs()
    df_proc = df_proc.fillna(df_proc.mean())

    df_norm = minmax_normalize(df_proc, cost_cols=cost_cols)
    weights = entropy_weights(df_norm)
    scores, dpos, dneg = topsis(df_norm, weights)

    result = pd.DataFrame({'score': scores, 'd_pos': dpos, 'd_neg': dneg}, index=df_norm.index)
    result['rank'] = result['score'].rank(ascending=False, method='min').astype(int)
    result = result.sort_values('score', ascending=False)

    if save_intermediate:
        result.to_csv(os.path.join(output_dir, 'topsis_entropy_results.csv'), index=True)
        weights.rename('weight').to_frame().to_csv(os.path.join(output_dir, 'topsis_entropy_weights.csv'))
        df_norm.to_csv(os.path.join(output_dir, 'topsis_normalized_matrix.csv'), index=True)
        
    print(result.head(20))
    return result, weights, df_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOPSIS evaluation with entropy weights")
    parser.add_argument("--input", "-i", default=os.path.join(os.path.dirname(__file__), "汇总结果/feature_stats_combined.csv"), help="Input CSV path")
    parser.add_argument("--output", "-o", default="综合评价结果", help="Output directory")
    args = parser.parse_args()
    
    try:
        run(args.input, args.output)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
