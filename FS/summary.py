import os
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Load and clean data
data_path = "data/alldata251124.xlsx"
data = pd.read_excel(data_path)

cols_to_drop = []
data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

data = data.drop_duplicates().dropna(how='all')
X = data.drop(columns=["speed"])
y = data["speed"]

# Group features
known_continuous = ["MW","Production","PL","Num","Diameter","BL","MTemp","BTemp"]
continuous_features = [c for c in known_continuous if c in X.columns]
binary_features = [c for c in X.columns if c not in continuous_features and set(X[c].dropna().unique()).issubset({0,1})]
numeric_features = continuous_features + binary_features

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.fillna(0))

# 1. Mutual Information
discrete_mask = np.array([col in binary_features for col in X.columns])
mi_scores = mutual_info_regression(X_scaled, y.fillna(y.mean()), discrete_features=discrete_mask)
mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': np.abs(mi_scores)})

# 2. Spearman Correlation
spearman_list = [(feat, abs(data[feat].corr(y, method='spearman'))) for feat in numeric_features]
spearman_df = pd.DataFrame(spearman_list, columns=['feature','spearman_corr']).dropna()

# 3. ANOVA F-test
X_cont = X[numeric_features].fillna(X[numeric_features].mean())
f_scores, f_pvals = f_regression(X_cont, y.fillna(y.mean()))
f_df = pd.DataFrame({'feature': X_cont.columns, 'f_score': np.abs(f_scores), 'f_pvalue': f_pvals})

# 4. Covariance
cov_base = data[numeric_features + ["speed"]].copy()
cov_series = cov_base.cov(numeric_only=True).get("speed", pd.Series(dtype=float)).drop(labels=["speed"], errors='ignore')
cov_df = pd.DataFrame({'feature': cov_series.index, 'covariance': cov_series.abs().values})

# 5. Tree-based and Linear Models
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.fillna(y.mean()), test_size=0.3, random_state=42)

rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
rf_imp_df = pd.DataFrame({'feature': X.columns, 'rf_importance': np.abs(rf.feature_importances_)})

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42).fit(X_train, y_train)
xgb_imp_df = pd.DataFrame({'feature': X.columns, 'xgb_importance': np.abs(xgb_model.feature_importances_)})

linear_model = LinearRegression().fit(X.fillna(0), y.fillna(y.mean()))
linear_df = pd.DataFrame({'feature': X.columns, 'linear_coefficient': np.abs(linear_model.coef_)})

# Merge evaluation metrics
dfs_to_merge = [mi_df, spearman_df, f_df, cov_df, rf_imp_df, linear_df]
combined_df = reduce(lambda left, right: pd.merge(left, right, on='feature', how='outer'), dfs_to_merge)
num_cols = combined_df.select_dtypes(include=['number']).columns
combined_df[num_cols] = combined_df[num_cols].abs()

# TOPSIS Evaluation
dec_mat = combined_df[num_cols].values.astype(float)
col_min, col_max = dec_mat.min(axis=0), dec_mat.max(axis=0)
range_ = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
r = (dec_mat - col_min) / range_ 

is_benefit = np.array([not ('pvalue' in c.lower() or 'p_val' in c.lower() or c.lower().endswith('_p')) for c in num_cols])

# Entropy weights with Stability Bias Term (SAM-FS approach)
m, n = dec_mat.shape
eps = 1e-24

dec_for_entropy = dec_mat.copy()
for j, benefit in enumerate(is_benefit):
    if not benefit:
        dec_for_entropy[:, j] = 1.0 / (dec_for_entropy[:, j] + eps)
dec_for_entropy[dec_for_entropy < 0] = 0.0

col_sums = np.where(dec_for_entropy.sum(axis=0) == 0, eps, dec_for_entropy.sum(axis=0))
pij = dec_for_entropy / col_sums

k = 1.0 / np.log(m) if m > 1 else 0.0
pj_log = np.where(pij > 0, pij * np.log(pij), 0.0)
entropy = -k * pj_log.sum(axis=0)
diversity = 1 - entropy

# Apply stability bias term
lam = np.mean(diversity)
var_j = np.var(dec_for_entropy, axis=0)
diversity_adjusted = diversity + lam * var_j

if np.allclose(diversity_adjusted, 0):
    weights = np.ones(n) / n
else:
    weights = diversity_adjusted / (diversity_adjusted.sum() + eps)

v = r * weights
ideal_best = np.where(is_benefit, v.max(axis=0), v.min(axis=0))
ideal_worst = np.where(is_benefit, v.min(axis=0), v.max(axis=0))

dist_pos = np.sqrt(((v - ideal_best) ** 2).sum(axis=1))
dist_neg = np.sqrt(((v - ideal_worst) ** 2).sum(axis=1))
den = np.where(dist_pos + dist_neg == 0, 1.0, dist_pos + dist_neg)

combined_df['topsis_score'] = dist_neg / den
combined_df['topsis_rank'] = combined_df['topsis_score'].rank(ascending=False, method='min').astype('Int64')

# Save outputs
os.makedirs("results", exist_ok=True)
combined_df.to_csv("results/feature_stats_combined.csv", index=False)
combined_df[['feature','topsis_score','topsis_rank']].to_csv("results/feature_ranking_topsis.csv", index=False)
print("Files generated: feature_stats_combined.csv & feature_ranking_topsis.csv")
