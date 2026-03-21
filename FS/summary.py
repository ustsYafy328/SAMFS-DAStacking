import pandas as pd
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.feature_selection import chi2
from scipy.stats import ttest_ind, pointbiserialr, mannwhitneyu, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
import numpy as np
import os

# 读取数据（只读一次）
data_path = "data/alldata251124.xlsx"
data = pd.read_excel(data_path)
# data = None
# for enc in ["utf-8", "gbk", "gb18030", "latin1"]:
#     try:
#         data = pd.read_excel(data_path, encoding=enc)
#         if enc != "utf-8":
#             print(f"文件编码检测：使用 {enc} 成功读取 {data_path}")
#         break
#     except UnicodeDecodeError as e:
#         last_err = e
# if data is None:
#     try:
#         # 最后降级方案：python 引擎 + latin1
#         data = pd.read_csv(data_path, encoding="latin1", engine="python")
#         print(f"文件编码降级：使用 latin1 + python 引擎读取 {data_path}")
#     except Exception:
#         raise last_err

# 新增：剔除不需要的列（若存在）
for col_to_drop in ["生产批号", "模具编码","生产批产量","加工长度","短棒长度","挤压出口温度","进棒区设定温度","出棒区设定温度","模具上机温度"]:
    if col_to_drop in data.columns:
        data = data.drop(columns=[col_to_drop])
        print(f"已剔除列: {col_to_drop}")

# 删除重复行
data = data.drop_duplicates()
# 删除空缺值的行
data = data.dropna(how='all')
# 提取特征和目标变量
X = data.drop(columns=["挤压速度"])
y = data["挤压速度"]

# 明确已知的连续特征（依据你提供的名单）并校验
known_continuous = [
                    "产品米重","模具直径","模具类型","进棒区设定温度","出棒区设定温度","淬火出口温度","型材截面面积","一出几",
                    "挤压比","一出几难度系数","表面及模具类型难度系数","修正系数和","型材难度系数","难度系数总合"
                    ]
continuous_features = [c for c in known_continuous if c in X.columns]
# 其余作为二元/类别（独热）特征（只保留值为 0/1 的列）
binary_candidates = [c for c in X.columns if c not in continuous_features]
binary_features = [c for c in binary_candidates if set(X[c].dropna().unique()).issubset({0,1})]

# 新增：将连续特征与二元(0/1)特征合并为可统一处理的 numeric_features
numeric_features = continuous_features + binary_features

# 1) 互信息（对数值型与二元特征均可）
scaler = MinMaxScaler()
X_scaled_for_mi = scaler.fit_transform(X.fillna(0))  # 互信息对缺失要处理
# discrete_features 告知哪些特征应作为离散特征处理（binary）
discrete_mask = np.array([col in binary_features for col in X.columns])
mi_scores = mutual_info_regression(X_scaled_for_mi, y.fillna(y.mean()), discrete_features=discrete_mask)
mi_df = pd.Series(mi_scores, index=X.columns).to_frame(name='mi_score').reset_index()
mi_df.columns = ['feature', 'mi_score']
mi_df['mi_score'] = mi_df['mi_score'].abs()

# 2) Pearson / Spearman / Kendall（对 numeric_features：包括连续与 0/1）
spearman_list = []
for feat in numeric_features:
    spearman_val = data[feat].corr(y, method='spearman')
    spearman_list.append((feat, abs(spearman_val) if pd.notna(spearman_val) else np.nan))

spearman_df = pd.DataFrame(spearman_list, columns=['feature','spearman_corr'])

# 5) ANOVA F检验：对 numeric_features（连续与 0/1 都可使用）
if len(numeric_features) > 0:
    X_cont = X[numeric_features].fillna(X[numeric_features].mean())
    f_scores, f_pvals = f_regression(X_cont, y.fillna(y.mean()))
    f_df = pd.DataFrame({'feature': X_cont.columns, 'f_score': np.abs(f_scores), 'f_pvalue': f_pvals})
else:
    f_df = pd.DataFrame(columns=['feature','f_score','f_pvalue'])

# 6) 互补的协方差（对所有数值列），但我们只保留与目标的协方差
# cov_series = data.cov().get("挤压速度", pd.Series()).drop(labels=["挤压速度"], errors='ignore')
# cov_df = cov_series.reset_index
# cov_df.columns = ['feature', 'covariance']
# cov_df['covariance'] = cov_df['covariance'].abs()
cov_base = data[(numeric_features + ["挤压速度"])].copy()
cov_matrix = cov_base.cov(numeric_only=True)
cov_series = cov_matrix.get("挤压速度", pd.Series(dtype=float)).drop(labels=["挤压速度"], errors='ignore')
cov_df = cov_series.reset_index()
cov_df.columns = ['feature', 'covariance']
cov_df['covariance'] = cov_df['covariance'].abs()

# 7) 随机森林、XGBoost、线性回归（与之前类似，但加入置换重要性）
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.fillna(0))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.fillna(y.mean()), test_size=0.3, random_state=42)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_imp_df = pd.DataFrame({'feature': X.columns, 'rf_importance': rf.feature_importances_})
rf_imp_df['rf_importance'] = rf_imp_df['rf_importance'].abs()
rf_imp_df = rf_imp_df.sort_values('rf_importance', ascending=False)

# 置换重要性
# perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
# perm_imp_df = pd.DataFrame({'feature': X.columns, 'perm_importance': np.abs(perm.importances_mean)})
# perm_imp_df = perm_imp_df.sort_values('perm_importance', ascending=False)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_imp_df = pd.DataFrame({'feature': X.columns, 'xgb_importance': np.abs(xgb_model.feature_importances_)})
xgb_imp_df = xgb_imp_df.sort_values('xgb_importance', ascending=False)

# 线性回归系数（对连续特征更有意义，但计算全部特征系数）
linear_model = LinearRegression()
linear_model.fit(X.fillna(0), y.fillna(y.mean()))
linear_coefficients = linear_model.coef_
linear_df = pd.DataFrame({'feature': X.columns, 'linear_coefficient': np.abs(linear_coefficients)})

# 8) 互相关或其他：Spearman 已在上面对连续变量计算，若需对所有可用列使用 Spearman 可另算（这里我们保留上面计算）

# 合并所有结果到一个 DataFrame（以 feature 为键，外连接）
from functools import reduce
features_df = pd.DataFrame({'feature': X.columns})

# 更新：移除仅针对二元/类别的统计表（pointbiserial_df, ttest_df, mannu_df, cohend_df, chi2_df）
dfs_to_merge = [
    mi_df,  spearman_df, 
    f_df, cov_df,
    rf_imp_df,  linear_df
]
combined_df = reduce(lambda left, right: pd.merge(left, right, on='feature', how='outer'), dfs_to_merge)
# 合并表对所有数值列取绝对值
num_cols = combined_df.select_dtypes(include=['number']).columns
combined_df[num_cols] = combined_df[num_cols].abs()

# 新增：为 TOPSIS 准备决策矩阵并计算得分与排名


# 2) 构建决策矩阵并进行向量归一化（TOPSIS 标准化）
dec_mat = combined_df[num_cols].values.astype(float)
col_min = dec_mat.min(axis=0)
col_max = dec_mat.max(axis=0)
range_ = col_max - col_min
range_[range_ == 0] = 1.0
r = (dec_mat - col_min) / range_  # 矩阵的列向量归一化

# 3) 确定收益/成本指标：默认所有指标为收益，pvalue 类指标视为成本（值越小越好）
num_col_names = list(num_cols)
is_benefit = np.array([
    not ('pvalue' in c.lower() or 'p_val' in c.lower() or c.lower().endswith('_p') or 'pvalue' == c.lower())
    for c in num_col_names
])

# 4) 权重（使用熵值法自动计算）
# 使用 dec_mat（决策矩阵，已为非负数）并根据收益/成本做方向调整后计算熵权
m, n = dec_mat.shape  # m: 备选方案数(行), n: 指标数(列)
eps = 1e-24

# 1) 对成本指标转换为收益方向：x' = max(x) - x
dec_for_entropy = dec_mat.copy()
for j, benefit in enumerate(is_benefit):
    if not benefit:
        dec_for_entropy[:, j] =1 / dec_for_entropy[:, j]

# 2) 确保非负
dec_for_entropy[dec_for_entropy < 0] = 0.0

# 3) 列归一化得到 pij（避免列和为0）
col_sums = dec_for_entropy.sum(axis=0)
col_sums = np.where(col_sums == 0, eps, col_sums)
pij = dec_for_entropy / col_sums

# 4) 计算信息熵 ej
k = 1.0 / np.log(m) if m > 1 else 0.0
# 当 pij==0 时，pij * ln(pij) 视为 0
pj_log = np.where(pij > 0, pij * np.log(pij), 0.0)
entropy = -k * pj_log.sum(axis=0)

# 5) 差异度 dj 与权重 wj
diversity = 1 - entropy
# 防止所有差异度为0（退化情况），回退到等权
if np.allclose(diversity, 0):
    weights = np.ones(n) / n
else:
    weights = diversity / (diversity.sum() + eps)

# 5) 加权归一化矩阵（保持原来的 r 为列向量归一化结果）
v = r * weights

# 6) 计算正理想解和负理想解（根据收益/成本区分）
ideal_best = np.where(is_benefit, v.max(axis=0), v.min(axis=0))
ideal_worst = np.where(is_benefit, v.min(axis=0), v.max(axis=0))

# 7) 计算与理想解的距离并得到 TOPSIS 得分（越大越好）
dist_pos = np.sqrt(((v - ideal_best) ** 2).sum(axis=1))
dist_neg = np.sqrt(((v - ideal_worst) ** 2).sum(axis=1))
# 防止除以 0
den = dist_pos + dist_neg
den[den == 0] = 1.0
topsis_score = dist_neg / den

combined_df['topsis_score'] = topsis_score
rank_series = combined_df['topsis_score'].rank(ascending=False, method='min')
combined_df['topsis_rank'] = rank_series.where(rank_series.notna(), pd.NA).astype('Int64')

# 保存结果：完整指标表与仅排名表
# combined_df.to_csv("汇总结果/feature_stats_combined.csv", index=False)
# combined_df[['feature','topsis_score','topsis_rank']].to_csv("汇总结果/feature_ranking_topsis.csv", index=False)
os.makedirs("汇总结果", exist_ok=True)
combined_df.to_csv("汇总结果/feature_stats_combined.csv", index=False)
combined_df[['feature','topsis_score','topsis_rank']].to_csv("汇总结果/feature_ranking_topsis.csv", index=False)

print("已生成合并文件: 汇总结果/feature_stats_combined.csv")
print("已生成 TOPSIS 排名文件: 汇总结果/feature_ranking_topsis.csv")



