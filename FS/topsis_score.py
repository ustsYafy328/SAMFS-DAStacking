import os
import sys
import argparse
import numpy as np
import pandas as pd

def infer_column_rules(cols):
	# 返回两个字典：abs_cols (需要取绝对值), cost_cols (越小越好)
	abs_keywords = ['pearson', 'spearman', 'covariance', 'covar', 't_stat', 't-stat', 'tstat']
	cost_keywords = ['pvalue', 'p_value', 'p-value', '_p', 'pval', 'p_val']
	abs_cols = set()
	cost_cols = set()
	for c in cols:
		cl = c.lower()
		if any(k in cl for k in abs_keywords):
			abs_cols.add(c)
		if any(k in cl for k in cost_keywords):
			cost_cols.add(c)
	# 根据常见名字补充
	# 这些为明确的“越大越好”列
	benefit_explicit = {'mi_score', 'chi2_score', 'f_score', 'rf_importance', 'xgb_importance'}
	for c in cols:
		if c in benefit_explicit:
			# ensure not marked as cost
			cost_cols.discard(c)
	# 返回
	return abs_cols, cost_cols

def minmax_normalize(df, cost_cols=set(), eps=1e-12):
	# df: DataFrame 数值矩阵，返回归一化矩阵与列方向的是否 cost 列集合
	df_norm = df.copy().astype(float)
	for col in df.columns:
		col_vals = df_norm[col].values.astype(float)
		minv = np.nanmin(col_vals)
		maxv = np.nanmax(col_vals)
		if np.isclose(maxv, minv):
			# 避免除0，若列恒定则置0
			df_norm[col] = 0.0
		else:
			if col in cost_cols:
				df_norm[col] = (maxv - col_vals) / (maxv - minv)
			else:
				df_norm[col] = (col_vals - minv) / (maxv - minv)
	# 若有 nan，填充为0
	df_norm = df_norm.fillna(0.0)
	return df_norm

def entropy_weights(df_norm, eps=1e-12):
	# df_norm: 非负且归一化到[0,1]的矩阵，行是方案，列是指标
	m, n = df_norm.shape
	# 计算 p_ij
	P = df_norm.values.astype(float)
	# 如果某列全为0，则将其设置为微小常数，避免除0
	col_sums = P.sum(axis=0)
	col_sums[col_sums == 0] = eps
	P = P / col_sums
	# 计算熵
	with np.errstate(divide='ignore', invalid='ignore'):
		logP = np.where(P > 0, np.log(P), 0.0)
	k = 1.0 / np.log(m) if m > 1 else 1.0
	e = -k * (P * logP).sum(axis=0)
	d = 1 - e
	# 若所有 d 都为0，避免除0，改为等权
	if np.allclose(d, 0):
		w = np.ones_like(d) / len(d)
	else:
		w = d / d.sum()
	# 返回 pandas Series 对应列名
	return pd.Series(w, index=df_norm.columns)

def topsis(df_norm, weights):
	# df_norm: 已归一化矩阵，weights: pandas Series 与列对齐
	V = df_norm.values * weights.values[np.newaxis, :]
	v_pos = V.max(axis=0)
	v_neg = V.min(axis=0)
	# 欧氏距离
	d_pos = np.sqrt(((V - v_pos) ** 2).sum(axis=1))
	d_neg = np.sqrt(((V - v_neg) ** 2).sum(axis=1))
	# 贴近度
	# 若分母为0，设为0以避免NaN
	den = d_pos + d_neg
	den[den == 0] = 1e-12
	C = d_neg / den
	return pd.Series(C, index=df_norm.index), pd.Series(d_pos, index=df_norm.index), pd.Series(d_neg, index=df_norm.index)

def run(input_path, output_dir=None, save_intermediate=True):
	if output_dir is None:
		output_dir = os.path.dirname(input_path) or '.'
	os.makedirs(output_dir, exist_ok=True)

	# 读取 CSV，默认第一列为索引（特征名）
	df = pd.read_csv(input_path, index_col=0)
	if df.shape[0] == 0:
		raise ValueError("输入文件没有数据: " + input_path)

	# 识别列规则
	abs_cols, cost_cols = infer_column_rules(df.columns)

	# 预处理：对需要取绝对值的列取绝对值
	df_proc = df.copy()
	for c in df_proc.columns:
		# 如果可能为非数值（如包含字符串），尝试转换
		df_proc[c] = pd.to_numeric(df_proc[c], errors='coerce')
		if c in abs_cols:
			df_proc[c] = df_proc[c].abs()
	# 缺失值用列均值填充
	df_proc = df_proc.fillna(df_proc.mean())

	# 归一化（min-max），cost 列反向
	df_norm = minmax_normalize(df_proc, cost_cols=cost_cols)

	# 计算熵权
	weights = entropy_weights(df_norm)

	# TOPSIS
	scores, dpos, dneg = topsis(df_norm, weights)

	# 汇总结果
	result = pd.DataFrame({
		'score': scores,
		'd_pos': dpos,
		'd_neg': dneg
	}, index=df_norm.index)
	result['rank'] = result['score'].rank(ascending=False, method='min').astype(int)
	result = result.sort_values('score', ascending=False)

	# 保存结果
	out_scores = os.path.join(output_dir, 'topsis_entropy_results.csv')
	out_weights = os.path.join(output_dir, 'topsis_entropy_weights.csv')
	out_norm = os.path.join(output_dir, 'topsis_normalized_matrix.csv')
	if save_intermediate:
		result.to_csv(out_scores, index=True)
		weights.rename('weight').to_frame().to_csv(out_weights)
		df_norm.to_csv(out_norm, index=True)
	# 简要打印前20
	print("TOPSIS 结果（按score降序，显示前20）：")
	print(result.head(20))
	print(f"\n已保存: {out_scores}, {out_weights}, {out_norm}")

	return result, weights, df_norm

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="基于熵值法的TOPSIS 综合评价")
	parser.add_argument("--input", "-i", default=os.path.join(os.path.dirname(__file__), "汇总结果/feature_stats_combined.csv"),
	                    help="输入 CSV 文件路径（第一列为特征名，作为索引）")
	parser.add_argument("--output", "-o", default="综合评价结果", help="输出目录，默认为输入文件所在目录")
	args = parser.parse_args()
	try:
		run(args.input, args.output)
	except Exception as e:
		print("运行出错:", e)
		sys.exit(1)