import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


class SAMFSSelector:
    """Stability-aware multi-view feature selection with entropy weights and TOPSIS."""

    def __init__(
        self,
        n_features: int = 17,
        stability_lambda: float = 0.1,
        n_bootstrap: int = 30,
        random_state: int = 43,
    ):
        self.n_features = n_features
        self.stability_lambda = stability_lambda
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.feature_scores_: pd.DataFrame | None = None
        self.selected_features_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y) -> "SAMFSSelector":
        X = X.copy()
        y = np.asarray(y)
        criteria = self._criteria_matrix(X, y)
        variances = self._bootstrap_criterion_variance(X, y)
        weights = self._entropy_weights(criteria, variances)
        topsis = self._topsis(criteria, weights)
        scores = pd.DataFrame({"feature": X.columns, "samfs_score": topsis})
        scores = scores.sort_values("samfs_score", ascending=False).reset_index(drop=True)
        self.feature_scores_ = scores
        self.selected_features_ = scores.head(self.n_features)["feature"].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("SAMFSSelector must be fitted before transform.")
        return X[self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _criteria_matrix(self, X: pd.DataFrame, y) -> np.ndarray:
        arr = X.to_numpy(dtype=float)
        scores = []

        spearman = []
        for i in range(arr.shape[1]):
            corr = spearmanr(arr[:, i], y, nan_policy="omit").correlation
            spearman.append(0.0 if np.isnan(corr) else abs(corr))
        scores.append(spearman)

        f_scores, _ = f_regression(arr, y)
        scores.append(np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0))

        scores.append(mutual_info_regression(arr, y, random_state=self.random_state))

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(arr, y)
        scores.append(rf.feature_importances_)

        lr = LinearRegression()
        scaler = MinMaxScaler()
        lr.fit(scaler.fit_transform(arr), y)
        scores.append(np.abs(lr.coef_))

        return np.asarray(scores, dtype=float).T

    def _bootstrap_criterion_variance(self, X: pd.DataFrame, y) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        values = []
        n = len(X)
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            values.append(self._criteria_matrix(X.iloc[idx], np.asarray(y)[idx]))
        return np.var(np.stack(values, axis=0), axis=(0, 1))

    def _entropy_weights(self, criteria: np.ndarray, variances: np.ndarray) -> np.ndarray:
        z = MinMaxScaler().fit_transform(criteria)
        z = z + 1e-12
        p = z / z.sum(axis=0, keepdims=True)
        entropy = -(p * np.log(p)).sum(axis=0) / np.log(z.shape[0])
        raw = 1.0 - entropy + self.stability_lambda * variances
        raw = np.maximum(raw, 1e-12)
        return raw / raw.sum()

    @staticmethod
    def _topsis(criteria: np.ndarray, weights: np.ndarray) -> np.ndarray:
        z = MinMaxScaler().fit_transform(criteria)
        weighted = z * weights
        positive = weighted.max(axis=0)
        negative = weighted.min(axis=0)
        d_pos = np.linalg.norm(weighted - positive, axis=1)
        d_neg = np.linalg.norm(weighted - negative, axis=1)
        return d_neg / (d_pos + d_neg + 1e-12)

