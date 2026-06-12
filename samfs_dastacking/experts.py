from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import clone
import numpy as np
import xgboost as xgb
from tabpfn import TabPFNRegressor


def build_experts(random_state: int = 43):
    """Heterogeneous expert committee used by DA-Stacking."""
    return [
        ("tabpfn", TabPFNRegressor()),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                max_features="sqrt",
                min_samples_leaf=4,
                bootstrap=True,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "xgb",
            xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.03,
                colsample_bytree=0.8,
                subsample=0.8,
                reg_alpha=0.5,
                reg_lambda=0.5,
                verbosity=0,
                random_state=random_state,
            ),
        ),
        (
            "et",
            ExtraTreesRegressor(
                n_estimators=200,
                max_depth=12,
                max_features="sqrt",
                min_samples_leaf=4,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]


def fit_experts_with_oof(X_train, y_train, X_val, X_test, random_state: int = 43):
    experts = build_experts(random_state)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    oof = np.zeros((X_train.shape[0], len(experts)), dtype=float)
    fitted = []

    for i, (name, estimator) in enumerate(experts):
        oof[:, i] = cross_val_predict(
            clone(estimator),
            X_train,
            y_train,
            cv=cv,
            method="predict",
            n_jobs=1,
        )
        model = clone(estimator)
        model.fit(X_train, y_train)
        fitted.append((name, model))

    train_full = np.column_stack([model.predict(X_train) for _, model in fitted])
    val_preds = np.column_stack([model.predict(X_val) for _, model in fitted])
    test_preds = np.column_stack([model.predict(X_test) for _, model in fitted])
    return fitted, oof, train_full, val_preds, test_preds

