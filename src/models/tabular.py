"""
Tabular baseline models: XGBoost, Random Forest, SVM.

All models are wrapped in a common interface so evaluate.py can treat them
identically. Each model is trained with class weights to handle imbalance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


def _sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights inversely proportional to class frequency."""
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)


def build_xgboost(n_classes: int = 3, random_state: int = 42) -> Pipeline:
    """XGBoost pipeline with median imputation fitted on train split only.

    XGBoost supports NaN natively (splits on available data), but we add an
    explicit SimpleImputer so the interface is consistent across all models and
    NaN handling is always scoped to the training fold.
    """
    if n_classes == 2:
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            objective="binary:logistic",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=n_classes,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb",     clf),
    ])


def build_random_forest(random_state: int = 42) -> Pipeline:
    """Random Forest pipeline with median imputation fitted on train split only."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf",      RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def build_svm() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svm",     SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ])


def fit_predict(model, X_train, y_train, X_test):
    """Fit model and return (y_pred, y_proba).

    All models are now sklearn Pipelines, so imputation is always fitted
    on X_train only — no leakage to X_test.
    XGBoost gets per-sample class weights; RF and SVM use class_weight="balanced".
    """
    if isinstance(model, Pipeline) and "xgb" in model.named_steps:
        sw = _sample_weights(y_train)
        model.fit(X_train, y_train, xgb__sample_weight=sw)
    else:
        model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return y_pred, y_proba


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Return a sorted feature importance DataFrame (XGBoost and RF only)."""
    try:
        if isinstance(model, Pipeline):
            if "xgb" in model.named_steps:
                imp = model.named_steps["xgb"].feature_importances_
            elif "rf" in model.named_steps:
                imp = model.named_steps["rf"].feature_importances_
            else:
                return pd.DataFrame()   # SVM Pipeline — no direct importances
        elif isinstance(model, XGBClassifier):
            imp = model.feature_importances_
        elif isinstance(model, RandomForestClassifier):
            imp = model.feature_importances_
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)