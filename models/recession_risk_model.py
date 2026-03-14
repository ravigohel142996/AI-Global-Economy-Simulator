"""
Recession Risk Model.

Random Forest Classifier that estimates the probability of a country
entering a recession in the next simulation round.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

from config import RECESSION_MODEL_N_ESTIMATORS, RECESSION_THRESHOLD, RANDOM_SEED

RECESSION_FEATURES: list[str] = [
    "inflation_rate",
    "gdp_growth_rate",
    "stability_index",
    "trade_balance",
    "energy_price",
    "technology_index",
]


class RecessionRiskModel:
    """Random-Forest recession probability estimator.

    Outputs a probability in [0, 1] where values above
    ``RECESSION_THRESHOLD`` indicate high recession risk.

    Usage
    -----
    >>> model = RecessionRiskModel()
    >>> model.train(features_df, recession_labels)
    >>> probs = model.predict_proba(new_features_df)
    """

    def __init__(self) -> None:
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=RECESSION_MODEL_N_ESTIMATORS,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
        self.scaler: StandardScaler = StandardScaler()
        self._is_trained: bool = False
        self.feature_importances_: pd.Series | None = None
        self.train_auc_: float | None = None

    # ------------------------------------------------------------------
    def train(self, X: pd.DataFrame, y: pd.Series) -> "RecessionRiskModel":
        """Fit the model.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain the columns in ``RECESSION_FEATURES``.
        y : pd.Series
            Binary labels (1 = recession risk, 0 = stable).  If a
            continuous risk score is supplied it is binarised at 0.5.
        """
        X_sel = X[RECESSION_FEATURES].fillna(0)
        X_scaled = self.scaler.fit_transform(X_sel)

        y_bin = (y >= 0.5).astype(int)

        # Guard against degenerate single-class case (may happen with tiny
        # datasets)
        if y_bin.nunique() < 2:
            y_bin = y_bin.copy()
            y_bin.iloc[0] = 1 - y_bin.iloc[0]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y_bin, test_size=0.2, random_state=RANDOM_SEED
        )
        self.model.fit(X_tr, y_tr)

        if len(np.unique(y_val)) > 1:
            proba_val = self.model.predict_proba(X_val)[:, 1]
            self.train_auc_ = float(roc_auc_score(y_val, proba_val))

        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=RECESSION_FEATURES
        ).sort_values(ascending=False)

        self._is_trained = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return recession probability for each row in *X*."""
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")
        X_sel = X[RECESSION_FEATURES].fillna(0)
        X_scaled = self.scaler.transform(X_sel)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper that appends probabilities to *df* copy."""
        result = df.copy()
        result["recession_probability"] = self.predict_proba(df)
        result["recession_risk_label"] = (
            result["recession_probability"] >= RECESSION_THRESHOLD
        ).map({True: "High Risk", False: "Low Risk"})
        return result
