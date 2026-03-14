"""
Inflation Predictor Model.

Gradient-Boosted regressor that forecasts future inflation given
energy price dynamics, trade balance, and GDP growth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from config import INFLATION_MODEL_N_ESTIMATORS, RANDOM_SEED

INFLATION_FEATURES: list[str] = [
    "energy_price",
    "trade_balance",
    "gdp_growth_rate",
    "technology_index",
    "stability_index",
]


class InflationPredictor:
    """Gradient-Boosted inflation-rate forecaster.

    Usage
    -----
    >>> predictor = InflationPredictor()
    >>> predictor.train(features_df, inflation_series)
    >>> predicted = predictor.predict(new_features_df)
    """

    def __init__(self) -> None:
        self.model: GradientBoostingRegressor = GradientBoostingRegressor(
            n_estimators=INFLATION_MODEL_N_ESTIMATORS,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
        )
        self.scaler: StandardScaler = StandardScaler()
        self._is_trained: bool = False
        self.feature_importances_: pd.Series | None = None
        self.train_mae_: float | None = None
        self.train_r2_: float | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> "InflationPredictor":
        """Fit the model.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain the columns listed in ``INFLATION_FEATURES``.
        y : pd.Series
            Inflation rate targets (%).
        """
        X_sel = X[INFLATION_FEATURES].fillna(0)
        X_scaled = self.scaler.fit_transform(X_sel)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=RANDOM_SEED
        )
        self.model.fit(X_tr, y_tr)

        y_pred_val = self.model.predict(X_val)
        self.train_mae_ = float(mean_absolute_error(y_val, y_pred_val))
        self.train_r2_  = float(r2_score(y_val, y_pred_val))

        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=INFLATION_FEATURES
        ).sort_values(ascending=False)

        self._is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted inflation rates for *X*."""
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")
        X_sel = X[INFLATION_FEATURES].fillna(0)
        X_scaled = self.scaler.transform(X_sel)
        return self.model.predict(X_scaled).clip(0.1, 50.0)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper that appends predictions to *df* copy."""
        result = df.copy()
        result["predicted_inflation"] = self.predict(df)
        return result
