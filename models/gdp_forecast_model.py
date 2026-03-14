"""
GDP Forecast Model.

Uses a Random Forest Regressor to predict future GDP growth rate from a
country's structural economic features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from config import (
    GDP_MODEL_MAX_DEPTH,
    GDP_MODEL_N_ESTIMATORS,
    RANDOM_SEED,
)

# Features consumed by this model
GDP_FEATURES: list[str] = [
    "population",
    "energy_production",
    "technology_index",
    "trade_balance",
    "energy_price",
    "stability_index",
]


class GDPForecastModel:
    """Random-Forest-based GDP growth rate forecaster.

    Usage
    -----
    >>> model = GDPForecastModel()
    >>> model.train(features_df, gdp_growth_series)
    >>> predictions = model.predict(new_features_df)
    """

    def __init__(self) -> None:
        self.model: RandomForestRegressor = RandomForestRegressor(
            n_estimators=GDP_MODEL_N_ESTIMATORS,
            max_depth=GDP_MODEL_MAX_DEPTH,
            random_state=RANDOM_SEED,
        )
        self.scaler: StandardScaler = StandardScaler()
        self._is_trained: bool = False
        self.feature_importances_: pd.Series | None = None
        self.train_mae_: float | None = None
        self.train_r2_: float | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> "GDPForecastModel":
        """Fit the model on features *X* and targets *y*.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain the columns listed in ``GDP_FEATURES``.
        y : pd.Series
            GDP growth rate targets (%).
        """
        X_sel = X[GDP_FEATURES].fillna(0)
        X_scaled = self.scaler.fit_transform(X_sel)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=RANDOM_SEED
        )
        self.model.fit(X_tr, y_tr)

        y_pred_val = self.model.predict(X_val)
        self.train_mae_ = float(mean_absolute_error(y_val, y_pred_val))
        self.train_r2_  = float(r2_score(y_val, y_pred_val))

        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=GDP_FEATURES
        ).sort_values(ascending=False)

        self._is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted GDP growth rates for *X*."""
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")
        X_sel = X[GDP_FEATURES].fillna(0)
        X_scaled = self.scaler.transform(X_sel)
        return self.model.predict(X_scaled)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper that appends predictions to *df* copy."""
        result = df.copy()
        result["predicted_gdp_growth"] = self.predict(df)
        return result
