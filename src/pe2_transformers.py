"""Shared fold-safe sound encoder for the PE2 notebooks and saved model.

Import from src.pe2_transformers before loading the final joblib pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SoundTertileEncoder(TransformerMixin, BaseEstimator):
    """Replace the full-training sound category with fold-training tertiles."""

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        assert "Avg_Sound" in X and "Sound_Level_Code" not in X
        assert X["Avg_Sound"].notna().all()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.thresholds_ = X["Avg_Sound"].quantile([1 / 3, 2 / 3]).to_numpy()
        assert np.isfinite(self.thresholds_).all()
        return self

    def transform(self, X):
        check_is_fitted(self, ["thresholds_", "feature_names_in_"])
        assert isinstance(X, pd.DataFrame)
        assert X.columns.tolist() == self.feature_names_in_.tolist()
        result = X.copy()
        low, high = self.thresholds_
        # Right-closed boundaries match PE1: <= low, <= high, otherwise high.
        result["Sound_Level_Code"] = np.select(
            [result["Avg_Sound"] <= low, result["Avg_Sound"] <= high],
            [0, 1], default=2,
        ).astype(int)
        return result
