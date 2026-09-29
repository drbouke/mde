"""
Fold-local preprocessing pipeline for cross-validation.

Imputation (median) and clipping (99.9th percentile) are fitted exclusively
on each training fold and applied to the corresponding test fold, preventing
any cross-fold information leakage from preprocessing statistics.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from entropy_features import compute_mde


class MDEFeatures(BaseEstimator, TransformerMixin):
    """
    Builds MDE entropy features inside a fold. fit() takes the training portion as a
    DataFrame of raw flow statistics (NaN allowed), stores its column medians and the
    min-max bounds of the entropy columns; transform() imputes with the stored medians,
    computes the entropy features, and returns them alone (include_raw=False) or
    appended to the raw columns (include_raw=True), as a float array. The raw columns
    keep their NaN so that the downstream imputer handles them fold-locally as well.
    """

    def __init__(self, dataset_name, include_raw=False):
        self.dataset_name = dataset_name
        self.include_raw = include_raw

    def _frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X, dtype=float), columns=self.raw_columns_)

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float))
        self.raw_columns_ = list(X.columns)
        num = X.replace([np.inf, -np.inf], np.nan)
        self.medians_ = num.median(numeric_only=True)
        F = compute_mde(num.fillna(self.medians_), self.dataset_name)
        self.bounds_ = F.attrs["bounds"]
        self.mde_columns_ = list(F.columns)
        self.feature_names_out_ = (self.raw_columns_ if self.include_raw else []) + self.mde_columns_
        return self

    def transform(self, X):
        X = self._frame(X)
        num = X.replace([np.inf, -np.inf], np.nan)
        F = compute_mde(num.fillna(self.medians_), self.dataset_name, bounds=self.bounds_)
        F = F[self.mde_columns_]
        if self.include_raw:
            return np.column_stack([num[self.raw_columns_].to_numpy(dtype=float), F.to_numpy(dtype=float)])
        return F.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clips each feature to the upper_pct percentile fitted on training data."""

    def __init__(self, upper_pct=99.9):
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        self.clip_hi_ = np.nanpercentile(X, self.upper_pct, axis=0)
        return self

    def transform(self, X, y=None):
        return np.minimum(np.asarray(X, dtype=float), self.clip_hi_)


def make_lgb_pipeline(clf, mde=None):
    """Wrap a LightGBM classifier with fold-local imputation and clipping, preceded by
    fold-local MDE feature construction when an MDEFeatures transformer is given."""
    steps = ([("mde", mde)] if mde is not None else []) + [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", PercentileClipper()),
        ("clf", clf),
    ]
    return Pipeline(steps)


def make_rf_pipeline(clf, mde=None):
    """Wrap a Random Forest classifier with fold-local imputation and clipping, preceded by
    fold-local MDE feature construction when an MDEFeatures transformer is given."""
    steps = ([("mde", mde)] if mde is not None else []) + [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", PercentileClipper()),
        ("clf", clf),
    ]
    return Pipeline(steps)
