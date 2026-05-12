"""
Fold-local preprocessing pipeline for scientifically correct cross-validation.

Imputation (median) and clipping (99.9th percentile) are fitted exclusively
on each training fold and applied to the corresponding test fold, preventing
any cross-fold information leakage from preprocessing statistics.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clips each feature to the upper_pct percentile fitted on training data."""

    def __init__(self, upper_pct=99.9):
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        self.clip_hi_ = np.nanpercentile(X, self.upper_pct, axis=0)
        return self

    def transform(self, X, y=None):
        return np.minimum(np.asarray(X, dtype=float), self.clip_hi_)


def make_lgb_pipeline(clf):
    """Wrap a LightGBM classifier with fold-local imputation and clipping."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", PercentileClipper()),
        ("clf", clf),
    ])


def make_rf_pipeline(clf):
    """Wrap a Random Forest classifier with fold-local imputation and clipping."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", PercentileClipper()),
        ("clf", clf),
    ])
