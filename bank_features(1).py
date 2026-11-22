import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

CAT_COLS = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
NUM_COLS = ["age", "balance", "day", "campaign", "pdays", "previous"]
RAW_COLS = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
            "contact", "day", "month", "campaign", "pdays", "previous", "poutcome"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


class BankCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, low=0.01, high=0.99, drop=()):
        self.low = low
        self.high = high
        self.drop = drop

    def fit(self, X, y=None):
        self.bounds_ = {c: tuple(X[c].quantile([self.low, self.high])) for c in ["age", "balance", "campaign"]}
        seen = X["pdays"] != -1
        self.medians_ = {c: float(X.loc[seen, c].median()) for c in ["pdays", "previous"]}
        self.caps_ = {}
        for c in ["pdays", "previous"]:
            v = X[c].where(seen, self.medians_[c])
            self.caps_[c] = float(v.quantile(self.high))
        return self

    def transform(self, X):
        d = pd.DataFrame(index=X.index)
        for c, (lo, hi) in self.bounds_.items():
            d[c] = X[c].astype(float).clip(lo, hi)
        d["day"] = X["day"].astype(float)
        seen = X["pdays"] != -1
        for c in ["pdays", "previous"]:
            d[c] = X[c].astype(float).where(seen, self.medians_[c]).clip(upper=self.caps_[c])
        for c in CAT_COLS:
            d[c] = X[c].astype(str).str.lower()
        return d.drop(columns=list(self.drop))


def build_pipeline(estimator, drop=()):
    num = [c for c in NUM_COLS if c not in drop]
    cat = [c for c in CAT_COLS if c not in drop]
    prep = ColumnTransformer([
        ("num", PowerTransformer(method="yeo-johnson", standardize=True), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat),
    ])
    return Pipeline([("clean", BankCleaner(drop=drop)), ("prep", prep), ("model", estimator)])
