import numpy as np
import pandas as pd

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def add_features(X):
    """Row-wise feature engineering. It uses only the row itself, so it cannot leak information."""
    X = X.copy()
    X["month_num"] = X["month"].str.lower().map(MONTHS)
    X["previously_contacted"] = (X["pdays"] != -1).astype(int)
    X["pdays_clean"] = X["pdays"].where(X["pdays"] != -1, 0)
    X["balance_log"] = np.sign(X["balance"]) * np.log1p(X["balance"].abs())
    return X.drop(columns=["month", "pdays", "balance"])
