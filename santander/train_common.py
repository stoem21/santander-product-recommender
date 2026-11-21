"""Shared loading / expansion / MAP@7 evaluation for every model (same code => comparable numbers)."""
import json, os
import numpy as np, pandas as pd
from santander.column_map import PRODUCT_COLUMNS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
PROC = os.path.join(HERE, "processed")
K = 7
NP = len(PRODUCT_COLUMNS)
FC = json.load(open(os.path.join(PROC, "feature_columns.json")))
FEATURES = FC["features"]
LABELS = ["new_" + p for p in PRODUCT_COLUMNS]
CAT_COLS = [c for c in ["segment", "relation_type_month_start", "customer_type_month_start", "sex",
                        "residence_index", "foreigner_index", "deceased_index", "employee_index",
                        "join_channel", "province_code"] if c in FEATURES]
OWNED_COLS = [p + "_lag1" for p in PRODUCT_COLUMNS]


def load_train():
    df = pd.read_parquet(os.path.join(PROC, "features_train.parquet"))
    return df


def to_matrix(df):
    return df[FEATURES].to_numpy(dtype="float32", na_value=np.nan)


def expand_rows(df):
    """One row per (customer, added product). Returns X, y, w (w = 1/n_new), customer-row index."""
    Y = df[LABELS].to_numpy(dtype="int8", na_value=0)
    r, c = np.nonzero(Y)
    X = to_matrix(df)[r]
    n = Y.sum(axis=1)[r]
    return X, c.astype("int32"), (1.0 / n).astype("float32"), r


def actual_and_owned(df):
    actual = df[LABELS].to_numpy(dtype="int8", na_value=0).astype(bool)
    owned = (df[OWNED_COLS].to_numpy(dtype="float32", na_value=np.nan) == 1)
    return actual, owned


def top_k(scores, owned, k=K):
    """Column indices of the k highest-scoring products the customer does not hold (ties keep product order)."""
    return np.argsort(-np.where(owned, -np.inf, scores), axis=1, kind="stable")[:, :k]


def evaluate(scores, owned, actual):
    """MAP@7 (Kaggle style, over customers with >= 1 add) and recall@7. Every model and baseline uses this."""
    top = top_k(scores, owned)
    rel = np.take_along_axis(actual, top, axis=1)
    prec = np.cumsum(rel, axis=1) / np.arange(1, K + 1)
    n = actual.sum(axis=1)
    ap = (prec * rel).sum(axis=1) / np.minimum(n, K)
    recall = rel.sum(axis=1) / n
    return float(ap.mean()), float(recall.mean())


def split_by_month(df, val_month):
    tr = df[df.target_month_idx < val_month].reset_index(drop=True)
    va = df[df.target_month_idx == val_month].reset_index(drop=True)
    return tr, va
