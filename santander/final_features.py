"""Final feature set 'cap+B' minus sex/foreigner/residence (175 features): the 166 base features with n_prior_months capped at 6, plus raw profile lags
(activity_index / segment / relation_type at s-1..s-4). Same function for train and test."""
import re
import json, os, numpy as np, pandas as pd
from santander.train_common import FEATURES, CAT_COLS, PROC
CH = json.load(open(os.path.join(PROC, "profile_change_columns.json")))
EXCLUDED = ["sex", "foreigner_index", "residence_index"]          # dropped after XAI: no measurable MAP@7 cost (-0.0003, 4 folds x 2 seeds)
FINAL_FEATURES = [c if c != "n_prior_months" else "n_prior_capped" for c in FEATURES if c not in EXCLUDED] + CH["B"]
FINAL_CATS = [c for c in CAT_COLS if c not in EXCLUDED] + [c for c in CH["B"] if c.startswith(("segment_", "relation_type_"))]
PARAMS = dict(objective="multiclass", num_class=24, learning_rate=0.1, num_leaves=31, min_child_samples=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=4, verbose=-1)

def load(name):
    f = pd.read_parquet(os.path.join(PROC, f"features_{name}.parquet"))
    p = pd.read_parquet(os.path.join(PROC, f"profile_change_{name}.parquet"))
    assert (f.customer_id.values == p.customer_id.values).all() and (f.target_month_idx.values == p.target_month_idx.values).all()
    f = pd.concat([f, p[CH["B"]]], axis=1)
    f["n_prior_capped"] = np.minimum(f["n_prior_months"], 6)
    return f

def matrix(df):
    return df[FINAL_FEATURES].to_numpy("float32", na_value=np.nan)

def cat_idx():
    return [FINAL_FEATURES.index(c) for c in FINAL_CATS]


ELIGIBILITY = "THIS product not held now (eligibility)"


def feature_group(f, product, split_eligibility=True):
    """Group of feature f when explaining the score of `product` (used to sum SHAP values by group).

    '<product>_lag1' is always 0 for a recommended product (held products are never recommended), so it is its own
    group, ELIGIBILITY, when split_eligibility=True; otherwise it counts as history of this product.
    """
    if split_eligibility and f == f"{product}_lag1":
        return ELIGIBILITY
    m = re.match(r"(.+)_lag\d+$", f) or re.match(r"ever_held_(.+)$", f)
    if m:
        if m.group(1) != product:
            return "history of OTHER products"
        return "past history of THIS product" if split_eligibility else "history of THIS product"
    if f.endswith(tuple(f"_plag{k}" for k in range(2, 6))):
        return "profile in previous months (set B)"
    if f in ("n_prev_products", "n_adds_last5", "n_drops_last5"):
        return "product counts"
    if f in ("sex", "foreigner_index", "residence_index"):
        return "sex / foreigner / residence"
    if f in ("age", "month_of_year", "n_prior_capped"):
        return f
    return "other profile"
