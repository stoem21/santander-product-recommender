"""Analysis functions for story.ipynb.

They compute the EDA, baseline and error-analysis tables live from the processed Parquet files and the saved
validation scores. The functions are pure: nothing is written to disk.

These tables were first computed by exploration scripts that read the raw train_ver2.csv (not part of this
repository). The functions below recompute them from the Parquet files; their results were checked against the
earlier ones and match (up to rounding).

Typical use (from the project folder):
    from santander.story_analysis import *
    panel = load_panel()                      # 13.6M rows, product flags only; about 20 s and about 0.5 GB
    customers_per_month(panel); ownership_rate_by_month(panel); target_summary(panel) ...
"""
import os
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from santander.column_map import PRODUCT_COLUMNS
from santander.train_common import evaluate, top_k     # MAP@7 and top-k, shared with every model

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
PROC = os.path.join(HERE, "processed")
TRAIN = os.path.join(PROC, "train_clean.parquet")
MODELS = os.path.join(HERE, "analysis_output", "models")
K = 7
NP = len(PRODUCT_COLUMNS)


def month_label(i):
    """0 -> '2015-01', 16 -> '2016-05'."""
    i = int(i)
    return f"{2015 + i // 12}-{i % 12 + 1:02d}"


# ------------------------------------------------------------------------------------------------ panel
def load_panel(extra=(), path=TRAIN):
    """All customer-month rows, sorted by (customer_id, month_idx).

    Product flags are int8, with -1 where the flag is missing (missing never counts as held or as an add).
    `extra` names further columns of train_clean.parquet to load, e.g. ("tenure_months",).
    Adds helper columns: first_row / last_row (first / last row of a customer) and has_prev (the same customer is
    present in the previous calendar month, i.e. the row is eligible for the target).
    """
    tb = pq.read_table(path, columns=["customer_id", "month_idx"] + PRODUCT_COLUMNS + list(extra))
    df = pd.DataFrame({"customer_id": tb.column("customer_id").to_numpy(),
                       "month_idx": tb.column("month_idx").to_numpy()})
    for c in PRODUCT_COLUMNS:
        df[c] = pc.fill_null(tb.column(c), -1).to_numpy().astype("int8")
    for c in extra:
        df[c] = tb.column(c).to_pandas()
    df = df.sort_values(["customer_id", "month_idx"], kind="stable").reset_index(drop=True)
    ids, m = df.customer_id.to_numpy(), df.month_idx.to_numpy()
    same = ids[1:] == ids[:-1]
    df["first_row"] = np.r_[True, ~same]
    df["last_row"] = np.r_[~same, True]
    df["has_prev"] = np.r_[False, same & (m[1:] == m[:-1] + 1)]
    return df


def _products(panel):
    return panel[PRODUCT_COLUMNS].to_numpy()


# ------------------------------------------------------------------------------------------------ section 2: data
def customers_per_month(panel):
    """Rows per month, IDs seen for the first time and IDs seen for the last time."""
    m, last_m = panel.month_idx.to_numpy(), int(panel.month_idx.max())
    n = last_m + 1
    out = pd.DataFrame({
        "customers": np.bincount(m, minlength=n),
        "first_seen_this_month": np.bincount(m[panel.first_row.to_numpy()], minlength=n),
        "last_seen_this_month(left_after)": np.bincount(m[panel.last_row.to_numpy() & (m < last_m)], minlength=n)},
        index=[month_label(i) for i in range(n)])
    out["net_change"] = out.customers.diff()
    return out


def ownership_rate_by_month(panel):
    """% of customers holding each product, by month (products x months).
    A missing flag counts as not held."""
    m, P = panel.month_idx.to_numpy(), _products(panel)
    cnt = np.bincount(m)
    own = pd.DataFrame({c: np.bincount(m, weights=(P[:, j] == 1), minlength=len(cnt)) / cnt * 100
                        for j, c in enumerate(PRODUCT_COLUMNS)}, index=[month_label(i) for i in range(len(cnt))])
    return own.T


def first_month_tenure(path=TRAIN):
    """For each ID's first month in the data: how many IDs, % with tenure <= 1 month, median tenure.
    Shows that 'first month in the data' is not the same as 'new to the bank'."""
    f = (pd.read_parquet(path, columns=["customer_id", "month_idx", "tenure_months"])
         .sort_values("month_idx", kind="stable").drop_duplicates("customer_id"))
    g = f.groupby("month_idx").tenure_months
    out = pd.DataFrame({"ids first seen": g.size(), "tenure <= 1 month %": g.apply(lambda s: (s <= 1).mean() * 100),
                        "median tenure (months)": g.median()})
    out.index = [month_label(i) for i in out.index]
    return out


# ------------------------------------------------------------------------------------------------ section 3: data quirks
def column_change_by_month(columns=None, path=TRAIN):
    """% of customers whose value changed vs their previous month (columns x months 2015-02..2016-05).
    Only customers present in both months count. Missing -> missing is 'no change'.
    The old CSV was made from an earlier version of the Parquet file, so its column names can differ
    (for example 'seniority_months' there, 'seniority_months_raw' here)."""
    names = pq.ParquetFile(path).schema_arrow.names
    cols = list(columns) if columns else [c for c in names if c not in ("customer_id", "snapshot_date", "month_idx")]
    read = lambda t: pq.read_table(path, columns=["customer_id"] + cols, filters=[("month_idx", "=", t)]).to_pandas()
    last_t = int(pq.read_table(path, columns=["month_idx"]).column(0).to_numpy().max())
    res, prev = {}, read(0)
    for t in range(1, last_t + 1):
        cur = read(t)
        mm = cur.merge(prev, on="customer_id", suffixes=("", "_p"))
        res[month_label(t)] = {c: (~((mm[c] == mm[c + "_p"]).fillna(False).to_numpy(bool)
                                     | (mm[c].isna() & mm[c + "_p"].isna()).to_numpy())).mean() * 100 for c in cols}
        prev = cur
    return pd.DataFrame(res).loc[cols]


def seniority_vs_tenure(path=TRAIN):
    """Raw seniority vs tenure recomputed from first_contract_date.

    Returns a dict:
      share_equal_from_jul : share of rows from 2015-07 on where the two agree
      n_kept, share_kept   : customers present in all 17 months whose two values agree in 2015-07
      median               : median of both values per month for those customers (months x 2)
      steps_from_jul       : month-to-month step of the raw seniority from 2015-07 on (share of +1 / +2 / 0 / other)
    """
    t = pd.read_parquet(path, columns=["customer_id", "month_idx", "seniority_months_raw", "tenure_months"])
    ok = t.seniority_months_raw.notna() & t.tenure_months.notna()
    d = (t.seniority_months_raw - t.tenure_months)[ok]
    share_eq = float((d[t.month_idx[ok] >= 6] == 0).mean())
    full = t[t.groupby("customer_id").month_idx.transform("size") == 17]
    jul = full[full.month_idx == 6]
    keep = jul.customer_id[jul.seniority_months_raw == jul.tenure_months]
    med = full[full.customer_id.isin(keep)].groupby("month_idx")[["seniority_months_raw", "tenure_months"]].median()
    med.index = [month_label(i) for i in med.index]
    s = t.sort_values(["customer_id", "month_idx"], kind="stable")
    step = s.groupby("customer_id").seniority_months_raw.diff()
    gap1 = s.groupby("customer_id").month_idx.diff() == 1
    st = step[gap1 & (s.month_idx > 6) & step.notna()]
    steps = pd.Series({"+1": (st == 1).mean(), "+2": (st == 2).mean(), "0": (st == 0).mean(),
                       "other": (~st.isin([0, 1, 2])).mean()}) * 100
    return dict(share_equal_from_jul=share_eq, n_kept=len(keep), share_kept=len(keep) / jul.customer_id.nunique(),
                median=med, steps_from_jul=steps)


def is_new_6m_agreement(path=TRAIN, from_month=7):
    """Share of rows (months >= from_month) where the recomputed is_new_6m equals the raw new_customer flag."""
    t = pd.read_parquet(path, columns=["month_idx", "new_customer", "is_new_6m"])
    m = (t.month_idx >= from_month) & t.new_customer.notna() & t.is_new_6m.notna()
    return float((t.new_customer[m] == t.is_new_6m[m]).mean())


def column_transitions(columns, first_t=7, top=30, path=TRAIN):
    """Most common from -> to changes per column, same customer, month t-1 -> t, for t >= first_t (2015-08).
    Missing values are shown as '<null>'. Returns column, from, to, count."""
    as_str = lambda s: s.astype("string").fillna("<null>")
    read = lambda t: pq.read_table(path, columns=["customer_id"] + list(columns),
                                   filters=[("month_idx", "=", t)]).to_pandas()
    last_t = int(pq.read_table(path, columns=["month_idx"]).column(0).to_numpy().max())
    trans = {c: Counter() for c in columns}
    prev = read(first_t - 1)
    for t in range(first_t, last_t + 1):
        cur = read(t)
        mm = cur.merge(prev, on="customer_id", suffixes=("", "_p"))
        for c in columns:
            a, b = as_str(mm[c + "_p"]), as_str(mm[c])
            ch = (a != b).to_numpy()
            trans[c].update(dict(pd.DataFrame({"from": a[ch].to_numpy(), "to": b[ch].to_numpy()}).value_counts().items()))
        prev = cur
    rows = []
    for c in columns:
        s = pd.Series(trans[c], dtype="float64").sort_values(ascending=False).head(top)
        rows += [{"column": c, "from": f, "to": to, "count": int(v)} for (f, to), v in s.items()]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ section 4: target
def _adds(panel):
    """Boolean (rows x products): flag 0 at t-1 and 1 at t, eligible rows only (missing never counts)."""
    P = _products(panel)
    a = np.zeros(P.shape, bool)
    a[1:] = (P[1:] == 1) & (P[:-1] == 0)
    a &= panel.has_prev.to_numpy()[:, None]
    return a


def _ever_held_before(panel):
    """Boolean (rows x products): the customer held the product in some earlier row (re-add if added now)."""
    P = _products(panel)
    ids = panel.customer_id.to_numpy()
    starts = np.flatnonzero(panel.first_row.to_numpy())
    gstart = np.repeat(starts, np.diff(np.r_[starts, len(ids)]))
    out = np.zeros(P.shape, bool)
    for j in range(NP):
        held = (P[:, j] == 1).astype(np.int32)
        cs = np.cumsum(held) - held
        out[:, j] = (cs - cs[gstart]) > 0
    return out


def target_summary(panel):
    """Target by month and by product (same definitions as pipeline/make_target_table.py).
    Returns (monthly, by_product)."""
    m = panel.month_idx.to_numpy()
    A = _adds(panel)
    E = _ever_held_before(panel)
    n = int(m.max()) + 1
    n_add = A.sum(1)
    adds_m = np.stack([np.bincount(m[A[:, j]], minlength=n) for j in range(NP)], 1)
    readd_m = np.stack([np.bincount(m[A[:, j] & E[:, j]], minlength=n) for j in range(NP)], 1)
    monthly = pd.DataFrame({
        "eligible_customers": np.bincount(m[panel.has_prev.to_numpy()], minlength=n),
        "customers_with_add": np.bincount(m[n_add > 0], minlength=n),
        "total_adds": adds_m.sum(1), "re_adds": readd_m.sum(1)},
        index=[month_label(i) for i in range(n)]).iloc[1:]
    monthly["share_customers_with_add_%"] = monthly.customers_with_add / monthly.eligible_customers * 100
    monthly["adds_per_adder"] = monthly.total_adds / monthly.customers_with_add
    monthly["re_add_share_%"] = monthly.re_adds / monthly.total_adds * 100
    byprod = pd.DataFrame({"adds": adds_m.sum(0), "re_adds": readd_m.sum(0)}, index=PRODUCT_COLUMNS)
    byprod["share_of_all_adds_%"] = byprod.adds / byprod.adds.sum() * 100
    byprod["re_add_share_%"] = byprod.re_adds / byprod.adds.clip(lower=1) * 100
    byprod = byprod.sort_values("adds", ascending=False)
    byprod["cumulative_%"] = byprod["share_of_all_adds_%"].cumsum()
    return monthly, byprod


def adds_mix_by_month(panel):
    """Share of each product in all adds of the month, in % (months 2015-02..2016-05 x products)."""
    m, A = panel.month_idx.to_numpy(), _adds(panel)
    n = int(m.max()) + 1
    adds = pd.DataFrame({c: np.bincount(m[A[:, j]], minlength=n) for j, c in enumerate(PRODUCT_COLUMNS)},
                        index=[month_label(i) for i in range(n)]).iloc[1:]
    return adds.div(adds.sum(1), axis=0) * 100


# ------------------------------------------------------------------------------------------------ section 5: baselines
def popularity_baselines(panel, eval_months=(13, 14, 15, 16), seed=0):
    """Popularity baselines, each month t scored using only months < t.

    Uses the project's target definition: a missing flag is never an add and never counts as held. The models are
    evaluated on the same definition.
    """
    m, P = panel.month_idx.to_numpy(), _products(panel)
    hp = panel.has_prev.to_numpy()
    A = _adds(panel)
    n = int(m.max()) + 1
    adds_m = np.stack([np.bincount(m[A[:, j]], minlength=n) for j in range(NP)], 1).astype(float)
    prev0 = np.zeros(P.shape, bool); prev0[1:] = P[:-1] == 0
    nonown_m = np.stack([np.bincount(m[hp & prev0[:, j]], minlength=n) for j in range(NP)], 1).astype(float)
    rng = np.random.default_rng(seed)
    rows = []
    for t in eval_months:
        idx = np.flatnonzero((m == t) & hp)
        owned, actual = P[idx - 1] == 1, A[idx]
        keep = actual.any(1)
        owned, actual = owned[keep], actual[keep]
        ne = int(keep.sum())
        pop = {"pop_all_time": adds_m[1:t].sum(0), "pop_last_month": adds_m[t - 1],
               "pop_last_3": adds_m[t - 3:t].sum(0), "pop_same_month_last_year": adds_m[t - 12],
               "add_rate_last_3": adds_m[t - 3:t].sum(0) / np.maximum(nonown_m[t - 3:t].sum(0), 1)}
        models = {"random": rng.random((ne, NP))}
        models.update({k: np.tile(v, (ne, 1)) for k, v in pop.items()})
        for name, sc in models.items():
            mp, rc = evaluate(sc, owned, actual)
            rows.append({"month": month_label(t), "model": name, "MAP@7": mp, "recall@7": rc,
                         "eval_customers": ne, "share_of_customers_with_add": ne / len(idx)})
    return pd.DataFrame(rows)


def association_rule(panel, eval_months=(13, 14, 15, 16)):
    """Item-association rule, the only baseline whose ranking depends on the customer (same rule as run_extra.py --mode assoc).

    For eval month t, it learns S[i, j] = P(add j | holds i at t-1) from the adder rows of months < t, where an adder
    row is a customer-month with at least one add. A customer's score for j is the mean of S[i, j] over the products i
    they hold. Customers holding nothing get the plain add frequency of each product.
    Returns the same columns as popularity_baselines(), with model = 'item association rule'.
    """
    m, P = panel.month_idx.to_numpy(), _products(panel)
    hp = panel.has_prev.to_numpy()
    A = _adds(panel)
    adder = hp & A.any(1)
    rows = []
    for t in eval_months:
        tr = np.flatnonzero(adder & (m < t))
        O, Y = (P[tr - 1] == 1).astype("float32"), A[tr].astype("float32")
        S = (O.T @ Y) / np.maximum(O.sum(0)[:, None], 1)
        pop = Y.mean(0)
        va = np.flatnonzero(adder & (m == t))
        own, act = P[va - 1] == 1, A[va]
        Ov = own.astype("float32"); n_own = Ov.sum(1, keepdims=True)
        scores = np.where(n_own > 0, (Ov @ S) / np.maximum(n_own, 1), pop)
        mp, rc = evaluate(scores, own, act)
        rows.append({"month": month_label(t), "model": "item association rule", "MAP@7": mp, "recall@7": rc,
                     "eval_customers": len(va), "share_of_customers_with_add": len(va) / int((hp & (m == t)).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ section 11: XAI
XAI = os.path.join(HERE, "analysis_output", "xai")


def _xai_sample(features, n):
    """Test customers of the saved SHAP sample (shap_sample_ids.csv; first n), their feature matrix and owned mask."""
    from santander.final_features import load
    ids = pd.read_csv(os.path.join(XAI, "shap_sample_ids.csv")).customer_id.to_numpy()[:n]
    te = load("test").set_index("customer_id").loc[ids].reset_index()
    X = te[features].to_numpy("float32", na_value=np.nan)
    owned = te[[f"{p}_lag1" for p in PRODUCT_COLUMNS]].to_numpy("float32", na_value=np.nan) == 1
    return X, owned


def _model_features(model_path):
    """Feature names of a saved model, from the final_lgbm_meta.json written next to it by final_train_predict.py."""
    import json
    return json.load(open(os.path.join(os.path.dirname(model_path), "final_lgbm_meta.json")))["features"]


def shap_group_shares(model_path=os.path.join(HERE, "models", "final_lgbm.txt"), n=5000):
    """Share of |SHAP| per feature group for each customer's #1 recommended product (TreeSHAP, LightGBM pred_contrib).

    Uses the first n customers of the saved SHAP sample; n=5000 is the same set as xai_compare.py / compare_groups.csv.
    The feature '<this product>_lag1' is its own group, 'eligibility': it is always 0 for a recommended product, so its
    large share says nothing about the customer. Grouping: santander.final_features.feature_group. Returns a Series (shares sum to 1).
    """
    import lightgbm as lgb
    from santander.final_features import feature_group
    b = lgb.Booster(model_file=model_path)
    feats = _model_features(model_path)                         # the model file stores no names (Column_0..)
    X, owned = _xai_sample(feats, n)
    P = b.predict(X)
    C = b.predict(X, pred_contrib=True).reshape(len(X), NP, len(feats) + 1)[:, :, :-1]
    top1 = top_k(P, owned, 1)[:, 0]

    A = np.abs(C[np.arange(len(X)), top1])                     # customers x features, for each customer's #1 product
    out = Counter()
    for j in np.unique(top1):
        rows = top1 == j
        for k, f in enumerate(feats):
            out[feature_group(f, PRODUCT_COLUMNS[j])] += A[rows, k].sum()
    s = pd.Series(out, dtype=float)
    return (s / s.sum()).sort_values(ascending=False)


def counterfactual_flips(model_path=os.path.join(HERE, "models", "v1_with_sensitive", "final_lgbm.txt"), n=8000):
    """Change ONE feature for every customer, predict again, and count how often the recommendations change.

    Default model: version 1, which still has sex / foreigner_index / residence_index. Uses the first n customers of the
    saved SHAP sample (n=8000 = the sample in xai_part2.py). Encodings: sex 0/1 = H/V, foreigner_index 0/1 = N/S.
    Returns a DataFrame: % of customers whose #1 product changes and whose top-7 set changes.
    """
    import lightgbm as lgb
    b = lgb.Booster(model_file=model_path)
    feats = _model_features(model_path)                         # the model file stores no names (Column_0..)
    X, owned = _xai_sample(feats, n)
    base = top_k(b.predict(X), owned)
    swap = lambda v: np.where(v == 0, 1, np.where(v == 1, 0, v))
    tests = [("sex", swap, "sex H <-> V"), ("foreigner_index", swap, "foreigner N <-> S"),
             ("age", lambda v: v + 10, "age + 10 years"), ("n_prior_capped", lambda v: np.full_like(v, 6), "n_prior_capped -> 6")]
    rows = []
    for col, fn, label in tests:
        if col not in feats:
            continue
        X2 = X.copy(); k = feats.index(col); X2[:, k] = fn(X2[:, k])
        t = top_k(b.predict(X2), owned)
        rows.append({"change (one feature only)": label, "#1 product changes %": (t[:, 0] != base[:, 0]).mean() * 100,
                     "top-7 set changes %": np.mean([set(a) != set(c) for a, c in zip(base, t)]) * 100})
    return pd.DataFrame(rows).set_index("change (one feature only)")


# ------------------------------------------------------------------------------------------------ section 5: noise
def bootstrap_noise(prefix="lgbm", folds=(13, 14, 15, 16), n_boot=1000, seed=0):
    """How much MAP@7 moves only because of WHICH customers happen to be in a validation month.

    For each fold: AP@7 per customer from the saved validation scores (analysis_output/models/{prefix}_val{v}_scores.npy),
    then the standard deviation of the mean over n_boot resamples of the customers (with replacement).
    Returns a Series of standard deviations per validation month.
    """
    from santander.train_common import load_train, actual_and_owned
    df = load_train(); rng = np.random.default_rng(seed); out = {}
    for v in folds:
        va = df[df.target_month_idx == v]
        act, own = actual_and_owned(va)
        top = top_k(np.load(os.path.join(MODELS, f"{prefix}_val{v}_scores.npy")), own)
        rel = np.take_along_axis(act, top, 1)
        ap = (np.cumsum(rel, 1) / np.arange(1, K + 1) * rel).sum(1) / np.minimum(act.sum(1), K)
        idx = rng.integers(0, len(ap), (n_boot, len(ap)))
        out[month_label(v)] = ap[idx].mean(1).std()
    return pd.Series(out, name="bootstrap sd of MAP@7")


# ------------------------------------------------------------------------------------------------ section 9: errors
def error_analysis(prefix="lgbm", folds=(13, 14, 15, 16)):
    """Error analysis of saved validation scores, pooled over the folds.

    Reads processed/features_train.parquet and analysis_output/models/{prefix}_val{v}_scores.npy, which are read-only.
    The default lgbm scores come from the earlier 166-feature LightGBM: 200 rounds for folds 13-15, 400 for fold 16.
    Returns a dict of tables: summary, by_num_added, by_product, first_vs_readd, by_group, calibration.
    """
    from santander.train_common import load_train, actual_and_owned
    df = load_train()
    S, A, O, meta = [], [], [], []
    mcols = ["target_month_idx", "segment", "age", "tenure_months", "is_new_6m", "n_prev_products", "n_prior_months",
             "n_adds_last5", "n_drops_last5"]
    for v in folds:
        va = df[df.target_month_idx == v].reset_index(drop=True)
        s = np.load(os.path.join(MODELS, f"{prefix}_val{v}_scores.npy"))
        assert s.shape == (len(va), NP), f"{prefix} fold {v}: scores {s.shape} vs {len(va)} rows"
        act, own = actual_and_owned(va)
        ev = va[[f"ever_held_{p}" for p in PRODUCT_COLUMNS]].to_numpy("float32", na_value=np.nan) == 1
        S.append(s); A.append(act); O.append(own); meta.append(va[mcols].copy())
        meta[-1]["_ever"] = list(ev)
    S, A, O = np.vstack(S), np.vstack(A), np.vstack(O)
    meta = pd.concat(meta, ignore_index=True)
    EV = np.vstack(meta.pop("_ever").to_list()); N = len(S)

    sc = np.where(O, -np.inf, S); order = np.argsort(-sc, axis=1, kind="stable"); top7 = order[:, :K]
    rel = np.take_along_axis(A, top7, 1); prec = np.cumsum(rel, 1) / np.arange(1, K + 1)
    n_add = A.sum(1); ap = (prec * rel).sum(1) / np.minimum(n_add, K)
    rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(1, NP + 1)[None, :].repeat(N, 0), 1)
    tot_short = (1 - ap).sum()
    out = {"summary": pd.Series({"customers": N, "MAP@7": ap.mean(), "recall@7": (rel.sum(1) / n_add).mean(),
                                 "top1_hit": rel[:, 0].mean(), "hit_in_top3": rel[:, :3].any(1).mean()})}

    kk = pd.Series(np.minimum(n_add, 3), name="k")
    g = pd.DataFrame({"ap": ap, "k": kk}).groupby("k").ap.agg(["size", "mean"])
    g["share_of_customers"] = g["size"] / N
    g["share_of_AP_shortfall"] = pd.Series(1 - ap).groupby(kk).sum() / tot_short
    out["by_num_added"] = g

    # every (customer, added product) event; first-time add = never held before
    r_, c_ = np.nonzero(A)
    ep = pd.DataFrame({"product": np.array(PRODUCT_COLUMNS)[c_], "rank": rank[r_, c_], "first_time": ~EV[r_, c_],
                       "single": n_add[r_] == 1})
    ep["rr"] = 1.0 / ep["rank"]; ep["hit1"] = ep["rank"] == 1; ep["hit7"] = ep["rank"] <= K
    sg = ep[ep.single]
    ps = sg.groupby("product").agg(adds_single=("rank", "size"),
                                   AP=("rr", lambda x: np.where(x >= 1 / K - 1e-12, x, 0).mean()),
                                   hit1=("hit1", "mean"), hit7=("hit7", "mean"), median_rank=("rank", "median"))
    ps = ps.join(ep.groupby("product").size().rename("adds_all"))
    ps["share_of_adds"] = ps["adds_all"] / len(ep)
    ps["share_of_AP_shortfall(single-add customers)"] = (
        sg.assign(loss=1 - np.where(sg["rank"] <= K, sg["rr"], 0)).groupby("product").loss.sum() / tot_short)
    ps["first_time_share"] = ep.groupby("product").first_time.mean()
    out["by_product"] = ps.sort_values("share_of_AP_shortfall(single-add customers)", ascending=False)
    out["first_vs_readd"] = sg.groupby("first_time").agg(events=("rank", "size"), hit1=("hit1", "mean"),
                                                         hit7=("hit7", "mean"), mean_rr=("rr", "mean"))

    bins = lambda s, e, l: pd.cut(s, e, labels=l, right=False)
    grp = {
        "segment": meta.segment.map({0: "01-TOP", 1: "02-PARTICULARES", 2: "03-UNIVERSITARIO"}).fillna("missing"),
        "age": bins(meta.age, [0, 25, 35, 45, 55, 65, 200], ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]),
        "tenure_months": bins(meta.tenure_months, [-1, 6, 12, 36, 84, 1000], ["0-5 (new)", "6-11", "12-35", "36-83", "84+"]),
        "n_prev_products": bins(meta.n_prev_products, [0, 1, 2, 3, 5, 8, 30], ["0", "1", "2", "3-4", "5-7", "8+"]),
        "history_months(n_prior)": bins(meta.n_prior_months, [0, 3, 6, 10, 20], ["1-2", "3-5", "6-9", "10+"]),
        "changes_last5(adds+drops)": bins(meta.n_adds_last5 + meta.n_drops_last5, [0, 1, 2, 4, 100], ["0", "1", "2-3", "4+"]),
    }
    rows = []
    for name, s in grp.items():
        t = pd.DataFrame({"g": s.astype(str), "ap": ap, "n_add": n_add, "l": 1 - ap}).groupby("g").agg(
            customers=("ap", "size"), MAP7=("ap", "mean"), mean_products_added=("n_add", "mean"), short=("l", "sum"))
        t["share"] = t.customers / N; t["share_of_AP_shortfall"] = t.pop("short") / tot_short
        t.insert(0, "dimension", name); rows.append(t.reset_index())
    out["by_group"] = pd.concat(rows, ignore_index=True)

    Pn = np.where(O, 0, S); Pn = Pn / np.maximum(Pn.sum(1, keepdims=True), 1e-12)
    mask = ~O; p_flat, y_flat = Pn[mask], A[mask].astype(float)
    q = pd.qcut(pd.Series(p_flat), [0, .5, .8, .9, .95, .98, .99, .995, 1], duplicates="drop")
    out["calibration"] = pd.DataFrame({"p": p_flat, "y": y_flat, "bin": q}).groupby("bin", observed=True).agg(
        pairs=("p", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    return out
