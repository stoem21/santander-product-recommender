#!/usr/bin/env python3
"""Step 2: build feature tables.  Everything is known at the END of month t-1.

Rows
  features_train.parquet : one row per (customer, month t) in which the customer ADDED >=1 product
                           (446,511 rows) + labels n_new and new_<product> x24
  features_test.parquet  : one row per test customer (month t = 2016-06 = idx 17)
Both come from the SAME functions, so train and test features are built identically.

Feature groups (all measured at month t-1 unless stated)
  products     24 flags x 5 lags  <product>_lag1 (= t-1) ... _lag5 (= t-5)
               1 = held, 0 = not held, NULL = unknown (missing month or missing flag)
  history      n_prev_products, n_adds_last5, n_drops_last5, n_prior_months,
               ever_held_<product> x24 (held in ANY month up to t-1: re-add signal)
  profile      age, tenure_months, is_new_6m, household_income_log, province_code,
               activity_index, primary_customer_index, demographics_stale, and codes for
               segment, relation/customer type, sex, residence/foreigner/deceased/employee,
               join_channel (top 15 + other; NULL = -1)
  calendar     month_of_year of the TARGET month t (test = 6)

Categorical codes use fixed maps saved to processed/category_maps.json (-1 = missing).

Run (from the repository root):
  python -m pipeline.make_features                  # 5 lags -> processed/features_{train,test}.parquet
  python -m pipeline.make_features --nlag 12        # 12 lags -> processed/features_{train,test}_lag12.parquet
                                                    #   (experiment in experiments/run_lagsets.py)
  --out DIR writes the tables to another folder (inputs are always read from processed/).
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from santander.column_map import PRODUCT_COLUMNS

ap = argparse.ArgumentParser()
ap.add_argument("--nlag", type=int, default=5, help="number of product-flag lags (5 = the main feature set)")
ap.add_argument("--out", default="processed")
A = ap.parse_args()
IN, OUT = "processed", A.out
SUF = "" if A.nlag == 5 else f"_lag{A.nlag}"         # file-name suffix of the non-default lag sets
os.makedirs(OUT, exist_ok=True)
CLEAN, TGT, TEST = f"{IN}/train_clean.parquet", f"{IN}/target.parquet", f"{IN}/test_clean.parquet"
NP, NLAG, T_TEST = len(PRODUCT_COLUMNS), A.nlag, 17
NEW = [f"new_{c}" for c in PRODUCT_COLUMNS]
CAT_COLS = ["segment", "relation_type_month_start", "customer_type_month_start", "sex",
            "residence_index", "foreigner_index", "deceased_index", "employee_index"]
PROFILE_COLS = (["customer_id", "age", "tenure_months", "is_new_6m", "household_income",
                 "province_code", "activity_index", "primary_customer_index",
                 "demographics_stale", "join_channel"] + CAT_COLS)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ category maps
def category_counts(col):
    tbl = pq.read_table(CLEAN, columns=[col], read_dictionary=[col])
    cnt = Counter()
    for ch in tbl.column(0).chunks:
        codes = ch.indices.fill_null(-1).to_numpy()
        vals = ch.dictionary.to_pylist()
        bc = np.bincount(codes[codes >= 0], minlength=len(vals))
        for v, n in zip(vals, bc):
            if n:
                cnt[v] += int(n)
    return cnt


def build_maps():
    maps = {c: {v: i for i, v in enumerate(sorted(category_counts(c)))} for c in CAT_COLS}
    top = [v for v, _ in category_counts("join_channel").most_common(15)]
    maps["join_channel"] = {v: i for i, v in enumerate(top)}          # all others -> 15
    json.dump(maps, open(f"{OUT}/category_maps.json", "w"), indent=1, ensure_ascii=False)
    return maps


def code(series, mapping, other=None):
    out = series.map(mapping)
    if other is not None:
        out = out.where(~(series.notna() & out.isna()), other)
    return out.fillna(-1).astype("int16")


def nullable(arr):
    """int8 array with -1 = unknown  ->  pandas nullable Int8 (unknown = NA)."""
    return pd.arrays.IntegerArray(np.where(arr < 0, 0, arr).astype("int8"), arr < 0)


# ------------------------------------------------------------------ load products
log("loading products")
t = pq.read_table(CLEAN, columns=["customer_id", "month_idx"] + PRODUCT_COLUMNS)
ids = t.column("customer_id").to_numpy().astype("int32")
m = t.column("month_idx").to_numpy().astype("int8")
P = np.empty((len(ids), NP), dtype="int8")
for j, c in enumerate(PRODUCT_COLUMNS):
    P[:, j] = t.column(c).fill_null(-1).to_numpy().astype("int8")      # -1 = flag unknown
del t
o = np.lexsort((m, ids))
ids, m, P = ids[o], m[o], P[o]
N = len(ids)
key = ids.astype("int64") * 32 + m                                      # sorted ascending
first = np.r_[True, ids[1:] != ids[:-1]]
starts = np.flatnonzero(first)
gstart = np.repeat(starts, np.diff(np.r_[starts, N]))
log(f"{N:,} rows, {len(starts):,} customers")
maps = build_maps()
log("category maps saved")


# ------------------------------------------------------------------ feature builders
def product_features(idx):
    f, L = {}, []
    for k in range(NLAG):
        j = np.maximum(idx - k, 0)
        valid = (idx - k >= 0) & (ids[j] == ids[idx]) & (m[j] == m[idx] - k)
        arr = np.where(valid[:, None], P[j], np.int8(-1)).astype("int8")
        L.append(arr)
        for p, c in enumerate(PRODUCT_COLUMNS):
            f[f"{c}_lag{k + 1}"] = nullable(arr[:, p])
    f["n_prev_products"] = (L[0] == 1).sum(axis=1).astype("int8")
    adds = np.zeros(len(idx), dtype="int8")
    drops = np.zeros(len(idx), dtype="int8")
    for k in range(4):                               # adds/drops always use the 5-month window
        adds += ((L[k] == 1) & (L[k + 1] == 0)).sum(axis=1).astype("int8")
        drops += ((L[k] == 0) & (L[k + 1] == 1)).sum(axis=1).astype("int8")
    f["n_adds_last5"], f["n_drops_last5"] = adds, drops
    f["n_prior_months"] = (idx - gstart[idx] + 1).astype("int8")
    for p, c in enumerate(PRODUCT_COLUMNS):                              # ever held up to t-1
        held = (P[:, p] == 1).astype("int32")
        cs = np.cumsum(held)
        base = cs[gstart] - held[gstart]
        f[f"ever_held_{c}"] = ((cs[idx] - base[idx]) > 0).astype("int8")
    return f


def profile_features(idx):
    s_arr, cid = m[idx], ids[idx]
    pieces = []
    for s in np.unique(s_arr):
        sel = np.flatnonzero(s_arr == s)
        prof = pq.read_table(CLEAN, columns=PROFILE_COLS,
                             filters=[("month_idx", "=", int(s))]).to_pandas()
        pieces.append(pd.DataFrame({"customer_id": cid[sel], "pos": sel})
                      .merge(prof, on="customer_id", how="left"))
    R = pd.concat(pieces).sort_values("pos").reset_index(drop=True)
    assert len(R) == len(idx)
    num = lambda c: pd.to_numeric(R[c], errors="coerce").astype("float32")
    f = {"age": num("age"), "tenure_months": num("tenure_months"), "is_new_6m": num("is_new_6m"),
         "household_income_log": np.log1p(num("household_income")).astype("float32"),
         "province_code": pd.to_numeric(R["province_code"], errors="coerce").fillna(-1).astype("int16"),
         "activity_index": num("activity_index"), "primary_customer_index": num("primary_customer_index"),
         "demographics_stale": R["demographics_stale"].astype("int8")}
    for c in CAT_COLS:
        f[c] = code(R[c], maps[c])
    f["join_channel"] = code(R["join_channel"], maps["join_channel"], other=15)
    return f


def build(idx, t_target, labels=None):
    meta = {"customer_id": ids[idx], "target_month_idx": np.full(len(idx), 0, dtype="int8") + t_target
            if np.isscalar(t_target) else t_target.astype("int8")}
    df = pd.DataFrame(meta)
    if labels is not None:
        for c in ["n_new"] + NEW:
            df[c] = labels[c].to_numpy().astype("int8")
    feats = {}
    feats.update(product_features(idx))
    feats.update(profile_features(idx))
    tt = df["target_month_idx"].to_numpy()
    feats["month_of_year"] = ((tt % 12) + 1).astype("int8")
    df = pd.concat([df, pd.DataFrame(feats)], axis=1)
    return df


def anchors(customer_ids, s_months):
    k = customer_ids.astype("int64") * 32 + s_months
    pos = np.searchsorted(key, k)
    assert (pos < N).all() and (key[pos] == k).all(), "anchor row (customer, t-1) not found"
    return pos


# ------------------------------------------------------------------ TRAIN (adder rows)
log("building train features")
tg = pq.read_table(TGT, columns=["customer_id", "month_idx", "n_new"] + NEW,
                   filters=[("n_new", ">", 0)]).to_pandas().sort_values(["month_idx", "customer_id"])
tg = tg.reset_index(drop=True)
idx_tr = anchors(tg["customer_id"].to_numpy(), tg["month_idx"].to_numpy().astype("int64") - 1)
train = build(idx_tr, tg["month_idx"].to_numpy(), labels=tg)
pq.write_table(pa.Table.from_pandas(train, preserve_index=False), f"{OUT}/features_train{SUF}.parquet",
               compression="zstd")
log(f"train: {train.shape}")
del train

# ------------------------------------------------------------------ TEST (all customers)
log("building test features")
te = pq.read_table(TEST, columns=["customer_id", "month_idx"]).to_pandas()
assert (te["month_idx"] == T_TEST).all()
idx_te = anchors(te["customer_id"].to_numpy(), np.full(len(te), T_TEST - 1, dtype="int64"))
test = build(idx_te, T_TEST)
pq.write_table(pa.Table.from_pandas(test, preserve_index=False), f"{OUT}/features_test{SUF}.parquet",
               compression="zstd")
log(f"test: {test.shape}")
del test

# ------------------------------------------------------------------ checks
log("checks")
tr = pq.read_table(f"{OUT}/features_train{SUF}.parquet").to_pandas()
ts = pq.read_table(f"{OUT}/features_test{SUF}.parquet").to_pandas()
feat_cols = [c for c in tr.columns if c not in ["customer_id", "target_month_idx", "n_new"] + NEW]
assert list(ts.columns) == ["customer_id", "target_month_idx"] + feat_cols
print(f"\nfeature columns: {len(feat_cols)}  | train rows {len(tr):,}  | test rows {len(ts):,}")
assert len(tr) == 446_511 and len(ts) == 929_615
assert not tr.duplicated(["customer_id", "target_month_idx"]).any()

bad = sum(int(((tr[f"new_{c}"] == 1) & (tr[f"{c}_lag1"] == 1)).sum()) for c in PRODUCT_COLUMNS)
print("rows where a label product was already held at t-1 (must be 0):", bad)
assert bad == 0
assert (tr["n_new"] == tr[NEW].sum(axis=1)).all()
assert set(ts["customer_id"]) == set(pq.read_table(TEST, columns=["customer_id"]).column(0).to_pylist())
print("test: target month", ts["target_month_idx"].unique(), "| month_of_year", ts["month_of_year"].unique(),
      "| unknown lag1 flags:", int(ts[[f"{c}_lag1" for c in PRODUCT_COLUMNS]].isna().sum().sum()))

print("\nSpot check customer 16713 (credit_card adds expected at t = 2015-04, 2015-08, 2016-04):")
x = tr[(tr.customer_id == 16713) & (tr.new_credit_card == 1)]
print(x[["target_month_idx", "new_credit_card", "credit_card_lag1", "credit_card_lag2",
         "ever_held_credit_card", "n_prior_months", "n_prev_products"]].to_string(index=False))

print("\nShare of NULL by lag (train adders):",
      {f"lag{k}": round(float(tr[[f'{c}_lag{k}' for c in PRODUCT_COLUMNS]].isna().mean().mean() * 100), 1)
       for k in range(1, NLAG + 1)})
print("Share of NULL in profile features (train / test, %):")
pf = ["age", "tenure_months", "household_income_log", "activity_index"]
print(pd.DataFrame({"train": tr[pf].isna().mean() * 100, "test": ts[pf].isna().mean() * 100}).round(2).to_string())
for f_ in (f"features_train{SUF}", f"features_test{SUF}"):
    print(f"{f_}.parquet: {os.path.getsize(f'{OUT}/{f_}.parquet') / 1e6:.0f} MB")
json.dump({"features": feat_cols, "labels": ["n_new"] + NEW, "meta": ["customer_id", "target_month_idx"]},
          open(f"{OUT}/feature_columns{SUF}.json", "w"), indent=1)
log("done")
