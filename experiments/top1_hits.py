"""Plain-language results for the README: for customers who added a product in the validation month, how often is a
product they actually added the #1 recommendation (and within the top 3)?

Final model (175 features, LightGBM, same parameters, 100 rounds, seed 1) retrained per fold on all earlier months,
next to the two baselines, all scored the same way on the same customers:
  same month last year : products ranked by how often they were added in the same month one year earlier
  association rule     : P(add j | holds i), averaged over the products the customer holds (as run_extra.py)
Also, for the final model: the same top-1 hit split by first-time adds vs re-adds (the customer held that product in
some earlier month), counting customers who added exactly one product, as in the error analysis.
And "close calls": customers whose #1 and #2 predicted probabilities differ by less than 0.05. Would showing two
options help them? Compares how often the added product is #1 vs within the top 2, for close calls and the rest.
And "quiet" customers (no product added or dropped in the last 5 months): their share of customers and of all lost
score (1 - AP@7 summed over customers).
Results -> analysis_output/models/top1_hits.csv, top1_by_add_type.csv, close_calls.csv, quiet_customers.csv
(all in analysis_output/models/)
Usage:  python -m experiments.top1_hits
"""
import os, time
import numpy as np, pandas as pd, lightgbm as lgb
from santander.train_common import LABELS, HERE, actual_and_owned, evaluate, top_k
from santander.final_features import PARAMS, load, matrix, cat_idx

OUT = os.path.join(HERE, "analysis_output", "models", "top1_hits.csv")
OUT2 = os.path.join(HERE, "analysis_output", "models", "top1_by_add_type.csv")
OUT3 = os.path.join(HERE, "analysis_output", "models", "close_calls.csv")
OUT4 = os.path.join(HERE, "analysis_output", "models", "quiet_customers.csv")
T0 = time.time()
df = load("train"); Y = df[LABELS].to_numpy("int8", na_value=0); tm = df.target_month_idx.to_numpy()
rows, rows2, rows3, rows4 = [], [], [], []
from santander.column_map import PRODUCT_COLUMNS
for V in (13, 14, 15, 16):
    tr, va = np.flatnonzero(tm < V), np.flatnonzero(tm == V)
    ri, ci = np.nonzero(Y[tr]); ri = tr[ri]
    b = lgb.train({**PARAMS, "seed": 1}, lgb.Dataset(matrix(df.iloc[ri]), ci, categorical_feature=cat_idx()), num_boost_round=100)
    act, own = actual_and_owned(df.iloc[va])
    O, A = actual_and_owned(df.iloc[tr])[1].astype("float32"), Y[tr].astype("float32")
    S = (O.T @ A) / np.maximum(O.sum(0)[:, None], 1)                     # association rule, learned on months < V
    Ov = own.astype("float32"); n_own = Ov.sum(1, keepdims=True)
    scores = {"final model": b.predict(matrix(df.iloc[va])),
              "same month last year": np.tile(Y[tm == V - 12].sum(0).astype(float), (len(va), 1)),
              "association rule": np.where(n_own > 0, (Ov @ S) / np.maximum(n_own, 1), A.mean(0))}
    for name, sc in scores.items():
        top = top_k(sc, own); rel = np.take_along_axis(act, top, 1)
        rows.append(dict(month=V, method=name, customers=len(va), top1_hit=rel[:, 0].mean(),
                         top3_hit=rel[:, :3].any(1).mean(), MAP7=evaluate(sc, own, act)[0]))
        if name == "final model":                                         # first-time add vs re-add, single-add customers
            ever = df.iloc[va][[f"ever_held_{p}" for p in PRODUCT_COLUMNS]].to_numpy("float32", na_value=np.nan) == 1
            one = act.sum(1) == 1; j = act[one].argmax(1)
            first = ~ever[one][np.arange(one.sum()), j]; hit = top[one, 0] == j
            for kind, m in (("first-time add", first), ("re-add", ~first)):
                rows2.append(dict(month=V, add_type=kind, customers=int(m.sum()), top1_hit=hit[m].mean()))
            pr = np.take_along_axis(sc, top[:, :2], 1); close = (pr[:, 0] - pr[:, 1]) < 0.05
            for kind, m in (("close call (gap < 0.05)", close), ("clear #1", ~close)):
                rows3.append(dict(month=V, group=kind, customers=int(m.sum()), top1_hit=rel[m, 0].mean(),
                                  top2_hit=rel[m, :2].any(1).mean()))
            ap = (np.cumsum(rel, 1) / np.arange(1, 8) * rel).sum(1) / np.minimum(act.sum(1), 7)
            quiet = (df.iloc[va].n_adds_last5 + df.iloc[va].n_drops_last5).to_numpy() == 0
            for kind, m in (("quiet (no change in 5 months)", quiet), ("other", ~quiet)):
                rows4.append(dict(month=V, group=kind, customers=int(m.sum()), MAP7=ap[m].mean(), lost=(1 - ap[m]).sum()))
    print(f"fold {V} done ({time.time() - T0:.0f}s)", flush=True)
res = pd.DataFrame(rows); res.to_csv(OUT, index=False)
r2 = pd.DataFrame(rows2); r2.to_csv(OUT2, index=False)
r3 = pd.DataFrame(rows3); r3.to_csv(OUT3, index=False)
r4 = pd.DataFrame(rows4); r4.to_csv(OUT4, index=False)
print(res.groupby("method", sort=False)[["top1_hit", "top3_hit", "MAP7"]].mean().round(4).to_string())
w = r2.assign(h=r2.top1_hit * r2.customers).groupby("add_type")[["h", "customers"]].sum()
print("\nfinal model, single-add customers, top-1 hit:", (w.h / w.customers).round(4).to_dict(), w.customers.to_dict())
w3 = r3.assign(a=r3.top1_hit * r3.customers, b=r3.top2_hit * r3.customers).groupby("group")[["a", "b", "customers"]].sum()
print("\nclose calls:", pd.DataFrame({"share of customers": w3.customers / w3.customers.sum(), "top-1 hit": w3.a / w3.customers,
                                     "top-2 hit": w3.b / w3.customers}).round(4).to_string())
w4 = r4.assign(a=r4.MAP7 * r4.customers).groupby("group")[["a", "customers", "lost"]].sum()
print("\nquiet customers:", pd.DataFrame({"share of customers": w4.customers / w4.customers.sum(), "MAP7": w4.a / w4.customers,
                                         "share of lost score": w4.lost / w4.lost.sum()}).round(4).to_string())
