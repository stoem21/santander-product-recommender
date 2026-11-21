"""LightGBM LambdaRank: one group per customer, candidates = products NOT owned at t-1, label = added.
Row features = customer features + candidate product id (categorical) + that product's own lag1..5 flags and ever_held.
Trained on a random sample of customers (memory). Usage: python -m experiments.run_rank --val-month 16 --customers 60000"""
import argparse, time, os
import numpy as np, pandas as pd, lightgbm as lgb
from santander.train_common import *
ap = argparse.ArgumentParser()
ap.add_argument("--val-month", type=int, default=16); ap.add_argument("--customers", type=int, default=60000)
ap.add_argument("--rounds", type=int, default=150); ap.add_argument("--lr", type=float, default=0.1)
a = ap.parse_args()
t0 = time.time()
df = load_train(); tr, va = split_by_month(df, a.val_month)
tr = tr.iloc[np.random.default_rng(0).choice(len(tr), min(a.customers, len(tr)), replace=False)].reset_index(drop=True)

def build(d):
    X = to_matrix(d); act, own = actual_and_owned(d)
    n = len(d)
    lag = np.stack([d[[f"{p}_lag{k}" for p in PRODUCT_COLUMNS]].to_numpy("float32", na_value=np.nan) for k in range(1, 6)], axis=2)  # n,NP,5
    ever = d[[f"ever_held_{p}" for p in PRODUCT_COLUMNS]].to_numpy("float32", na_value=np.nan)
    ci, pj = np.nonzero(~own)                       # candidate rows (customer, product)
    Xr = np.concatenate([X[ci], pj[:, None].astype("float32"), lag[ci, pj], ever[ci, pj][:, None]], axis=1)
    y = act[ci, pj].astype("int8")
    groups = np.bincount(ci, minlength=n)
    return Xr, y, groups, ci, pj, act, own
Xr, y, g, *_ = build(tr)
names = FEATURES + ["cand_product"] + [f"cand_lag{k}" for k in range(1, 6)] + ["cand_ever_held"]
cat = [names.index(c) for c in CAT_COLS] + [names.index("cand_product")]
print(f"rank rows {len(y):,} for {len(tr):,} customers, positives {int(y.sum()):,}", flush=True)
params = dict(objective="lambdarank", learning_rate=a.lr, num_leaves=63, min_child_samples=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=4, verbose=-1, seed=1,
              lambdarank_truncation_level=7, eval_at=[7])
t1 = time.time()
bst = lgb.train(params, lgb.Dataset(Xr, y, group=g[g > 0], categorical_feature=cat), num_boost_round=a.rounds)
tt = time.time() - t1; print(f"train {tt:.0f}s ({tt/a.rounds:.2f}s/round)", flush=True)
Xv, yv, gv, civ, pjv, act_v, own_v = build(va)
res = []
for it in [50, 100, a.rounds]:
    sc = np.full((len(va), NP), -1e9, dtype="float32"); sc[civ, pjv] = bst.predict(Xv, num_iteration=it)
    m, r = evaluate(sc, own_v, act_v); res.append(dict(iter=it, val_MAP7=m, val_recall7=r)); print(f"iter {it}: MAP@7 {m:.4f} recall@7 {r:.4f}")
pd.DataFrame(res).assign(val_month=a.val_month, tag="lambdarank", customers=len(tr), train_seconds=tt).to_csv(
    os.path.join(HERE, "analysis_output", "models", f"lambdarank_val{a.val_month}.csv"), index=False)
print("done", f"{time.time()-t0:.0f}s")
