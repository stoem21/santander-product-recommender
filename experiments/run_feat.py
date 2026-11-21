"""Feature / training-recipe experiments with LightGBM multiclass, same split+metric as everything else.
Variants (comma-combinable, e.g. bundle+income):
  base     : existing 166 features
  bundle   : payroll/pensions bundle flags (count held, exactly-one-held)
  income   : household_income_log minus the province median (median from that fold's TRAIN rows only)
  recency  : per product, months since its flag last changed inside the 5-month window (1..4, 5 = no change seen)
  early8   : drop training target months 1..7 (profile columns frozen in those source months)
  weight1k : weight each expanded row by 1/n_new
Results appended to analysis_output/models/feat_experiments.csv (existing variant x fold pairs are skipped).
Usage: python -m experiments.run_feat --variants base,bundle --val-months 13 14 [--rounds 100]"""
import argparse, time, os
import numpy as np, pandas as pd, lightgbm as lgb
from santander.train_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--variants", required=True); ap.add_argument("--val-months", type=int, nargs="+", default=[13, 14, 15, 16])
ap.add_argument("--rounds", type=int, default=100); ap.add_argument("--budget", type=float, default=150)
a = ap.parse_args()
T0 = time.time()
OUT = os.path.join(HERE, "analysis_output", "models", "feat_experiments.csv")
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["variant", "val_month"])
df_all = load_train()

def derive(d, prov_med, feats_wanted):
    """returns dict of new columns for dataframe d"""
    new = {}
    if "bundle" in feats_wanted:
        pr = d["payroll_lag1"].to_numpy("float32", na_value=np.nan); pe = d["pensions_lag1"].to_numpy("float32", na_value=np.nan)
        new["bundle_held_n"] = np.nan_to_num(pr == 1, nan=0).astype("float32") + np.nan_to_num(pe == 1, nan=0).astype("float32")
        new["bundle_exactly_one"] = (new["bundle_held_n"] == 1).astype("float32")
    if "income" in feats_wanted:
        inc = d["household_income_log"].to_numpy("float32", na_value=np.nan); prv = d["province_code"].to_numpy("float32", na_value=-1)
        med = np.array([prov_med.get(int(p), np.nan) for p in prv], dtype="float32")
        new["income_vs_province"] = inc - med
    if "recency" in feats_wanted:
        L = np.stack([d[[f"{p}_lag{k}" for p in PRODUCT_COLUMNS]].to_numpy("float32", na_value=np.nan) for k in range(1, 6)], axis=2)
        since = np.full(L.shape[:2], 5.0, dtype="float32")
        for k in range(3, -1, -1):                     # k=0 compares lag1 vs lag2 (most recent change)
            ch = (L[:, :, k] != L[:, :, k + 1]) & ~np.isnan(L[:, :, k]) & ~np.isnan(L[:, :, k + 1])
            since = np.where(ch, k + 1, since)
        for j, p in enumerate(PRODUCT_COLUMNS): new[f"since_change_{p}"] = since[:, j]
    return new

for vname in a.variants.split(","):
    parts = set(vname.split("+")); parts.discard("base")
    for V in a.val_months:
        if ((done.variant == vname) & (done.val_month == V)).any(): continue
        if time.time() - T0 > a.budget: print("time budget reached; re-run to resume"); raise SystemExit
        s0 = time.time()
        tr = df_all[df_all.target_month_idx < V]
        if "early8" in parts: tr = tr[tr.target_month_idx >= 8]
        tr = tr.reset_index(drop=True); va = df_all[df_all.target_month_idx == V].reset_index(drop=True)
        prov_med = tr.groupby("province_code")["household_income_log"].median().to_dict()
        ex_tr, ex_va = derive(tr, prov_med, parts), derive(va, prov_med, parts)
        feats = FEATURES + list(ex_tr)
        Xtr = np.concatenate([tr[FEATURES].to_numpy("float32", na_value=np.nan)] + [v[:, None] for v in ex_tr.values()], axis=1)
        Xva = np.concatenate([va[FEATURES].to_numpy("float32", na_value=np.nan)] + [v[:, None] for v in ex_va.values()], axis=1)
        Y = tr[LABELS].to_numpy("int8", na_value=0); ri, ci = np.nonzero(Y); nn_ = Y.sum(1)[ri]
        w = (1.0 / nn_).astype("float32") if "weight1k" in parts else None
        cat = [feats.index(c) for c in CAT_COLS]
        params = dict(objective="multiclass", num_class=NP, learning_rate=0.1, num_leaves=31, min_child_samples=100, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=4, verbose=-1, seed=1)
        b = lgb.train(params, lgb.Dataset(Xtr[ri], ci, weight=w, categorical_feature=cat), num_boost_round=a.rounds)
        act, own = actual_and_owned(va); m, r = evaluate(b.predict(Xva), own, act)
        row = dict(variant=vname, val_month=V, MAP7=m, recall7=r, n_features=len(feats), train_rows=len(ri), seconds=time.time() - s0)
        pd.concat([done, pd.DataFrame([row])], ignore_index=True).to_csv(OUT, index=False); done = pd.read_csv(OUT)
        print(f"{vname:22s} val{V}: MAP@7 {m:.4f}  ({len(feats)} feats, {time.time()-s0:.0f}s)", flush=True)
