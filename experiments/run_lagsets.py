"""Compare which lag columns are included (everything else identical). LightGBM multiclass, 100 rounds, same params/folds/metric as run_feat.py.
Sets: L5 (reference = existing table), L6 (lag1-6), L12 (lag1-12), L6+12 (lag1-6 + lag12). Results -> analysis_output/models/lag_experiments.csv (resume-able)."""
import argparse, re, time, os, json
import numpy as np, pandas as pd, lightgbm as lgb
import santander.train_common as tc
from santander.train_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--sets", default="L5,L6,L12,L6+12"); ap.add_argument("--val-months", type=int, nargs="+", default=[13, 14, 15, 16])
ap.add_argument("--rounds", type=int, default=100); ap.add_argument("--budget", type=float, default=110); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args(); T0 = time.time()
OUT = os.path.join(HERE, "analysis_output", "models", "lag_experiments.csv")
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["set", "val_month", "seed"])
df = pd.read_parquet(os.path.join(PROC, "features_train_lag12.parquet"))
allf = json.load(open(os.path.join(PROC, "feature_columns_lag12.json")))["features"]
lagk = lambda c: int(re.search(r"_lag(\d+)$", c).group(1)) if re.search(r"_lag\d+$", c) else None
other = [c for c in allf if lagk(c) is None]
def cols_for(s):
    keep = {"L5": set(range(1, 6)), "L6": set(range(1, 7)), "L12": set(range(1, 13)), "L6+12": set(range(1, 7)) | {12}}[s]
    return [c for c in allf if lagk(c) is None or lagk(c) in keep]
for s in a.sets.split(","):
    feats = cols_for(s)
    for V in a.val_months:
        if ((done.set == s) & (done.val_month == V) & (done.seed == a.seed)).any(): continue
        if time.time() - T0 > a.budget: print("time budget reached; re-run to resume"); raise SystemExit
        t0 = time.time()
        tr = df[df.target_month_idx < V].reset_index(drop=True); va = df[df.target_month_idx == V].reset_index(drop=True)
        Xtr = tr[feats].to_numpy("float32", na_value=np.nan); Xva = va[feats].to_numpy("float32", na_value=np.nan)
        Y = tr[LABELS].to_numpy("int8", na_value=0); ri, ci = np.nonzero(Y)
        cat = [feats.index(c) for c in CAT_COLS]
        p = dict(objective="multiclass", num_class=NP, learning_rate=0.1, num_leaves=31, min_child_samples=100, feature_fraction=0.7,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=4, verbose=-1, seed=a.seed)
        b = lgb.train(p, lgb.Dataset(Xtr[ri], ci, categorical_feature=cat), num_boost_round=a.rounds)
        act, own = actual_and_owned(va); m, r = evaluate(b.predict(Xva), own, act)
        row = dict(set=s, val_month=V, seed=a.seed, MAP7=m, recall7=r, n_features=len(feats), seconds=time.time() - t0)
        pd.concat([done, pd.DataFrame([row])], ignore_index=True).to_csv(OUT, index=False); done = pd.read_csv(OUT)
        print(f"{s:6s} val{V}: MAP@7 {m:.4f} ({len(feats)} feats, {time.time()-t0:.0f}s)", flush=True)
