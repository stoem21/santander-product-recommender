"""base / cap (n_prior_months capped at 6) / cap+A / cap+B ; LightGBM multiclass 100 rounds, same params as run_feat.py.
Also MAP@7 on 'stagnant' val customers (no add/drop in the 5-month window). Results -> analysis_output/models/change_experiments.csv (resume-able)."""
import argparse, time, os, json
import numpy as np, pandas as pd, lightgbm as lgb
from santander.train_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--variants", default="base,cap,cap+A,cap+B"); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
ap.add_argument("--budget", type=float, default=120); a = ap.parse_args(); T0 = time.time()
OUT = os.path.join(HERE, "analysis_output", "models", "change_experiments.csv")
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["variant", "val_month", "seed"])
df = load_train(); pc_ = pd.read_parquet(os.path.join(PROC, "profile_change_train.parquet"))
assert (pc_.customer_id.values == df.customer_id.values).all()
CH = json.load(open(os.path.join(PROC, "profile_change_columns.json")))
df = pd.concat([df, pc_.drop(columns=["customer_id", "target_month_idx"])], axis=1)
df["n_prior_capped"] = np.minimum(df["n_prior_months"], 6)
def feats_for(v):
    if v == "base": return FEATURES
    f = [c if c != "n_prior_months" else "n_prior_capped" for c in FEATURES]
    if "+A" in v: f += CH["A"]
    if "+B" in v: f += CH["B"]
    return f
CATX = CAT_COLS + ["segment_prev", "segment_plag2", "segment_plag3", "segment_plag4", "segment_plag5", "relation_type_plag2", "relation_type_plag3", "relation_type_plag4", "relation_type_plag5"]
for sd in a.seeds:
    for v in a.variants.split(","):
        feats = feats_for(v)
        for V in (13, 14, 15, 16):
            if ((done.variant == v) & (done.val_month == V) & (done.seed == sd)).any(): continue
            if time.time() - T0 > a.budget: print("time budget reached; re-run to resume"); raise SystemExit
            t0 = time.time(); tr = df[df.target_month_idx < V]; va = df[df.target_month_idx == V]
            Y = tr[LABELS].to_numpy("int8", na_value=0); ri, ci = np.nonzero(Y)
            cat = [feats.index(c) for c in CATX if c in feats]
            p = dict(objective="multiclass", num_class=NP, learning_rate=0.1, num_leaves=31, min_child_samples=100, feature_fraction=0.7,
                     bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=4, verbose=-1, seed=sd)
            b = lgb.train(p, lgb.Dataset(tr[feats].to_numpy("float32", na_value=np.nan)[ri], ci, categorical_feature=cat), num_boost_round=100)
            act, own = actual_and_owned(va); sc = b.predict(va[feats].to_numpy("float32", na_value=np.nan))
            m, r = evaluate(sc, own, act); stag = ((va.n_adds_last5 + va.n_drops_last5) == 0).to_numpy()
            ms, _ = evaluate(sc[stag], own[stag], act[stag])
            row = dict(variant=v, val_month=V, seed=sd, MAP7=m, MAP7_stagnant=ms, n_stagnant=int(stag.sum()), n_features=len(feats), seconds=time.time() - t0)
            pd.concat([done, pd.DataFrame([row])], ignore_index=True).to_csv(OUT, index=False); done = pd.read_csv(OUT)
            print(f"seed{sd} {v:6s} val{V}: MAP@7 {m:.4f} stagnant {ms:.4f} ({len(feats)} feats, {time.time()-t0:.0f}s)", flush=True)
