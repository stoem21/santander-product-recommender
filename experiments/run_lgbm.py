"""LightGBM multi-class on adder rows. Usage: python -m experiments.run_lgbm --val-month 16 [--rounds 300] [--weight]"""
import argparse, time, resource, json, os
import numpy as np, pandas as pd, lightgbm as lgb
from santander.train_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--val-month", type=int, default=16)
ap.add_argument("--rounds", type=int, default=300)
ap.add_argument("--lr", type=float, default=0.08)
ap.add_argument("--leaves", type=int, default=31)
ap.add_argument("--min-child", type=int, default=100)
ap.add_argument("--weight", action="store_true")
ap.add_argument("--min-month", type=int, default=1, help="drop train target months below this")
ap.add_argument("--max-train-rows", type=int, default=0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--ff", type=float, default=0.7)
ap.add_argument("--tag", default="lgbm")
a = ap.parse_args()

t0 = time.time()
df = load_train()
tr, va = split_by_month(df, a.val_month)
tr = tr[tr.target_month_idx >= a.min_month].reset_index(drop=True)
X, y, w, _ = expand_rows(tr)
if a.max_train_rows and len(y) > a.max_train_rows:
    keep = np.random.default_rng(0).choice(len(y), a.max_train_rows, replace=False); X, y, w = X[keep], y[keep], w[keep]
Xv_act, owned_v = actual_and_owned(va)
Xv = to_matrix(va)
rng = np.random.default_rng(0)
sub = rng.choice(len(tr), size=min(30000, len(tr)), replace=False)
Xs, (act_s, own_s) = to_matrix(tr.iloc[sub]), actual_and_owned(tr.iloc[sub])
print(f"train rows {len(y):,} (customers {len(tr):,}), val customers {len(va):,}, load {time.time()-t0:.0f}s", flush=True)

# validation expanded rows only for logloss curve
Xve, yve, wve, _ = expand_rows(va)
params = dict(objective="multiclass", num_class=NP, learning_rate=a.lr, num_leaves=a.leaves,
              min_child_samples=a.min_child, feature_fraction=a.ff, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, max_bin=127, num_threads=4, verbose=-1, seed=a.seed)
cat = [FEATURES.index(c) for c in CAT_COLS]
dtr = lgb.Dataset(X, y, weight=w if a.weight else None, categorical_feature=cat, free_raw_data=False)
dva = lgb.Dataset(Xve, yve, categorical_feature=cat, reference=dtr)
ev = {}
t1 = time.time()
bst = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dtr, dva], valid_names=["train", "val"],
                callbacks=[lgb.record_evaluation(ev), lgb.log_evaluation(25)])
train_time = time.time() - t1
print(f"train time {train_time:.0f}s ({train_time/a.rounds:.2f}s/round), peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB", flush=True)

rows = []
for it in sorted(set([25, 50, 100, 150, 200, 300, 400, 500, a.rounds])):
    if it > a.rounds: continue
    mv, rv = evaluate(bst.predict(Xv, num_iteration=it), owned_v, Xv_act)
    ms, rs = evaluate(bst.predict(Xs, num_iteration=it), own_s, act_s)
    rows.append(dict(iter=it, val_MAP7=mv, val_recall7=rv, train_MAP7=ms, train_recall7=rs,
                     val_logloss=ev["val"]["multi_logloss"][it-1], train_logloss=ev["train"]["multi_logloss"][it-1]))
res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
os.makedirs(os.path.join(HERE, "analysis_output", "models"), exist_ok=True)
res.assign(val_month=a.val_month, tag=a.tag, weight=a.weight, train_seconds=train_time).to_csv(
    os.path.join(HERE, "analysis_output", "models", f"{a.tag}_val{a.val_month}.csv"), index=False)
np.save(os.path.join(HERE, "analysis_output", "models", f"{a.tag}_val{a.val_month}_scores.npy"),
        bst.predict(Xv).astype("float32"))
print("done", f"{time.time()-t0:.0f}s total")
