"""XGBoost / CatBoost multi-class on the same adder rows + same evaluation. 
Usage: python -m experiments.run_other --model xgb|cat --val-month 16 --rounds N [--max-train-rows N]"""
import argparse, time, resource, os
import numpy as np, pandas as pd
from santander.train_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["xgb", "cat"])
ap.add_argument("--val-month", type=int, default=16)
ap.add_argument("--rounds", type=int, default=200)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--depth", type=int, default=6)
ap.add_argument("--max-train-rows", type=int, default=0)
ap.add_argument("--tag", default=None)
a = ap.parse_args()
tag = a.tag or a.model

t0 = time.time()
df = load_train()
tr, va = split_by_month(df, a.val_month)
X, y, w, _ = expand_rows(tr)
if a.max_train_rows and len(y) > a.max_train_rows:
    keep = np.random.default_rng(0).choice(len(y), a.max_train_rows, replace=False); X, y = X[keep], y[keep]
act_v, own_v = actual_and_owned(va); Xv = to_matrix(va)
sub = np.random.default_rng(0).choice(len(tr), size=min(30000, len(tr)), replace=False)
Xs = to_matrix(tr.iloc[sub]); act_s, own_s = actual_and_owned(tr.iloc[sub])
print(f"{a.model}: train rows {len(y):,}, val customers {len(va):,}", flush=True)

t1 = time.time()
if a.model == "xgb":
    import xgboost as xgb
    dtr = xgb.DMatrix(X, label=y)
    params = dict(objective="multi:softprob", num_class=NP, eta=a.lr, max_depth=a.depth, subsample=0.8,
                  colsample_bytree=0.7, min_child_weight=5, reg_lambda=5.0, tree_method="hist", max_bin=127,
                  nthread=4, seed=1)
    bst = xgb.train(params, dtr, num_boost_round=a.rounds)
    def predict(M, it):
        return bst.predict(xgb.DMatrix(M), iteration_range=(0, it))
else:
    from catboost import CatBoostClassifier, Pool
    cat = [FEATURES.index(c) for c in CAT_COLS]
    def prep(M):
        d = pd.DataFrame(M, columns=FEATURES)
        for c in CAT_COLS: d[c] = d[c].fillna(-1).astype("int32")
        return d
    m = CatBoostClassifier(loss_function="MultiClass", iterations=a.rounds, learning_rate=a.lr, depth=a.depth,
                           l2_leaf_reg=5, thread_count=4, random_seed=1, verbose=10, border_count=64,
                           save_snapshot=True, snapshot_file=f"/tmp/{tag}_val{a.val_month}.snap", snapshot_interval=15)
    m.fit(Pool(prep(X), y, cat_features=CAT_COLS))
    def predict(M, it):
        p = m.predict_proba(prep(M), ntree_end=it)
        full = np.zeros((len(M), NP), dtype="float32")      # classes absent from a subsample get probability 0
        full[:, np.asarray(m.classes_, dtype=int)] = p
        return full
train_time = time.time() - t1
print(f"train time {train_time:.0f}s ({train_time/a.rounds:.2f}s/round), peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB", flush=True)

rows = []
for it in sorted(set([a.rounds // 4, a.rounds // 2, 3 * a.rounds // 4, a.rounds])):
    mv, rv = evaluate(predict(Xv, it), own_v, act_v)
    ms, rs = evaluate(predict(Xs, it), own_s, act_s)
    rows.append(dict(iter=it, val_MAP7=mv, val_recall7=rv, train_MAP7=ms, train_recall7=rs))
res = pd.DataFrame(rows); print(res.round(4).to_string(index=False))
res.assign(val_month=a.val_month, tag=tag, train_seconds=train_time, train_rows=len(y)).to_csv(
    os.path.join(HERE, "analysis_output", "models", f"{tag}_val{a.val_month}.csv"), index=False)
np.save(os.path.join(HERE, "analysis_output", "models", f"{tag}_val{a.val_month}_scores.npy"), predict(Xv, a.rounds).astype("float32"))
print("done", f"{time.time()-t0:.0f}s total")
