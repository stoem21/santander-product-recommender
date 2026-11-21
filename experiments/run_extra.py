"""Non-multiclass alternatives, same split/evaluation.  modes:
  assoc  : item-association rule  S[i,j] = P(add j | owns i), score_j = mean_i-owned S[i,j]   (ownership vector only)
  binary : 24 independent LightGBM binary models; model j trained on adder customers who do NOT own j at t-1, label new_j
Usage: python -m experiments.run_extra --mode assoc|binary --val-month 16"""
import argparse, time, os
import numpy as np, pandas as pd
from santander.train_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["assoc", "binary"])
ap.add_argument("--val-month", type=int, default=16)
ap.add_argument("--rounds", type=int, default=100)
ap.add_argument("--lr", type=float, default=0.1)
a = ap.parse_args()
OUT = os.path.join(HERE, "analysis_output", "models")
t0 = time.time()
df = load_train()
tr, va = split_by_month(df, a.val_month)
act_tr, own_tr = actual_and_owned(tr)
act_v, own_v = actual_and_owned(va)

if a.mode == "assoc":
    O = own_tr.astype("float32")
    A = act_tr.astype("float32")
    cnt = O.sum(axis=0)                                   # customers owning i
    S = (O.T @ A) / np.maximum(cnt[:, None], 1)           # P(add j | owns i)
    pop = A.mean(axis=0)
    Ov = own_v.astype("float32")
    n_own = Ov.sum(axis=1, keepdims=True)
    scores = np.where(n_own > 0, (Ov @ S) / np.maximum(n_own, 1), pop)
    m, r = evaluate(scores, own_v, act_v)
    print(f"assoc val{a.val_month}: MAP@7 {m:.4f} recall@7 {r:.4f}")
    pd.DataFrame([dict(iter=0, val_MAP7=m, val_recall7=r, val_month=a.val_month, tag="assoc", train_seconds=time.time()-t0)]).to_csv(
        os.path.join(OUT, f"assoc_val{a.val_month}.csv"), index=False)
else:
    import lightgbm as lgb
    Xtr, Xv = to_matrix(tr), to_matrix(va)
    cat = [FEATURES.index(c) for c in CAT_COLS]
    part = os.path.join("/tmp", f"binary_val{a.val_month}")
    os.makedirs(part, exist_ok=True)
    params = dict(objective="binary", learning_rate=a.lr, num_leaves=31, min_child_samples=100, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=4, verbose=-1, seed=1)
    secs = 0.0
    for j, p in enumerate(PRODUCT_COLUMNS):
        f = os.path.join(part, f"{j}.npy")
        if os.path.exists(f): continue
        if time.time() - t0 > 140: print("time budget reached; re-run to resume"); raise SystemExit
        use = ~own_tr[:, j]; y = act_tr[use, j].astype(int)
        s0 = time.time()
        if y.sum() < 50:                                     # too few positives: fall back to base rate
            sc = np.full(len(va), y.mean(), dtype="float32")
        else:
            b = lgb.train(params, lgb.Dataset(Xtr[use], y, categorical_feature=cat), num_boost_round=a.rounds)
            sc = b.predict(Xv).astype("float32")
        np.save(f, sc); print(f"{p}: positives {int(y.sum()):,} / {use.sum():,}  {time.time()-s0:.0f}s", flush=True)
    scores = np.stack([np.load(os.path.join(part, f"{j}.npy")) for j in range(NP)], axis=1)
    m, r = evaluate(scores, own_v, act_v)
    print(f"binary x24 val{a.val_month}: MAP@7 {m:.4f} recall@7 {r:.4f}")
    np.save(os.path.join(OUT, f"binary_val{a.val_month}_scores.npy"), scores)
    pd.DataFrame([dict(iter=a.rounds, val_MAP7=m, val_recall7=r, val_month=a.val_month, tag="binary24")]).to_csv(
        os.path.join(OUT, f"binary24_val{a.val_month}.csv"), index=False)
