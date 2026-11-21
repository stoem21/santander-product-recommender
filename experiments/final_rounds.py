"""Number of boosting rounds for the final model: LightGBM on the final features, MAP@7 at 50..250 rounds on each of the
4 folds. 75..150 rounds scored the same, 100 was chosen. Results -> analysis_output/models/final_rounds.csv."""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb, os
from santander.train_common import LABELS, actual_and_owned, evaluate, HERE
from santander.final_features import *
OUT = os.path.join(HERE, "analysis_output", "models", "final_rounds.csv")
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["val_month"])
df = load("train"); T0 = time.time()
for V in (13, 14, 15, 16):
    if (done.val_month == V).any(): continue
    if time.time() - T0 > 60: print("budget; re-run"); break
    tr, va = df[df.target_month_idx < V], df[df.target_month_idx == V]
    Y = tr[LABELS].to_numpy("int8", na_value=0); ri, ci = np.nonzero(Y)
    b = lgb.train({**PARAMS, "seed": 1}, lgb.Dataset(matrix(tr)[ri], ci, categorical_feature=cat_idx()), num_boost_round=250)
    act, own = actual_and_owned(va); Xv = matrix(va)
    rows = [dict(val_month=V, rounds=it, MAP7=evaluate(b.predict(Xv, num_iteration=it), own, act)[0]) for it in (50, 75, 100, 125, 150, 200, 250)]
    done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); done.to_csv(OUT, index=False)
    print(V, {r["rounds"]: round(r["MAP7"], 4) for r in rows}, f"{time.time()-T0:.0f}s", flush=True)
