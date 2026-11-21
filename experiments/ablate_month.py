"""Final features (sex / foreigner / residence already removed) with vs without month_of_year; 4 folds x 2 seeds.
Results -> analysis_output/models/month_ablation.csv (resume-able)."""
import time, os, numpy as np, pandas as pd, lightgbm as lgb
from santander.train_common import LABELS, actual_and_owned, evaluate, HERE
from santander.final_features import *
OUT = os.path.join(HERE, "analysis_output", "models", "month_ablation.csv")
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["val_month", "seed"])
DROP = ["sex", "foreigner_index", "residence_index", "month_of_year"]
feats = [f for f in FINAL_FEATURES if f not in DROP]; cat = [feats.index(c) for c in FINAL_CATS if c in feats]
df = load("train"); T0 = time.time()
for sd in (1, 2):
    for V in (13, 14, 15, 16):
        if ((done.val_month == V) & (done.seed == sd)).any(): continue
        if time.time() - T0 > 110: print("budget; re-run"); raise SystemExit
        tr, va = df[df.target_month_idx < V], df[df.target_month_idx == V]
        Y = tr[LABELS].to_numpy("int8", na_value=0); ri, ci = np.nonzero(Y)
        b = lgb.train({**PARAMS, "seed": sd}, lgb.Dataset(tr[feats].to_numpy("float32", na_value=np.nan)[ri], ci, categorical_feature=cat), num_boost_round=100)
        act, own = actual_and_owned(va); m = evaluate(b.predict(va[feats].to_numpy("float32", na_value=np.nan)), own, act)[0]
        done = pd.concat([done, pd.DataFrame([dict(val_month=V, seed=sd, MAP7_no_month=m)])], ignore_index=True); done.to_csv(OUT, index=False)
        print(f"seed{sd} val{V}: {m:.4f}", flush=True)
w = pd.read_csv(os.path.join(HERE, "analysis_output", "models", "sensitive_ablation.csv")).rename(columns={"MAP7_no_sensitive": "MAP7"})
m = done.astype({"val_month": int, "seed": int}).merge(w.astype({"val_month": int, "seed": int}), on=["val_month", "seed"]); m["diff"] = m.MAP7_no_month - m.MAP7
print(m.round(4).to_string(index=False)); print("mean with month %.4f  without %.4f  diff %+.4f  (without better in %d/8)" % (m.MAP7.mean(), m.MAP7_no_month.mean(), m["diff"].mean(), (m["diff"] > 0).sum()))
