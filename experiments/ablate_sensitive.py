"""cap+B with vs without sex / foreigner_index / residence_index; 4 folds x 2 seeds; 'with' numbers come from change_experiments.csv (same code path)."""
import time, os, numpy as np, pandas as pd, lightgbm as lgb
from santander.train_common import LABELS, actual_and_owned, evaluate, HERE
from santander.final_features import *
OUT = os.path.join(HERE, "analysis_output", "models", "sensitive_ablation.csv")
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["val_month", "seed"])
DROP = ["sex", "foreigner_index", "residence_index"]
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
        done = pd.concat([done, pd.DataFrame([dict(val_month=V, seed=sd, MAP7_no_sensitive=m)])], ignore_index=True); done.to_csv(OUT, index=False)
        print(f"seed{sd} val{V}: {m:.4f}", flush=True)
w = pd.read_csv(os.path.join(HERE, "analysis_output", "models", "change_experiments.csv")).query("variant == 'cap+B'")[["val_month", "seed", "MAP7"]]
m = done.astype({"val_month": int, "seed": int}).merge(w, on=["val_month", "seed"]); m["diff"] = m.MAP7_no_sensitive - m.MAP7
print(m.round(4).to_string(index=False)); print("mean with %.4f  without %.4f  diff %+.4f  (without better in %d/8)" % (m.MAP7.mean(), m.MAP7_no_sensitive.mean(), m["diff"].mean(), (m["diff"] > 0).sum()))
