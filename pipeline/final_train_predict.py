"""Final model: LightGBM multiclass on ALL adder rows (target months 2015-02 .. 2016-05), feature set cap+B, 100 rounds (chosen on 4 folds).
Predicts 2016-06 for every test customer: top-7 products NOT held at 2016-05, original Spanish product names, Kaggle format."""
import time, json, os, numpy as np, pandas as pd, lightgbm as lgb
from santander.column_map import COLUMN_MAP, PRODUCT_COLUMNS
from santander.train_common import LABELS, HERE, top_k
from santander.final_features import *
T0 = time.time(); ROUNDS, SEED = 100, 1
tr = load("train"); Y = tr[LABELS].to_numpy("int8", na_value=0); ri, ci = np.nonzero(Y)
print(f"train: {len(ri):,} expanded rows from {len(tr):,} adder rows, months {tr.target_month_idx.min()}..{tr.target_month_idx.max()}, {len(FINAL_FEATURES)} features", flush=True)
b = lgb.train({**PARAMS, "seed": SEED}, lgb.Dataset(matrix(tr)[ri], ci, categorical_feature=cat_idx()), num_boost_round=ROUNDS)
b.save_model(os.path.join(HERE, "models", "final_lgbm.txt"))
json.dump({"features": FINAL_FEATURES, "categorical": FINAL_CATS, "params": {**PARAMS, "seed": SEED}, "rounds": ROUNDS,
           "train_target_months": [int(tr.target_month_idx.min()), int(tr.target_month_idx.max())], "train_rows": int(len(ri))},
          open(os.path.join(HERE, "models", "final_lgbm_meta.json"), "w"), indent=1)
print(f"trained in {time.time()-T0:.0f}s", flush=True)
te = load("test"); X = matrix(te)
P = b.predict(X).astype("float32")
owned = te[[f"{p}_lag1" for p in PRODUCT_COLUMNS]].to_numpy("float32", na_value=np.nan) == 1
top = top_k(P, owned)
spanish = np.array([k for k in COLUMN_MAP if k.endswith("_ult1")]); assert [COLUMN_MAP[k] for k in spanish] == PRODUCT_COLUMNS
n_free = (~owned).sum(1); valid = np.arange(7)[None, :] < n_free[:, None]          # never pad with owned products
added = [" ".join(spanish[t][v]) for t, v in zip(top, valid)]
sub = pd.DataFrame({"ncodpers": te.customer_id.to_numpy(), "added_products": added})
sub.to_csv(os.path.join(HERE, "submission", "submission.csv"), index=False)
np.save(os.path.join(HERE, "submission", "test_probabilities.npy"), P)
print(f"predicted + wrote submission in {time.time()-T0:.0f}s", flush=True)
