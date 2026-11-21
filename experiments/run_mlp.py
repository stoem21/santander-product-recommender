"""Small MLP (sklearn) multi-class on the same adder rows. Epoch-chunked with a pickle checkpoint so it fits 3-min calls.
Usage: python -m experiments.run_mlp --val-month 16 --epochs 12"""
import argparse, time, os, pickle
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from santander.train_common import *
ap = argparse.ArgumentParser()
ap.add_argument("--val-month", type=int, default=16); ap.add_argument("--epochs", type=int, default=12)
a = ap.parse_args()
t0 = time.time()
df = load_train(); tr, va = split_by_month(df, a.val_month)
maps = json.load(open(os.path.join(PROC, "category_maps.json")))
card = {c: (len(maps[c]) if c in maps else 53) for c in CAT_COLS}      # province_code: 0..52
card["join_channel"] = 16
num_cols = [c for c in FEATURES if c not in CAT_COLS]
med = tr[num_cols].median()

def prep(d):
    Xn = d[num_cols].to_numpy("float32", na_value=np.nan)
    lagcols = [i for i, c in enumerate(num_cols) if "_lag" in c]
    miss = np.isnan(Xn[:, [num_cols.index(c) for c in ["household_income_log", "age"]]]).astype("float32")
    Xn = np.where(np.isnan(Xn), np.nan_to_num(med.to_numpy("float32")[None, :]), Xn)
    for i in lagcols: Xn[:, i] = np.where(d[num_cols[i]].isna().to_numpy(), -1.0, Xn[:, i])
    return Xn, miss
Xn_tr, m_tr = prep(tr)
mu, sd = Xn_tr.mean(0), Xn_tr.std(0) + 1e-6
def full(d):
    Xn, m = prep(d)
    parts = [(Xn - mu) / sd, m]
    for c in CAT_COLS:
        v = d[c].to_numpy("float32", na_value=-1).astype(int); oh = np.zeros((len(d), card[c]), dtype="float32")
        ok = (v >= 0) & (v < card[c]); oh[np.nonzero(ok)[0], v[ok]] = 1; parts.append(oh)
    return np.concatenate(parts, axis=1).astype("float32")
_, y, w, r = expand_rows(tr)
Xe = full(tr)[r]; Xv = full(va); act_v, own_v = actual_and_owned(va)
print(f"mlp input dim {Xe.shape[1]}, rows {len(y):,}, prep {time.time()-t0:.0f}s", flush=True)
ck = f"/tmp/mlp_val{a.val_month}.pkl"
st = pickle.load(open(ck, "rb")) if os.path.exists(ck) else {"epochs": 0, "model": None, "secs": 0.0}
mdl = st["model"] or MLPClassifier(hidden_layer_sizes=(256, 128), batch_size=512, learning_rate_init=1e-3, alpha=1e-4,
                                   max_iter=1, warm_start=True, random_state=1)
import warnings; warnings.filterwarnings("ignore")
while st["epochs"] < a.epochs and time.time() - t0 < 120:
    e0 = time.time(); mdl.fit(Xe, y); st["epochs"] += 1; st["secs"] += time.time() - e0
    print(f"epoch {st['epochs']} loss {mdl.loss_:.4f} ({time.time()-e0:.0f}s)", flush=True)
st["model"] = mdl; pickle.dump(st, open(ck, "wb"))
if st["epochs"] >= a.epochs:
    P = np.zeros((len(va), NP), dtype="float32"); P[:, np.asarray(mdl.classes_, dtype=int)] = mdl.predict_proba(Xv)
    m, rr = evaluate(P, own_v, act_v); print(f"mlp val{a.val_month}: MAP@7 {m:.4f} recall@7 {rr:.4f}")
    np.save(os.path.join(HERE, "analysis_output", "models", f"mlp_val{a.val_month}_scores.npy"), P)
    pd.DataFrame([dict(iter=st["epochs"], val_MAP7=m, val_recall7=rr, val_month=a.val_month, tag="mlp", train_seconds=st["secs"])]).to_csv(
        os.path.join(HERE, "analysis_output", "models", f"mlp_val{a.val_month}.csv"), index=False)
else:
    print("not finished; re-run to resume")
