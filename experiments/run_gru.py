"""Sequence model: GRU over 12 monthly product-ownership vectors (lag12 -> lag1) + static (non-lag) features via embeddings/MLP.
Same folds / metric / epoch-selection rule as run_nn.py: epochs chosen on month (val-1), trained on months < val-1, val month never used for choices.
Checkpointed per epoch (re-run to resume). Usage: python -m experiments.run_gru --val-month 16 --epochs 12"""
import argparse, time, os, json, re
import numpy as np, pandas as pd, torch, torch.nn as nn
from santander.train_common import PRODUCT_COLUMNS, LABELS, CAT_COLS, NP, PROC, HERE, evaluate, actual_and_owned

ap = argparse.ArgumentParser()
ap.add_argument("--val-month", type=int, default=16); ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--drop", type=float, default=0.25)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--tag", default="gru")
a = ap.parse_args()
t0 = time.time(); torch.set_num_threads(4); torch.manual_seed(1); np.random.seed(1)
V, MON, L = a.val_month, a.val_month - 1, 12
df = pd.read_parquet(os.path.join(PROC, "features_train_lag12.parquet"))
allf = json.load(open(os.path.join(PROC, "feature_columns_lag12.json")))["features"]
static = [c for c in allf if not re.search(r"_lag\d+$", c)]
num_cols = [c for c in static if c not in CAT_COLS]
tr = df[df.target_month_idx < MON].reset_index(drop=True)
mon = df[df.target_month_idx == MON].reset_index(drop=True)
va = df[df.target_month_idx == V].reset_index(drop=True)
maps = json.load(open(os.path.join(PROC, "category_maps.json")))
card = {c: (len(maps[c]) if c in maps else 53) for c in CAT_COLS}; card["join_channel"] = 16
med = tr[num_cols].median().to_numpy("float32")
Xn0 = tr[num_cols].to_numpy("float32", na_value=np.nan); Xn0 = np.where(np.isnan(Xn0), med, Xn0); mu, sd = Xn0.mean(0), Xn0.std(0) + 1e-6
miss_idx = [num_cols.index(c) for c in ["household_income_log", "age"]]

def encode(d):
    X = d[num_cols].to_numpy("float32", na_value=np.nan); miss = np.isnan(X[:, miss_idx]).astype("float32")
    X = np.where(np.isnan(X), med, X); xn = np.concatenate([(X - mu) / sd, miss], 1).astype("float32")
    xc = np.stack([np.where((v := d[c].to_numpy("float32", na_value=-1).astype(int)) >= 0, np.minimum(v, card[c] - 1), card[c]) for c in CAT_COLS], 1)
    seq = np.stack([d[[f"{p}_lag{k}" for p in PRODUCT_COLUMNS]].to_numpy("float32", na_value=np.nan) for k in range(L, 0, -1)], 1)  # n, 12 (oldest->newest), 24
    s = np.concatenate([np.nan_to_num(seq, nan=0.0), np.isnan(seq).astype("float32")], 2)                                         # flag + unknown indicator
    return torch.from_numpy(xn), torch.from_numpy(xc.astype("int64")), torch.from_numpy(s.astype("float32"))
Y_all = tr[LABELS].to_numpy("int8", na_value=0); ri, ci = np.nonzero(Y_all)
Xn, Xc, Xs = encode(tr); Xn, Xc, Xs = Xn[ri], Xc[ri], Xs[ri]; Y = torch.from_numpy(ci.astype("int64"))
Em, Ev = encode(mon), encode(va); act_m, own_m = actual_and_owned(mon); act_v, own_v = actual_and_owned(va)
print(f"train rows {len(Y):,}, static {len(num_cols)} num + {len(CAT_COLS)} cat, seq {L}x{Xs.shape[2]}, prep {time.time()-t0:.0f}s", flush=True)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.ModuleList([nn.Embedding(card[c] + 1, min(8, max(2, card[c] // 2))) for c in CAT_COLS])
        s.gru = nn.GRU(Xs.shape[2], a.hidden, batch_first=True)
        d = Xn.shape[1] + sum(e.embedding_dim for e in s.emb) + a.hidden
        s.head = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(a.drop),
                               nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(a.drop), nn.Linear(256, NP))
    def forward(s, xn, xc, xs):
        _, h = s.gru(xs)
        return s.head(torch.cat([xn, h[-1]] + [e(xc[:, i]) for i, e in enumerate(s.emb)], 1))
net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
spe = (len(Y) + 1023) // 1024
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe, pct_start=0.15)
ck = f"/tmp/{a.tag}_val{V}.pt"; st = dict(epoch=0, hist=[], best=None, secs=0.0)
if os.path.exists(ck):
    c = torch.load(ck, weights_only=False); net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); st = c["st"]; print("resumed at epoch", st["epoch"], flush=True)

@torch.no_grad()
def predict(E):
    net.eval(); xn, xc, xs = E
    return torch.cat([torch.softmax(net(xn[i:i+8192], xc[i:i+8192], xs[i:i+8192]), 1) for i in range(0, len(xn), 8192)]).numpy()

while st["epoch"] < a.epochs and time.time() - t0 < 120:
    e0 = time.time(); net.train(); perm = torch.randperm(len(Y)); tot = 0.0
    for i in range(0, len(Y), 1024):
        b = perm[i:i+1024]
        if len(b) < 2: continue
        loss = nn.functional.cross_entropy(net(Xn[b], Xc[b], Xs[b]), Y[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
    st["epoch"] += 1; st["secs"] += time.time() - e0
    pv = predict(Ev); mm, _ = evaluate(predict(Em), own_m, act_m); mv, rv = evaluate(pv, own_v, act_v)
    st["hist"].append(dict(epoch=st["epoch"], train_loss=tot / len(Y), monitor_MAP7=mm, val_MAP7=mv))
    if st["best"] is None or mm > st["best"]["monitor_MAP7"]:
        st["best"] = dict(epoch=st["epoch"], monitor_MAP7=mm, val_MAP7=mv, val_recall7=rv, scores=pv.astype("float32"))
    print(f"epoch {st['epoch']:2d} loss {tot/len(Y):.4f} monitor({MON}) {mm:.4f} val({V}) {mv:.4f} ({time.time()-e0:.0f}s)", flush=True)
torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), st=st), ck)
if st["epoch"] >= a.epochs:
    b = st["best"]; od = os.path.join(HERE, "analysis_output", "models")
    print(f"BEST by monitor month: epoch {b['epoch']} val MAP@7 {b['val_MAP7']:.4f}")
    np.save(os.path.join(od, f"{a.tag}_val{V}_scores.npy"), b["scores"])
    pd.DataFrame(st["hist"]).to_csv(os.path.join(od, f"{a.tag}_val{V}_history.csv"), index=False)
    pd.DataFrame([dict(iter=b["epoch"], val_MAP7=b["val_MAP7"], val_recall7=b["val_recall7"], monitor_MAP7=b["monitor_MAP7"], val_month=V, tag=a.tag, train_seconds=st["secs"])]).to_csv(os.path.join(od, f"{a.tag}_val{V}.csv"), index=False)
else:
    print("not finished; re-run to resume")
