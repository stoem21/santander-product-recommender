"""PyTorch NN (embeddings for categorical cols + MLP, dropout), multi-class on adder rows.
Epoch selection uses month (val_month-1) as an *early-stopping month* - trained on months < val_month-1 - so the reported
val_month is never used for any choice.  Checkpointed per epoch so it fits the 3-minute call limit (re-run to resume).
Usage: python -m experiments.run_nn --val-month 16 --epochs 14"""
import argparse, time, os, json, copy
import numpy as np, pandas as pd, torch, torch.nn as nn
from santander.train_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--val-month", type=int, default=16); ap.add_argument("--epochs", type=int, default=14)
ap.add_argument("--layers", default="512,256"); ap.add_argument("--drop", type=float, default=0.25)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--tag", default="nn")
a = ap.parse_args()
t0 = time.time(); torch.set_num_threads(4); torch.manual_seed(1); np.random.seed(1)
V, MON = a.val_month, a.val_month - 1
df = load_train()
tr = df[df.target_month_idx < MON].reset_index(drop=True)
mon = df[df.target_month_idx == MON].reset_index(drop=True)
va = df[df.target_month_idx == V].reset_index(drop=True)

maps = json.load(open(os.path.join(PROC, "category_maps.json")))
card = {c: (len(maps[c]) if c in maps else 53) for c in CAT_COLS}; card["join_channel"] = 16
num_cols = [c for c in FEATURES if c not in CAT_COLS]
med = tr[num_cols].median().to_numpy("float32")
lag_idx = [i for i, c in enumerate(num_cols) if "_lag" in c]
miss_cols = [num_cols.index(c) for c in ["household_income_log", "age"]]

def numeric(d):
    X = d[num_cols].to_numpy("float32", na_value=np.nan)
    miss = np.isnan(X[:, miss_cols]).astype("float32")
    nanm = np.isnan(X)
    X = np.where(nanm, med[None, :], X)
    X[:, lag_idx] = np.where(nanm[:, lag_idx], -1.0, X[:, lag_idx])
    return X, miss
Xn0, _ = numeric(tr); mu, sd = Xn0.mean(0), Xn0.std(0) + 1e-6
def encode(d):
    X, miss = numeric(d)
    xn = np.concatenate([(X - mu) / sd, miss], axis=1).astype("float32")
    xc = np.stack([np.where((v := d[c].to_numpy("float32", na_value=-1).astype(int)) >= 0, np.minimum(v, card[c] - 1), card[c])
                   for c in CAT_COLS], axis=1).astype("int64")
    return torch.from_numpy(xn), torch.from_numpy(xc)
_, y, _, r = expand_rows(tr)
Xn_all, Xc_all = encode(tr)
Xn, Xc, Y = Xn_all[r], Xc_all[r], torch.from_numpy(y.astype("int64"))
Xn_m, Xc_m = encode(mon); Xn_v, Xc_v = encode(va)
act_m, own_m = actual_and_owned(mon); act_v, own_v = actual_and_owned(va)
print(f"train rows {len(Y):,} (months < {MON}), early-stop month {MON}: {len(mon):,}, val month {V}: {len(va):,}, prep {time.time()-t0:.0f}s", flush=True)

class Net(nn.Module):
    def __init__(s, n_num, layers, p):
        super().__init__()
        s.emb = nn.ModuleList([nn.Embedding(card[c] + 1, min(8, max(2, card[c] // 2))) for c in CAT_COLS])
        d = n_num + sum(e.embedding_dim for e in s.emb); mods = []
        for h in layers:
            mods += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]; d = h
        s.mlp = nn.Sequential(*mods, nn.Linear(d, NP))
    def forward(s, xn, xc):
        return s.mlp(torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(s.emb)], dim=1))
net = Net(Xn.shape[1], [int(x) for x in a.layers.split(',')], a.drop)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
steps_per_epoch = (len(Y) + 1023) // 1024
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.15)
ck = f"/tmp/{a.tag}_val{V}.pt"
st = dict(epoch=0, hist=[], best=None, secs=0.0)
if os.path.exists(ck):
    c = torch.load(ck, weights_only=False); net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); st = c["st"]
    print("resumed at epoch", st["epoch"], flush=True)

@torch.no_grad()
def predict(xn, xc):
    net.eval(); out = []
    for i in range(0, len(xn), 8192): out.append(torch.softmax(net(xn[i:i+8192], xc[i:i+8192]), 1))
    return torch.cat(out).numpy()

while st["epoch"] < a.epochs and time.time() - t0 < 125:
    e0 = time.time(); net.train(); perm = torch.randperm(len(Y)); tot = 0.0
    for i in range(0, len(Y), 1024):
        b = perm[i:i+1024]
        if len(b) < 2: continue
        loss = nn.functional.cross_entropy(net(Xn[b], Xc[b]), Y[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
    st["epoch"] += 1; st["secs"] += time.time() - e0
    pm = predict(Xn_m, Xc_m); pv = predict(Xn_v, Xc_v)
    mm, _ = evaluate(pm, own_m, act_m); mv, rv = evaluate(pv, own_v, act_v)
    st["hist"].append(dict(epoch=st["epoch"], train_loss=tot / len(Y), monitor_MAP7=mm, val_MAP7=mv, val_recall7=rv))
    if st["best"] is None or mm > st["best"]["monitor_MAP7"]:
        st["best"] = dict(epoch=st["epoch"], monitor_MAP7=mm, val_MAP7=mv, val_recall7=rv, scores=pv.astype("float32"))
    print(f"epoch {st['epoch']:2d} loss {tot/len(Y):.4f} monitor({MON}) MAP@7 {mm:.4f}  val({V}) MAP@7 {mv:.4f}  ({time.time()-e0:.0f}s)", flush=True)
torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), st=st), ck)
if st["epoch"] >= a.epochs:
    b = st["best"]; print(f"BEST by monitor month: epoch {b['epoch']}  val MAP@7 {b['val_MAP7']:.4f} recall@7 {b['val_recall7']:.4f}; last epoch val MAP@7 {st['hist'][-1]['val_MAP7']:.4f}")
    od = os.path.join(HERE, "analysis_output", "models")
    np.save(os.path.join(od, f"{a.tag}_val{V}_scores.npy"), b["scores"])
    pd.DataFrame(st["hist"]).assign(val_month=V, tag=a.tag).to_csv(os.path.join(od, f"{a.tag}_val{V}_history.csv"), index=False)
    pd.DataFrame([dict(iter=b["epoch"], val_MAP7=b["val_MAP7"], val_recall7=b["val_recall7"], monitor_MAP7=b["monitor_MAP7"],
                       val_month=V, tag=a.tag, train_seconds=st["secs"], train_rows=len(Y))]).to_csv(os.path.join(od, f"{a.tag}_val{V}.csv"), index=False)
else:
    print("not finished; re-run to resume")
