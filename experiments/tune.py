"""Equal-budget hyperparameter search for LightGBM, XGBoost and the PyTorch NN on the final 175 features.

* 10 configs per model; config 0 = the settings used so far; configs 1-9 sampled with a fixed seed.
* Evaluated on the same 4 folds as everything else (validate 2016-02..05, train on all earlier target months).
* Trees: trained in chunks and resumed from a saved model, so one unit can span several 3-minute calls;
  MAP@7 is recorded at every checkpoint, and the number of rounds is chosen later from the 4-fold mean.
* NN: number of epochs is part of the config (OneCycle schedule), checkpointed every epoch, scored at the end only.
* Phase 2 (confirm): units listed in analysis_output/tuning/confirm.json are re-run with seed 2.
* Scope: only configs 0..TUNE_N_CFG-1 of each model are run in phase 1 (env var, default 3; the full search space has 10).
  Reduced from 10 on 2026-11-20 to get results sooner; configs are a random sample, so a prefix is still an unbiased subset.
  Results of configs already run beyond that (lgbm cfg3, cfg4 partial) stay in results.csv.

Usage:  python -m experiments.tune init            # write configs.json
        python -m experiments.tune run [--budget 140] [--only lgbm|xgb|nn]
"""
import argparse, json, os, random, time
import numpy as np, pandas as pd
from santander.train_common import LABELS, actual_and_owned, evaluate, HERE, PROC
from santander.final_features import FINAL_FEATURES, FINAL_CATS, PARAMS, load, matrix, cat_idx

THREADS = int(os.environ.get("TUNE_THREADS", "4"))
N_CFG = int(os.environ.get("TUNE_N_CFG", "3"))          # configs per model in phase 1 (search space has 10)
TD = os.path.join(HERE, "analysis_output", "tuning"); os.makedirs(TD, exist_ok=True)
CK = "/tmp/tune"; os.makedirs(CK, exist_ok=True)
FOLDS = [13, 14, 15, 16]
RES, DONE, CFG, CONF = (os.path.join(TD, f) for f in ("results.csv", "units_done.csv", "configs.json", "confirm.json"))
LGB_MAX = {0.1: (250, 25), 0.05: (450, 50), 0.03: (700, 70)}      # lr -> (max rounds, checkpoint step)
XGB_MAX = {0.1: (300, 25), 0.05: (500, 50)}


def make_configs():
    rng = random.Random(42)
    C = {"lgbm": [dict(lr=0.1, num_leaves=31, min_child=100, l2=5.0, ff=0.7, bag=0.8)],
         "xgb": [dict(eta=0.1, depth=6, mcw=5, sub=0.8, col=0.7, lam=5.0)],
         "nn": [dict(layers="512,256", drop=0.25, lr=2e-3, wd=1e-4, batch=1024, epochs=14, emb=8)]}
    space = {"lgbm": dict(lr=[0.1, 0.05, 0.03], num_leaves=[15, 31, 63, 127], min_child=[20, 50, 100, 300],
                          l2=[0.0, 1.0, 5.0, 20.0], ff=[0.5, 0.7, 0.9], bag=[0.7, 0.8, 1.0]),
             "xgb": dict(eta=[0.1, 0.05], depth=[4, 6, 8], mcw=[1, 5, 20], sub=[0.7, 0.8, 1.0], col=[0.5, 0.7, 0.9], lam=[1.0, 5.0, 20.0]),
             "nn": dict(layers=["256,128", "512,256", "1024,512", "512,256,128"], drop=[0.1, 0.25, 0.4], lr=[1e-3, 2e-3, 4e-3],
                        wd=[1e-5, 1e-4, 1e-3], batch=[512, 1024, 2048], epochs=[8, 14, 20], emb=[4, 8, 16])}
    for m, sp in space.items():
        while len(C[m]) < 10:
            c = {k: rng.choice(v) for k, v in sp.items()}
            if c not in C[m]:
                C[m].append(c)
    json.dump(C, open(CFG, "w"), indent=1)
    return C


def append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def is_done(model, cid, fold, seed):
    if not os.path.exists(DONE):
        return False
    d = pd.read_csv(DONE)
    return bool(((d.model == model) & (d.cfg == cid) & (d.fold == fold) & (d.seed == seed)).any())


class Data:
    """Loaded once per call; per-fold arrays built on demand."""
    def __init__(self):
        self.df = load("train")
        self.X = matrix(self.df)
        self.Y = self.df[LABELS].to_numpy("int8", na_value=0)
        self.tm = self.df.target_month_idx.to_numpy()

    def fold(self, V):
        tr = np.flatnonzero(self.tm < V); va = np.flatnonzero(self.tm == V)
        ri, ci = np.nonzero(self.Y[tr])
        act, own = actual_and_owned(self.df.iloc[va])
        return tr, va, tr[ri], ci, act, own


def run_tree(model, cid, cfg, V, seed, D, deadline, max_iter=None):
    tag = f"{model}_{cid}_{V}_{seed}"; path = os.path.join(CK, tag + (".txt" if model == "lgbm" else ".json"))
    stp = os.path.join(CK, tag + ".state")
    lr = cfg["lr"] if model == "lgbm" else cfg["eta"]
    top, step = (LGB_MAX if model == "lgbm" else XGB_MAX)[lr]
    if max_iter: top = max_iter
    it = json.load(open(stp))["iter"] if os.path.exists(stp) and os.path.exists(path) else 0
    tr, va, rows, y, act, own = D.fold(V)
    Xtr, Xva = D.X[rows], D.X[va]
    t0 = time.time()
    if model == "lgbm":
        import lightgbm as lgb
        p = {**PARAMS, "learning_rate": lr, "num_leaves": cfg["num_leaves"], "min_child_samples": cfg["min_child"],
             "lambda_l2": cfg["l2"], "feature_fraction": cfg["ff"], "bagging_fraction": cfg["bag"],
             "bagging_freq": 1 if cfg["bag"] < 1 else 0, "seed": seed, "num_threads": THREADS}
        dtr = lgb.Dataset(Xtr, y, categorical_feature=cat_idx(), free_raw_data=False)
        b = path if it else None
    else:
        import xgboost as xgb
        p = dict(objective="multi:softprob", num_class=24, eta=lr, max_depth=cfg["depth"], min_child_weight=cfg["mcw"],
                 subsample=cfg["sub"], colsample_bytree=cfg["col"], reg_lambda=cfg["lam"], tree_method="hist",
                 max_bin=127, nthread=THREADS, seed=seed)
        dtr = xgb.DMatrix(Xtr, label=y); dva = xgb.DMatrix(Xva)
        b = None
        if it:
            b = xgb.Booster(); b.load_model(path)
    if it >= top:
        return True
    # one continuous training call per process; a callback scores + saves at every checkpoint and stops when the
    # next chunk would not fit in the time budget (resuming re-scores the saved trees only once per call)
    st = {"t": time.time(), "stopped": False}
    def checkpoint(model_obj, g, pred):
        m = evaluate(pred, own, act)[0]
        append(RES, dict(model=model, cfg=cid, seed=seed, fold=V, iter=g, MAP7=m))
        model_obj.save_model(path); json.dump({"iter": g}, open(stp, "w"))
        took = time.time() - st["t"]; st["t"] = time.time()
        print(f"  {tag} iter {g}/{top} MAP@7 {m:.4f} ({took:.0f}s)", flush=True)
        if g < top and time.time() + took * 1.2 > deadline:
            st["stopped"] = True
            return True
        return False
    if model == "lgbm":
        def cb(env):
            g = env.model.current_iteration()
            if g % step == 0 or g == top:
                if checkpoint(env.model, g, env.model.predict(Xva)):
                    raise lgb.callback.EarlyStopException(env.iteration, [])
        lgb.train(p, dtr, num_boost_round=top - it, init_model=b, callbacks=[cb])
    else:
        class CB(xgb.callback.TrainingCallback):
            def after_iteration(self, bst, epoch, evals_log):
                g = bst.num_boosted_rounds()
                if g % step == 0 or g == top:
                    return checkpoint(bst, g, bst.predict(dva))
                return False
        xgb.train(p, dtr, num_boost_round=top - it, xgb_model=b, callbacks=[CB()])
    return not st["stopped"]


def run_nn(cid, cfg, V, seed, D, deadline):
    import torch, torch.nn as nn
    torch.set_num_threads(THREADS)
    tag = f"nn_{cid}_{V}_{seed}"; ck = os.path.join(CK, tag + ".pt")
    maps = json.load(open(os.path.join(PROC, "category_maps.json")))
    def card_of(c):
        if c in maps: return len(maps[c])
        if c.startswith("segment_"): return len(maps["segment"])
        if c.startswith("relation_type_"): return len(maps["relation_type_month_start"])
        return {"province_code": 53, "join_channel": 16}[c]
    card = {c: card_of(c) for c in FINAL_CATS}
    num = [i for i, c in enumerate(FINAL_FEATURES) if c not in FINAL_CATS]
    cat = [FINAL_FEATURES.index(c) for c in FINAL_CATS]
    tr, va, rows, y, act, own = D.fold(V)
    Xn_tr = D.X[tr][:, num]
    med = np.nanmedian(Xn_tr, axis=0)                                   # statistics from this fold's training months only
    filled = np.where(np.isnan(Xn_tr), med[None, :], Xn_tr); mu, sd = filled.mean(0), filled.std(0) + 1e-6
    del filled
    miss_cols = np.flatnonzero(np.isnan(Xn_tr).mean(0) > 0.001)        # missing indicators for columns that are ever missing
    def encode(Xs):
        Xn = Xs[:, num]; nanm = np.isnan(Xn)
        xn = (np.where(nanm, med[None, :], Xn) - mu) / sd
        xn = np.concatenate([xn, nanm[:, miss_cols].astype("float32")], 1).astype("float32")
        C = Xs[:, cat]; xc = np.empty(C.shape, dtype="int64")
        for k, c in enumerate(FINAL_CATS):
            v = C[:, k]; ok = ~np.isnan(v) & (v >= 0) & (v < card[c])
            xc[:, k] = np.where(ok, np.nan_to_num(v), card[c]).astype("int64")
        return torch.from_numpy(xn), torch.from_numpy(xc)
    Xn, Xc = encode(D.X[rows]); Y = torch.from_numpy(y.astype("int64")); Vn, Vc = encode(D.X[va])
    torch.manual_seed(seed); np.random.seed(seed)
    layers = [int(h) for h in cfg["layers"].split(",")]
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.ModuleList([nn.Embedding(card[c] + 1, min(cfg["emb"], max(2, card[c] // 2))) for c in FINAL_CATS])
            d = Xn.shape[1] + sum(e.embedding_dim for e in s.emb); mods = []
            for h in layers:
                mods += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(cfg["drop"])]; d = h
            s.mlp = nn.Sequential(*mods, nn.Linear(d, 24))
        def forward(s, xn, xc):
            return s.mlp(torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(s.emb)], 1))
    net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    bs = cfg["batch"]; spe = (len(Y) + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * spe, pct_start=0.15)
    ep = 0
    if os.path.exists(ck):
        c = torch.load(ck, weights_only=False); net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"]); ep = c["epoch"]; torch.set_rng_state(c["rng"])
    est = 15.0
    while ep < cfg["epochs"]:
        if time.time() + est > deadline:
            return False
        s0 = time.time(); net.train(); perm = torch.randperm(len(Y))
        for i in range(0, len(Y), bs):
            b = perm[i:i + bs]
            if len(b) < 2: continue
            loss = nn.functional.cross_entropy(net(Xn[b], Xc[b]), Y[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ep += 1
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, rng=torch.get_rng_state()), ck)
        est = (time.time() - s0) * 1.2
        print(f"  {tag} epoch {ep}/{cfg['epochs']} ({time.time()-s0:.0f}s)", flush=True)
    net.eval()
    with torch.no_grad():
        pred = torch.cat([torch.softmax(net(Vn[i:i + 8192], Vc[i:i + 8192]), 1) for i in range(0, len(Vn), 8192)]).numpy()
    m = evaluate(pred, own, act)[0]
    append(RES, dict(model="nn", cfg=cid, seed=seed, fold=V, iter=cfg["epochs"], MAP7=m))
    print(f"  {tag} MAP@7 {m:.4f}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--budget", type=float, default=140)
    ap.add_argument("--only", default=None); a = ap.parse_args()
    if a.cmd == "init":
        C = make_configs(); [print(m, i, c) for m in C for i, c in enumerate(C[m])]; return
    T0 = time.time(); deadline = T0 + a.budget
    C = json.load(open(CFG))
    units = [(m, i, V, 1, None) for m in ("lgbm", "xgb", "nn") for i in range(N_CFG) for V in FOLDS]
    if os.path.exists(CONF):
        units += [(u["model"], u["cfg"], V, 2, u.get("iter")) for u in json.load(open(CONF)) for V in FOLDS]
    if a.only:
        units = [u for u in units if u[0] == a.only]
    todo = [u for u in units if not is_done(u[0], u[1], u[2], u[3])]
    print(f"{len(todo)} units left", flush=True)
    if not todo: return
    D = Data(); print(f"data loaded {time.time()-T0:.0f}s", flush=True)
    for m, i, V, sd, mx in todo:
        if time.time() > deadline - 25: break
        print(f"{m} cfg{i} fold{V} seed{sd}: {C[m][i]}", flush=True)
        ok = run_nn(i, C[m][i], V, sd, D, deadline) if m == "nn" else run_tree(m, i, C[m][i], V, sd, D, deadline, mx)
        if not ok:
            print("  paused (budget)"); break
        append(DONE, dict(model=m, cfg=i, fold=V, seed=sd))
    print(f"call time {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
