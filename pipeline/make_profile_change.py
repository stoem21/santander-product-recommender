"""Profile-change features, one row per row of features_train.parquet / features_test.parquet (same order).
Anchor month s = t-1. Only months >= 7 (2015-08, after the frozen 2015-01..07 block) are used; a transition between months
j-1 and j is 'known' only if the customer has a non-null value in both months and both months are >= 7.
Value <-> null is NOT a change.
 A (summary):  activity_since_change, activity_change_dir, relation_since_change, relation_change_dir, segment_since_change,
               segment_prev, rare_since_change (customer_type / province / primary combined), n_profile_changes_5m
               since_change: 1..5 = last change k months ago (k=1: between s-1 and s), 6 = no change in 5 known transitions,
               NaN = no change seen before the first unknown transition (cannot tell).
 B (raw lags): activity_index_plag{2..5}, segment_plag{2..5}, relation_type_plag{2..5}  (value at s-1..s-4; current value is already a feature)"""
import json, time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
T0 = time.time(); log = lambda m: print(f"[{time.time()-T0:5.0f}s] {m}", flush=True)
P = "processed"; maps = json.load(open(f"{P}/category_maps.json"))
COLS = ["activity_index", "segment", "relation_type_month_start", "customer_type_month_start", "province_code", "primary_customer_index"]
t = pq.read_table(f"{P}/train_clean.parquet", columns=["customer_id", "month_idx"] + COLS, filters=[("month_idx", ">=", 7)])
key = t.column("customer_id").to_numpy().astype("int64") * 32 + t.column("month_idx").to_numpy().astype("int64")
V = {}
for c in COLS:
    col = t.column(c).combine_chunks()
    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
        mp = maps[c]; e = pc.dictionary_encode(col); dic = e.dictionary.to_pylist()
        lut = np.array([mp.get(x, np.nan) if x is not None else np.nan for x in dic] + [np.nan], dtype="float64")
        idx = e.indices.fill_null(len(dic)).to_numpy(zero_copy_only=False); V[c] = lut[idx]
    else:
        V[c] = pc.cast(col, "float64").to_numpy(zero_copy_only=False)
del t
o = np.argsort(key); key = key[o]; V = {c: v[o] for c, v in V.items()}
log(f"profile rows months>=7: {len(key):,}")

def lookup(anchor, k):
    q = anchor - k; p = np.minimum(np.searchsorted(key, q), len(key) - 1); ok = key[p] == q
    return {c: np.where(ok, V[c][p], np.nan) for c in COLS}

def since(vals, c, n):
    """months since last change of column c; also returns value before change and value after change"""
    out = np.full(n, np.nan); prev = np.full(n, np.nan); new = np.full(n, np.nan)
    open_ = np.ones(n, bool)                                     # still scanning (no change found, no unknown hit)
    for k in range(1, 6):
        a, b = vals[k - 1][c], vals[k][c]; known = ~np.isnan(a) & ~np.isnan(b)
        hit = open_ & known & (a != b)
        out[hit] = k; prev[hit] = b[hit]; new[hit] = a[hit]
        open_ &= known & ~hit
    out[open_] = 6                                               # all 5 transitions known, none changed
    return out, prev, new

def build(cust, s):
    anchor = cust.astype("int64") * 32 + s.astype("int64"); n = len(anchor)
    vals = [lookup(anchor, k) for k in range(6)]                 # month s, s-1, ..., s-5
    f = {}
    for c, short in [("activity_index", "activity"), ("relation_type_month_start", "relation"), ("segment", "segment")]:
        sc, pv, nv = since(vals, c, n); f[f"{short}_since_change"] = sc
        if short == "activity":
            f["activity_change_dir"] = np.where(np.isnan(nv), np.where(sc == 6, 0, np.nan), np.where(nv > pv, 1, -1))
        if short == "relation":                                   # +1 became A (active), -1 became I (inactive), 2 other
            A_, I_ = maps[c]["A"], maps[c]["I"]
            f["relation_change_dir"] = np.where(np.isnan(nv), np.where(sc == 6, 0, np.nan), np.where(nv == A_, 1, np.where(nv == I_, -1, 2)))
        if short == "segment":
            f["segment_prev"] = pv
    rare = [since(vals, c, n)[0] for c in ["customer_type_month_start", "province_code", "primary_customer_index"]]
    r = np.vstack(rare); f["rare_since_change"] = np.where(np.isnan(r).all(0), np.nan, np.nanmin(np.where(np.isnan(r), 99, r), 0))
    f["rare_since_change"] = np.where(f["rare_since_change"] == 99, np.nan, f["rare_since_change"])
    cnt = np.zeros(n); allk = np.ones(n, bool)
    for c in COLS:
        for k in range(1, 6):
            a, b = vals[k - 1][c], vals[k][c]; known = ~np.isnan(a) & ~np.isnan(b); cnt += known & (a != b)
            if c == "activity_index": allk &= known
    f["n_profile_changes_5m"] = np.where(allk, cnt, np.nan)
    for c, short in [("activity_index", "activity_index"), ("segment", "segment"), ("relation_type_month_start", "relation_type")]:
        for k in range(1, 5):
            f[f"{short}_plag{k + 1}"] = vals[k][c]
    return pd.DataFrame({k: v.astype("float32") for k, v in f.items()})

for name in ("train", "test"):
    ft = pd.read_parquet(f"{P}/features_{name}.parquet", columns=["customer_id", "target_month_idx"])
    out = build(ft.customer_id.to_numpy(), ft.target_month_idx.to_numpy() - 1)
    out.insert(0, "customer_id", ft.customer_id.to_numpy()); out.insert(1, "target_month_idx", ft.target_month_idx.to_numpy())
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), f"{P}/profile_change_{name}.parquet", compression="zstd")
    log(f"{name}: {out.shape}")
A = ["activity_since_change", "activity_change_dir", "relation_since_change", "relation_change_dir", "segment_since_change", "segment_prev", "rare_since_change", "n_profile_changes_5m"]
B = [f"{c}_plag{k}" for c in ("activity_index", "segment", "relation_type") for k in range(2, 6)]
json.dump({"A": A, "B": B}, open(f"{P}/profile_change_columns.json", "w"), indent=1)
log("done")
