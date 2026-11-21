"""Summarise the equal-budget hyperparameter search (tune.py). READ-ONLY on all result files.

Steps:
  1. results.csv may contain duplicate rows after resumes -> keep the LAST row per (model, cfg, seed, fold, iter).
  2. Only units listed in units_done.csv count as finished; rows from unfinished units are reported but not ranked.
  3. Trees (lgbm / xgb): mean MAP@7 over the 4 folds per (cfg, iter) -> best iter per cfg
     (ties -> fewer rounds). NN: a single score per fold at its configured number of epochs.
  4. Rank configs per model by the 4-fold mean at their best iter; compare each with config 0 per fold (paired).
  5. Proposal for confirm.json = config 0 + top-2 of configs 1..9 per model (iter = chosen rounds; ignored for NN).
     Printed only. It is written to disk only with --write-confirm PATH (refuses to overwrite unless --force).

Usage:  python -m experiments.tune_summary [--seed 1] [--csv OUT.csv] [--write-confirm PATH [--force]]
"""
import argparse, json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
TD = os.path.join(HERE, "analysis_output", "tuning")
FOLDS = [13, 14, 15, 16]
MODELS = ["lgbm", "xgb", "nn"]
KEY = ["model", "cfg", "seed", "fold", "iter"]


def load(td):
    res = pd.read_csv(os.path.join(td, "results.csv"))
    n_raw = len(res)
    res = res.drop_duplicates(KEY, keep="last").reset_index(drop=True)
    done = pd.read_csv(os.path.join(td, "units_done.csv")).drop_duplicates()
    cfgs = json.load(open(os.path.join(td, "configs.json")))
    return res, n_raw, done, cfgs


def short(c):
    return " ".join(f"{k}={v}" for k, v in c.items())


def summarise(res, done, cfgs, seed):
    """Returns (ranking table, per-iter curve table, list of partial units)."""
    r = res[res.seed == seed]
    d = done[done.seed == seed]
    fin = r.merge(d[["model", "cfg", "fold"]], on=["model", "cfg", "fold"], how="inner")
    partial = (r.merge(d[["model", "cfg", "fold"]], on=["model", "cfg", "fold"], how="left", indicator=True)
                .query("_merge == 'left_only'").groupby(["model", "cfg", "fold"]).iter.max().reset_index())

    # mean over folds per (model, cfg, iter); only iters present in all 4 finished folds are eligible
    curve = (fin.groupby(["model", "cfg", "iter"]).MAP7.agg(mean="mean", n_folds="count").reset_index())
    rows = []
    for (m, c), g in curve.groupby(["model", "cfg"]):
        full = g[g.n_folds == len(FOLDS)]
        if full.empty:
            continue
        best = full.sort_values(["mean", "iter"], ascending=[False, True]).iloc[0]
        it = int(best["iter"])
        per_fold = fin[(fin.model == m) & (fin.cfg == c) & (fin.iter == it)].set_index("fold").MAP7
        rows.append(dict(model=m, cfg=int(c), best_iter=it, map7=best["mean"],
                         max_iter=int(g.iter.max()), **{f"f{f}": per_fold.get(f) for f in FOLDS},
                         params=short(cfgs[m][int(c)])))
    rank = pd.DataFrame(rows)
    if rank.empty:
        return rank, curve, partial

    # paired comparison with config 0 of the same model (each at its own best iter)
    out = []
    for m, g in rank.groupby("model"):
        g = g.copy()
        base = g[g.cfg == 0]
        fcols = [f"f{f}" for f in FOLDS]
        if len(base):
            b = base.iloc[0]
            g["diff_vs0"] = g.map7 - b.map7
            diffs = g[fcols].to_numpy() - b[fcols].to_numpy(dtype=float)
            g["wins_vs0"] = [f"{int((dd > 0).sum())}/{len(FOLDS)}" for dd in diffs]
            g["min_fold_diff"] = diffs.min(1)
        else:
            g["diff_vs0"] = g["wins_vs0"] = g["min_fold_diff"] = None
        g = g.sort_values("map7", ascending=False)
        g.insert(2, "rank", range(1, len(g) + 1))
        out.append(g)
    rank = pd.concat(out)
    rank["model"] = pd.Categorical(rank.model, MODELS, ordered=True)
    return rank.sort_values(["model", "rank"]).reset_index(drop=True), curve, partial


def proposal(rank, top=2, n_cfg=10):
    conf, missing = [], []
    for m in MODELS:
        g = rank[rank.model == m]
        if len(g) < n_cfg:
            missing.append(f"{m}: {len(g)}/{n_cfg} configs finished")
        b = g[g.cfg == 0]
        if len(b):
            conf.append({"model": m, "cfg": 0, "iter": int(b.iloc[0].best_iter)})
        for _, r in g[g.cfg != 0].head(top).iterrows():
            conf.append({"model": m, "cfg": int(r.cfg), "iter": int(r.best_iter)})
    return conf, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=TD)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--n-cfg", type=int, default=int(os.environ.get("TUNE_N_CFG", "3")),
                    help="configs per model in the equal-budget comparison (cfg 0..n-1); others are listed separately")
    ap.add_argument("--csv", default=None, help="optional path to save the ranking table")
    ap.add_argument("--curve", action="store_true", help="also print the 4-fold mean per iter for tree configs")
    ap.add_argument("--write-confirm", default=None, help="write the proposal to this path (only after approval)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    res, n_raw, done, cfgs = load(a.dir)
    print(f"results.csv: {n_raw} rows, {n_raw - len(res)} duplicates dropped (kept last) -> {len(res)}")
    dn = done[done.seed == a.seed]
    print(f"finished units (seed {a.seed}): {len(dn)}  " +
          "  ".join(f"{m}={int(((dn.model == m) & (dn.cfg < a.n_cfg)).sum())}/{4 * a.n_cfg}" for m in MODELS) +
          f"  (equal-budget scope: cfg 0..{a.n_cfg - 1})")

    rank, curve, partial = summarise(res, done, cfgs, a.seed)
    if len(partial):
        print("\nunfinished units (rows present, not ranked):")
        for _, p in partial.iterrows():
            print(f"  {p.model} cfg{p.cfg} fold{p.fold}: up to iter {p.iter}")
    if rank.empty:
        print("\nno config has all 4 folds finished yet"); return
    extra = rank[rank.cfg >= a.n_cfg]
    rank = rank[rank.cfg < a.n_cfg].copy()
    rank["rank"] = rank.groupby("model", observed=True).map7.rank(ascending=False, method="first").astype(int)
    rank = rank.sort_values(["model", "rank"]).reset_index(drop=True)

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    fmt = {c: "{:.4f}".format for c in ["map7", "f13", "f14", "f15", "f16"]}
    fmt.update({c: "{:+.4f}".format for c in ["diff_vs0", "min_fold_diff"]})
    for m in MODELS:
        g = rank[rank.model == m]
        if g.empty:
            continue
        print(f"\n== {m}: {len(g)}/{a.n_cfg} configs with all 4 folds (MAP@7, seed {a.seed}, each at its best iter)")
        cols = ["rank", "cfg", "best_iter", "max_iter", "map7", "diff_vs0", "wins_vs0", "min_fold_diff",
                "f13", "f14", "f15", "f16", "params"]
        print(g[cols].to_string(index=False, formatters=fmt))
        if a.curve and m != "nn":
            for c in g.cfg:
                cv = curve[(curve.model == m) & (curve.cfg == c) & (curve.n_folds == len(FOLDS))]
                print(f"   cfg{c} curve: " + "  ".join(f"{int(i)}:{v:.4f}" for i, v in zip(cv.iter, cv["mean"])))

    if len(extra):
        print(f"\noutside the equal-budget comparison (cfg >= {a.n_cfg}, run before the scope was reduced; not ranked/proposed):")
        print(extra[["model", "cfg", "best_iter", "map7", "diff_vs0", "wins_vs0", "params"]].to_string(
            index=False, formatters={"map7": "{:.4f}".format, "diff_vs0": "{:+.4f}".format}))

    print("\nnote: picking the best of N configs on these same 4 folds is optimistic (best-of-N);"
          " seed noise ~±0.0005, month-to-month ~±0.006. Seed 2 (phase 2) is the check.")

    conf, missing = proposal(rank, a.top, a.n_cfg)
    print(f"\nproposed confirm.json (config 0 + top-{a.top} of configs 1-{a.n_cfg - 1} per model; iter ignored for nn):")
    print(json.dumps(conf, indent=1))
    if missing:
        print("WARNING: phase 1 incomplete -> proposal is provisional: " + "; ".join(missing))

    if a.csv:
        rank.to_csv(a.csv, index=False); print(f"saved ranking -> {a.csv}")
    if a.write_confirm:
        if missing and not a.force:
            sys.exit("refusing to write: phase 1 incomplete (use --force to override)")
        if os.path.exists(a.write_confirm) and not a.force:
            sys.exit(f"refusing to overwrite {a.write_confirm} (use --force)")
        json.dump(conf, open(a.write_confirm, "w"), indent=1); print(f"wrote {a.write_confirm}")


if __name__ == "__main__":
    main()
