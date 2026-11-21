"""Extra models for the model-comparison table, on the SAME setup as tune.py:
final 175 features, 4 rolling-origin folds (validate on target months 13..16, train on all earlier ones),
expanded adder rows (one row per added product), and train_common.evaluate() (MAP@7 / recall@7).

  logreg : multinomial logistic regression. A simple, interpretable step between the popularity baseline and the trees.
           Numerics: median impute (training fold only) + missing-value flags + standardise. Categoricals: one-hot.
           One fixed setting (C=1.0), not tuned.
  cat    : CatBoost MultiClass, one default-style config (lr 0.1, depth 6, l2 5), not tuned.
           one_hot_max_size=64: every categorical here has <= 53 levels, so all are one-hot encoded instead of target
           statistics (CTR). With 24 classes the CTRs made training ~6 s/round even on 20k rows.
           MAP@7 recorded every --step rounds; the number of rounds is chosen later from the 4-fold mean, like tune.py.

Results are appended to analysis_output/models/extra_models.csv (one row per model / fold / iter).
It never touches the tuning files. It refuses to run while tune.py is running (the machine is busy) unless --force.

Usage:  python -m experiments.compare_extra logreg|cat [--folds 13 14 15 16] [--threads N] [--rounds 1000] [--step 50]
        python -m experiments.compare_extra logreg|cat --smoke     # tiny subsample, 1 thread, writes to a temp file
Needs scikit-learn / catboost (installed in .venv_extra, kept separate from the tuning environment).
"""
import argparse, os, subprocess, sys, tempfile, time
import numpy as np, pandas as pd
from santander.train_common import LABELS, actual_and_owned, evaluate, HERE
from santander.final_features import FINAL_FEATURES, FINAL_CATS, load, matrix

OUT = os.path.join(HERE, "analysis_output", "models", "extra_models.csv")
FOLDS = [13, 14, 15, 16]
NC = len(LABELS)
CAT_IDX = [FINAL_FEATURES.index(c) for c in FINAL_CATS]
NUM_IDX = [i for i in range(len(FINAL_FEATURES)) if i not in CAT_IDX]


def tuning_running():
    # anchored: a plain "tune.py run" also matches any shell whose command line merely mentions it (e.g. a watcher loop)
    return subprocess.run(["pgrep", "-f", "^(caffeinate -i )?python3? -m experiments.tune run"], capture_output=True).returncode == 0


def full_proba(p, classes):
    """Classes absent from a training fold (e.g. guarantees: 4 adds in total) get probability 0."""
    out = np.zeros((p.shape[0], NC), "float32")
    out[:, np.asarray(classes, dtype=int)] = p
    return out


def run_logreg(Xtr, y, Xva, threads, record):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from threadpoolctl import threadpool_limits
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM_IDX),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value=-1),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, dtype=np.float32)), CAT_IDX)])
    lr = LogisticRegression(C=1.0, max_iter=3000, tol=1e-4)
    model = make_pipeline(pre, lr)
    t0 = time.time()
    with threadpool_limits(threads):
        model.fit(Xtr, y)
        p = model.predict_proba(Xva)
    secs = time.time() - t0
    n_iter = int(np.max(lr.n_iter_))
    if n_iter >= lr.max_iter:
        print(f"  WARNING: logistic regression hit max_iter={lr.max_iter} (not fully converged)", flush=True)
    record(n_iter, full_proba(p, model.classes_), secs)


def run_cat(Xtr, y, Xva, threads, rounds, step, record):
    from catboost import CatBoostClassifier, Pool
    def prep(M):
        d = pd.DataFrame(M, columns=FINAL_FEATURES)
        for c in FINAL_CATS:
            d[c] = d[c].fillna(-1).astype("int32")
        return d
    m = CatBoostClassifier(loss_function="MultiClass", iterations=rounds, learning_rate=0.1, depth=6, l2_leaf_reg=5,
                           border_count=128, one_hot_max_size=64, thread_count=threads, random_seed=1, verbose=step, allow_writing_files=False)
    t0 = time.time()
    m.fit(Pool(prep(Xtr), y, cat_features=FINAL_CATS))
    secs = time.time() - t0
    dva = Pool(prep(Xva), cat_features=FINAL_CATS)
    for k, p in enumerate(m.staged_predict_proba(dva, eval_period=step), start=1):
        record(min(k * step, rounds), full_proba(p, m.classes_), secs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["logreg", "cat"])
    ap.add_argument("--folds", type=int, nargs="+", default=FOLDS)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TUNE_THREADS", "4")))
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--smoke", action="store_true", help="20k training rows, 3k validation rows, 1 thread, temp output")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="run even while tune.py is running")
    a = ap.parse_args()

    if a.smoke:
        a.threads, a.folds = 1, [16]
        a.rounds, a.step = min(a.rounds, 20), min(a.step, 10)
    out = a.out or (os.path.join(tempfile.gettempdir(), "extra_models_smoke.csv") if a.smoke else OUT)
    if tuning_running() and not (a.smoke or a.force):
        sys.exit("tune.py is running - not starting a full run (would compete for the CPU). Use --smoke or wait.")

    done = set()
    if os.path.exists(out) and not a.smoke:
        prev = pd.read_csv(out)
        done = set(prev.loc[prev.model == a.model, "fold"])

    T0 = time.time()
    df = load("train"); X = matrix(df)
    Y = df[LABELS].to_numpy("int8", na_value=0); tm = df.target_month_idx.to_numpy()
    print(f"data loaded {time.time()-T0:.0f}s, threads {a.threads}", flush=True)
    rng = np.random.default_rng(0)
    for V in a.folds:
        if V in done:
            print(f"{a.model} fold{V}: already in {out}, skipped", flush=True); continue
        tr = np.flatnonzero(tm < V); va = np.flatnonzero(tm == V)
        ri, ci = np.nonzero(Y[tr]); rows = tr[ri]
        if a.smoke:
            k = rng.choice(len(rows), min(20000, len(rows)), replace=False); rows, ci = rows[k], ci[k]
            va = va[:3000]
        act, own = actual_and_owned(df.iloc[va])
        print(f"{a.model} fold{V}: train rows {len(rows):,}, validation customers {len(va):,}", flush=True)

        def record(it, scores, secs):
            m, r = evaluate(scores, own, act)
            row = dict(model=a.model, fold=V, iter=it, MAP7=m, recall7=r, train_seconds=round(secs, 1),
                       train_rows=len(rows), threads=a.threads, smoke=a.smoke)
            pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"  {a.model} fold{V} iter {it}: MAP@7 {m:.4f} recall@7 {r:.4f} (train {secs:.0f}s)", flush=True)

        if a.model == "logreg":
            run_logreg(X[rows], ci, X[va], a.threads, record)
        else:
            run_cat(X[rows], ci, X[va], a.threads, a.rounds, a.step, record)
    print(f"results -> {out}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
