"""Follow-up to xai_shap.py (version 1 model, same 8,000 customers): SHAP shares with the eligibility feature split out,
counterfactual flips of sex / foreigner / age / n_prior_capped, and example customer explanations."""
import re, numpy as np, pandas as pd, lightgbm as lgb
from santander.column_map import PRODUCT_COLUMNS as PC
from santander.final_features import *
from santander.train_common import top_k
OUT = "analysis_output/xai"; b = lgb.Booster(model_file="models/final_lgbm.txt")
C = np.load(f"{OUT}/shap_sample.npy"); ids = pd.read_csv(f"{OUT}/shap_sample_ids.csv").customer_id.to_numpy()
te = load("test"); te = te.set_index("customer_id").loc[ids].reset_index(); X = matrix(te); F = len(FINAL_FEATURES)
P = np.load(f"{OUT}/probs_sample.npy"); t12 = np.load(f"{OUT}/top12.npy"); top1, top2 = t12[:, 0], t12[:, 1]
owned = te[[f"{p}_lag1" for p in PC]].to_numpy("float32", na_value=np.nan) == 1
group = lambda f, j: feature_group(f, PC[j])   # santander.final_features
A = np.abs(C[np.arange(len(X)), top1]); tot = A.sum()
gs = {}
for i, j in enumerate(top1):
    for k, f in enumerate(FINAL_FEATURES): gs[group(f, j)] = gs.get(group(f, j), 0) + A[i, k]
g = pd.Series(gs) / tot; g = g.sort_values(ascending=False); g.to_csv(f"{OUT}/global_groups_top1_split.csv")
print("== drivers of the TOP-1 recommendation, eligibility split out =="); print(g.round(3).to_string())
ex = g.drop(ELIGIBILITY); print("\nexcluding eligibility:"); print((ex / ex.sum()).round(3).to_string())

# --- sensitive attributes: counterfactual flip of sex, and of foreigner_index
print("\n== counterfactual: change only this feature, how many customers get a different top-1 / top-7 set? ==")
base_top7 = top_k(P, owned)
def flip(col, fn):
    X2 = X.copy(); k = FINAL_FEATURES.index(col); X2[:, k] = fn(X2[:, k]); P2 = b.predict(X2)
    t7 = top_k(P2, owned)
    same_set = np.array([set(a) == set(c) for a, c in zip(base_top7, t7)])
    return (t7[:, 0] != base_top7[:, 0]).mean(), 1 - same_set.mean(), np.abs(P2 - P).max(1).mean()
for col, fn, lab in [("sex", lambda v: np.where(v == 0, 1, np.where(v == 1, 0, v)), "sex H<->V"),
                     ("foreigner_index", lambda v: np.where(v == 0, 1, np.where(v == 1, 0, v)), "foreigner N<->S"),
                     ("age", lambda v: v + 10, "age +10 years"), ("n_prior_capped", lambda v: np.full_like(v, 6), "n_prior_capped -> 6")]:
    a1, a7, dp = flip(col, fn); print(f"{lab:22s} top-1 changes {a1*100:5.2f}%   top-7 set changes {a7*100:5.2f}%   mean max |dprob| {dp:.4f}")
print("n_prior_capped values in sample:", pd.Series(te.n_prior_capped).value_counts().sort_index().to_dict())

# --- customer cards
def card(i):
    j, j2 = top1[i], top2[i]; r = te.iloc[i]
    s = C[i, j]; k = np.argsort(-np.abs(s))[:6]
    d = C[i, j] - C[i, j2]; kd = np.argsort(-d)[:4]
    val = lambda f: X[i, FINAL_FEATURES.index(f)]
    held = [p for p in PC if r[f"{p}_lag1"] == 1]
    lines = [f"### customer {int(r.customer_id)}", f"- holds now: {', '.join(held) if held else 'nothing'} | age {val('age'):.0f}, tenure {val('tenure_months'):.0f} months, segment code {val('segment'):.0f}, activity_index {val('activity_index'):.0f}",
             f"- recommended #1: **{PC[j]}** (p={P[i, j]:.2f}), #2: {PC[j2]} (p={P[i, j2]:.2f})",
             "- strongest contributions to #1 (feature = value: SHAP): " + "; ".join(f"{FINAL_FEATURES[q]} = {val(FINAL_FEATURES[q]):g}: {s[q]:+.2f}" for q in k),
             f"- why {PC[j]} ranks above {PC[j2]} (largest SHAP differences): " + "; ".join(f"{FINAL_FEATURES[q]} = {val(FINAL_FEATURES[q]):g}: {d[q]:+.2f}" for q in kd)]
    return "\n".join(lines)
picks = []
for p in ["e_account", "credit_card", "taxes", "pensions"]:
    c = np.flatnonzero(top1 == PC.index(p)); picks.append(c[len(c) // 2] if len(c) else None)
txt = "\n\n".join(card(i) for i in picks if i is not None)
open(f"{OUT}/customer_cards.md", "w").write("# Example explanations (final model, 2016-06 recommendations)\nSHAP in raw-score units; positive pushes the product up. Not causal.\n\n" + txt + "\n")
print("\n" + txt)
