"""Do the models 'think alike'? Same 5,000 test customers, three models on the same 175 features:
lgbm (final, seed 1), lgbm_s2 (seed 2 = reference for how much explanations move by chance), xgb.
SHAP via exact TreeSHAP in each library."""
import re, time, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from santander.column_map import PRODUCT_COLUMNS as PC
from santander.final_features import *
from santander.train_common import top_k
T0 = time.time(); OUT = "analysis_output/xai"
ids = pd.read_csv(f"{OUT}/shap_sample_ids.csv").customer_id.to_numpy()[:5000]
te = load("test").set_index("customer_id").loc[ids].reset_index(); X = matrix(te); F = len(FINAL_FEATURES)
owned = te[[f"{p}_lag1" for p in PC]].to_numpy("float32", na_value=np.nan) == 1
L1, L2 = lgb.Booster(model_file="models/final_lgbm.txt"), lgb.Booster(model_file="models/compare/lgbm_s2.txt")
XB = xgb.Booster(); XB.load_model("models/compare/xgb.json"); DX = xgb.DMatrix(X, feature_names=FINAL_FEATURES)
M = {"lgbm": (L1.predict(X), L1.predict(X, pred_contrib=True).reshape(len(X), 24, F + 1)[:, :, :F]),
     "lgbm_s2": (L2.predict(X), L2.predict(X, pred_contrib=True).reshape(len(X), 24, F + 1)[:, :, :F]),
     "xgb": (XB.predict(DX), XB.predict(DX, pred_contribs=True)[:, :, :F])}
print(f"SHAP computed in {time.time()-T0:.0f}s", flush=True)
for k, (p, c) in M.items():                                      # SHAP must add up to the raw score
    assert c.shape == (len(X), 24, F), (k, c.shape)
top = {k: top_k(p, owned) for k, (p, c) in M.items()}
ref = top["lgbm"][:, 0]                                           # explain the SAME product (final model's #1) in every model
grp = lambda f, j: feature_group(f, PC[j])   # santander.final_features
GJ = [[grp(f, j) for f in FINAL_FEATURES] for j in range(24)]
rows = {}
for k, (p, c) in M.items():
    A = np.abs(c[np.arange(len(X)), ref]); s = {}
    for i, j in enumerate(ref):
        for q, g in enumerate(GJ[j]): s[g] = s.get(g, 0) + A[i, q]
    s.pop(ELIGIBILITY); t = sum(s.values()); rows[k] = {g: v / t for g, v in s.items()}
G = pd.DataFrame(rows).sort_values("lgbm", ascending=False); print("\n== share of |SHAP| for the final model's #1 product, eligibility excluded =="); print(G.round(3).to_string())
print("\n== agreement of recommendations ==")
for a_, b_ in [("lgbm", "lgbm_s2"), ("lgbm", "xgb"), ("lgbm_s2", "xgb")]:
    same1 = (top[a_][:, 0] == top[b_][:, 0]).mean(); jac = np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(top[a_], top[b_])])
    print(f"{a_:8s} vs {b_:8s}: same #1 {same1*100:5.1f}%   top-7 Jaccard {jac:.3f}")
print("\n== per-customer explanation overlap: top-5 features by |SHAP| for the same product ==")
T5 = {k: np.argsort(-np.abs(c[np.arange(len(X)), ref]), 1)[:, :5] for k, (p, c) in M.items()}
for a_, b_ in [("lgbm", "lgbm_s2"), ("lgbm", "xgb")]:
    ov = np.mean([len(set(x) & set(y)) / 5 for x, y in zip(T5[a_], T5[b_])]); first = (T5[a_][:, 0] == T5[b_][:, 0]).mean()
    print(f"{a_:8s} vs {b_:8s}: mean overlap {ov*100:5.1f}%   same #1 feature {first*100:5.1f}%")
print("\n== feature-level agreement: Spearman of mean |SHAP| per (product, feature), customers not holding the product ==")
imp = {}
for k, (p, c) in M.items():
    v = np.stack([np.abs(c[~owned[:, j], j]).mean(0) for j in range(24) if (~owned[:, j]).sum() > 0]); imp[k] = v.ravel()
from scipy.stats import spearmanr
for a_, b_ in [("lgbm", "lgbm_s2"), ("lgbm", "xgb")]:
    print(f"{a_:8s} vs {b_:8s}: rho {spearmanr(imp[a_], imp[b_]).correlation:.3f}")
print("\n== June effect (mean SHAP of month_of_year) on key products ==")
k = FINAL_FEATURES.index("month_of_year")
J = pd.DataFrame({m: c[:, :, k].mean(0) for m, (p, c) in M.items()}, index=PC).loc[["taxes", "current_account", "payroll", "long_term_deposit", "securities", "e_account", "credit_card"]]
print(J.round(3).to_string()); G.to_csv(f"{OUT}/compare_groups.csv"); J.to_csv(f"{OUT}/compare_june_effect.csv")
np.save(f"{OUT}/compare_top1.npy", np.stack([top[k][:, 0] for k in M], 1))

print("\n== when #1 feature differs (lgbm vs xgb): which pairs? ==")
diff = T5["lgbm"][:, 0] != T5["xgb"][:, 0]
pairs = pd.Series([f"{grp(FINAL_FEATURES[a], j)}  ->  {grp(FINAL_FEATURES[b], j)}" for a, b, j in zip(T5["lgbm"][diff, 0], T5["xgb"][diff, 0], ref[diff])]).value_counts(normalize=True).head(6)
print(pairs.round(3).to_string())
ex = pd.Series([f"{FINAL_FEATURES[a]} -> {FINAL_FEATURES[b]}" for a, b in zip(T5["lgbm"][diff, 0], T5["xgb"][diff, 0])]).value_counts().head(6); print(ex.to_string())
print("\n#1 recommendation changes only because of the seed for %.1f%% of customers; how close were #1 and #2 in those cases?" % ((top['lgbm'][:,0] != top['lgbm_s2'][:,0]).mean()*100))
p1 = M["lgbm"][0]; srt = np.sort(np.where(owned, -np.inf, p1), 1)[:, ::-1]; gap = srt[:, 0] - srt[:, 1]; ch = top["lgbm"][:, 0] != top["lgbm_s2"][:, 0]
print(f"median gap p(#1)-p(#2): customers whose #1 changed {np.median(gap[ch]):.3f}  vs unchanged {np.median(gap[~ch]):.3f}")
print("share of customers with gap < 0.05:", round((gap < 0.05).mean(), 3))
