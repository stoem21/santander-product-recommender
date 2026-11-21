"""TreeSHAP (LightGBM pred_contrib) for the final model on a random sample of 8,000 test customers (the real 2016-06 recommendations).
SHAP values are in raw-score (log-odds-like) units per product class; not probabilities, not causal effects."""
import re, numpy as np, pandas as pd, lightgbm as lgb
from santander.column_map import PRODUCT_COLUMNS as PC
from santander.final_features import *
OUT = "analysis_output/xai"; b = lgb.Booster(model_file="models/final_lgbm.txt")
te = load("test"); rs = np.random.default_rng(0).choice(len(te), 8000, replace=False); te = te.iloc[np.sort(rs)].reset_index(drop=True)
X = matrix(te); F = len(FINAL_FEATURES); NPc = len(PC)
C = b.predict(X, pred_contrib=True).reshape(len(X), NPc, F + 1)[:, :, :F]            # n, class, feature (bias dropped)
P = b.predict(X); owned = te[[f"{p}_lag1" for p in PC]].to_numpy("float32", na_value=np.nan) == 1
S = np.where(owned, -np.inf, P); order = np.argsort(-S, 1, kind="stable"); top1, top2 = order[:, 0], order[:, 1]
np.save(f"{OUT}/shap_sample.npy", C.astype("float32")); te[["customer_id"]].to_csv(f"{OUT}/shap_sample_ids.csv", index=False)

group = lambda f, j: feature_group(f, PC[j], split_eligibility=False)   # santander.final_features
groups = sorted({group(f, 0) for f in FINAL_FEATURES})
G = np.zeros((NPc, F), dtype=object)
for j in range(NPc):
    for k, f in enumerate(FINAL_FEATURES): G[j, k] = group(f, j)
# 1) what drives the recommended (top-1) product: share of |SHAP| by group
rows = []
A = np.abs(C[np.arange(len(X)), top1])                                                  # n, F  for the recommended class
tot = A.sum(1, keepdims=True)
for g in groups:
    msk = np.stack([G[j] == g for j in top1]); rows.append((g, float((A * msk).sum() / tot.sum())))
glob = pd.DataFrame(rows, columns=["feature_group", "share_of_abs_SHAP_for_top1"]).sort_values("share_of_abs_SHAP_for_top1", ascending=False)
glob.to_csv(f"{OUT}/global_groups_top1.csv", index=False); print("== what drives the TOP-1 recommendation (share of |SHAP|) =="); print(glob.round(3).to_string(index=False))
# 2) per product: top features by mean |SHAP| among customers who do NOT hold it
keyp = ["direct_debit", "current_account", "pensions", "payroll", "e_account", "credit_card", "taxes"]
out = []
for p in keyp:
    j = PC.index(p); m = ~owned[:, j]; a = np.abs(C[m, j]).mean(0); top = np.argsort(-a)[:6]
    out.append({"product": p, "top_features(mean|SHAP|)": ", ".join(f"{FINAL_FEATURES[k]} {a[k]:.2f}" for k in top)})
pp = pd.DataFrame(out); pp.to_csv(f"{OUT}/per_product_top_features.csv", index=False); print("\n== per product top features =="); [print(f"{r['product']:16s} {r['top_features(mean|SHAP|)']}") for _, r in pp.iterrows()]
# 3) month_of_year is constant (June) in test: its SHAP = how much 'June' shifts each product
k = FINAL_FEATURES.index("month_of_year"); mj = C[:, :, k].mean(0)
print("\n== effect of month_of_year=June on raw score, by product (mean SHAP) =="); print(pd.Series(mj, index=PC).sort_values(ascending=False).round(3).head(6).to_string()); print("...", pd.Series(mj, index=PC).sort_values().round(3).head(4).to_dict())
pd.Series(mj, index=PC).to_csv(f"{OUT}/june_effect_by_product.csv")
np.save(f"{OUT}/top12.npy", np.stack([top1, top2], 1)); np.save(f"{OUT}/probs_sample.npy", P.astype("float32"))
