"""Models for the explanation-agreement check, all on the final 175 features and ALL training months:
 lgbm_s2 : LightGBM, same params as final, seed 2   |  xgb : XGBoost multi:softprob, eta 0.1, depth 6, 200 rounds (setup validated earlier)"""
import sys, time, numpy as np, lightgbm as lgb
from santander.train_common import LABELS
from santander.final_features import *
which = sys.argv[1]; T0 = time.time()
tr = load("train"); Y = tr[LABELS].to_numpy("int8", na_value=0); ri, ci = np.nonzero(Y); X = matrix(tr)[ri]
if which == "lgbm_s2":
    b = lgb.train({**PARAMS, "seed": 2}, lgb.Dataset(X, ci, categorical_feature=cat_idx()), num_boost_round=100); b.save_model("models/compare/lgbm_s2.txt")
else:
    import xgboost as xgb
    p = dict(objective="multi:softprob", num_class=24, eta=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
             reg_lambda=5.0, tree_method="hist", max_bin=127, nthread=4, seed=1)
    b = xgb.train(p, xgb.DMatrix(X, label=ci, feature_names=FINAL_FEATURES), num_boost_round=200); b.save_model("models/compare/xgb.json")
print(which, f"{time.time()-T0:.0f}s")
