"""Independent checks of submission/submission.csv against the RAW data (not the feature tables)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
from santander.column_map import COLUMN_MAP
SP = [k for k in COLUMN_MAP if k.endswith("_ult1")]
sub = pd.read_csv("submission/submission.csv"); smp = pd.read_csv("sample_submission.csv.zip")
print("rows", len(sub), "| columns", list(sub.columns), "| same columns as sample:", list(sub.columns) == list(smp.columns))
print("same customer set as sample:", set(sub.ncodpers) == set(smp.ncodpers), "| duplicates:", sub.ncodpers.duplicated().sum())
lists = sub.added_products.fillna("").str.split()
print("products per row:", lists.str.len().value_counts().sort_index().to_dict())
print("unknown product names:", sum(p not in SP for l in lists for p in l), "| duplicate product within a row:", sum(len(set(l)) != len(l) for l in lists))
# owned at 2016-05 straight from the cleaned train file (Spanish names via COLUMN_MAP)
own = pq.read_table("processed/train_clean.parquet", columns=["customer_id"] + [COLUMN_MAP[k] for k in SP], filters=[("month_idx", "=", 16)]).to_pandas().set_index("customer_id")
own.columns = SP
m = sub.set_index("ncodpers").join(own, how="left")
print("test customers present in 2016-05:", m[SP[0]].notna().mean().round(4))
bad = sum(int(m.loc[cid, p] == 1) for cid, l in zip(sub.ncodpers, lists) for p in l if cid in own.index)
print("recommended products already held at 2016-05 (must be 0):", bad)
first = lists.str[0].value_counts(normalize=True).head(8).round(3); print("\nshare of customers by FIRST recommended product:"); print(first.rename(index=lambda k: COLUMN_MAP[k]).to_string())
