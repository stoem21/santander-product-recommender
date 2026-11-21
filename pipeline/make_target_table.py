#!/usr/bin/env python3
"""Step 1b: build the target table from processed/train_clean.parquet

Definition:
  unit      : one (customer, month t) pair where the customer ALSO appears in month t-1
  new_<p>   : 1 if product p was 0 at t-1 AND 1 at t, else 0
              (a NULL flag never counts as an add: NULL comparisons are False)
  excluded  : a customer's first month in the file, and a month right after a gap

All months are built in one run (looping over months only to keep memory small);
train/validation is a later filter on month_idx, not separate files.

Run:  python -m pipeline.make_target_table
Out:  processed/target.parquet   one row per eligible pair, one row group per month
      columns: customer_id, snapshot_date, month_idx, new_<product> x24, n_new
"""
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from santander.column_map import PRODUCT_COLUMNS

CLEAN = "processed/train_clean.parquet"
OUT = "processed/target.parquet"
T0 = time.time()
NEW_COLS = [f"new_{c}" for c in PRODUCT_COLUMNS]
READ_COLS = ["customer_id", "snapshot_date"] + PRODUCT_COLUMNS


def label(i):
    return f"{2015 + i // 12}-{i % 12 + 1:02d}"


def read_month(t):
    return pq.read_table(CLEAN, columns=READ_COLS,
                         filters=[("month_idx", "=", t)]).to_pandas()


last_t = int(pq.read_table(CLEAN, columns=["month_idx"]).column(0).to_numpy().max())
prev = read_month(0)
writer, schema = None, None
stats = []
by_product = np.zeros(len(PRODUCT_COLUMNS), dtype=np.int64)

for t in range(1, last_t + 1):
    cur = read_month(t)
    m = cur.merge(prev[["customer_id"] + PRODUCT_COLUMNS], on="customer_id",
                  suffixes=("", "_prev"))
    c = m[PRODUCT_COLUMNS].to_numpy(dtype="float32", na_value=np.nan)
    p = m[[f"{x}_prev" for x in PRODUCT_COLUMNS]].to_numpy(dtype="float32", na_value=np.nan)
    new = ((c == 1) & (p == 0)).astype("int8")
    out = pd.DataFrame(new, columns=NEW_COLS)
    out.insert(0, "month_idx", np.full(len(m), t, dtype="int8"))
    out.insert(0, "snapshot_date", m["snapshot_date"].to_numpy())
    out.insert(0, "customer_id", m["customer_id"].to_numpy(dtype="int32"))
    out["n_new"] = new.sum(axis=1).astype("int8")
    tbl = pa.Table.from_pandas(out, schema=schema, preserve_index=False)
    if writer is None:
        schema = tbl.schema
        writer = pq.ParquetWriter(OUT, schema, compression="zstd")
    writer.write_table(tbl)
    by_product += new.sum(axis=0)
    stats.append({"month": label(t), "rows_in_month": len(cur), "eligible": len(m),
                  "not_eligible": len(cur) - len(m),
                  "customers_with_add": int((out["n_new"] > 0).sum()),
                  "total_adds": int(new.sum())})
    prev = cur
writer.close()

s = pd.DataFrame(stats).set_index("month")
s["share_with_add_%"] = (s.customers_with_add / s.eligible * 100).round(2)
pd.set_option("display.width", 200)
print("TARGET TABLE BUILT:", OUT, f"({os.path.getsize(OUT) / 1e6:,.0f} MB, {time.time() - T0:.0f}s)")
print(s.to_string())
print(f"\nTotal eligible rows: {s.eligible.sum():,} | total adds: {s.total_adds.sum():,}")
os.makedirs("analysis_output", exist_ok=True)
s.to_csv("analysis_output/target_table_monthly.csv")

print("\nAdds per product:")
bp = pd.Series(by_product, index=PRODUCT_COLUMNS).sort_values(ascending=False)
print(pd.DataFrame({"adds": bp, "%": (bp / bp.sum() * 100).round(2)}).to_string())

# ---------------------------------------------------------------- checks
print("\nCHECKS")
assert s.eligible.sum() == 12_682_421, "eligible pairs differ from earlier analysis"
assert s.total_adds.sum() == 561_710, "total adds differ from earlier analysis"
print("OK: 12,682,421 eligible pairs and 561,710 adds match the earlier independent analysis")
ref = "analysis_output/adds_drops_monthly.csv"
if os.path.exists(ref):
    r = pd.read_csv(ref, index_col=0)["total_adds"]
    same = (r.reindex(s.index).to_numpy() == s.total_adds.to_numpy())
    print("per-month adds vs the earlier raw-CSV analysis:", "ALL MATCH" if same.all()
          else f"MISMATCH in {list(s.index[~same])}")
ex = pq.read_table(OUT, filters=[("customer_id", "=", 16713)]).to_pandas()
print("\nExample customer 16713, months where new_credit_card = 1:")
print(ex.loc[ex["new_credit_card"] == 1, ["snapshot_date", "new_credit_card"]].to_string(index=False))
print("Expected: 2015-04, 2015-08, 2016-04")
