#!/usr/bin/env python3
"""Step 1a: clean train_ver2.csv and test_ver2.csv -> processed/*.parquet

Cleaning rules (all documented here so the portfolio can cite them):
  * columns renamed to English (column_map.py)
  * age: numeric; values outside 2..110 -> null (ages 2-17 are real junior accounts, kept)
  * seniority_months_raw: the original column (numeric; negative sentinels -> null).
    In 2015-01..2015-07 it is frozen at a ~2015-07 value, so it is NOT used for modelling
  * tenure_months: recomputed = calendar-month difference snapshot_date - first_contract_date
    (matches the raw value exactly 88-90% of the time in the live months, 2015-07 onward)
  * is_new_6m: tenure_months <= 5 (replaces new_customer, which has ~900 impossible 0->1 flips)
  * demographics_stale: 1 for month_idx 0..6 (2015-01..2015-07), when non-product columns
    other than first_contract_date do not update month to month
  * customer_type_month_start: "1.0" and "1" merged -> "1" (also 2, 3, 4, P)
  * dates parsed to timestamps; month_idx = months since 2015-01 (0..16, test = 17)
  * product flags kept as nullable Int8: a missing flag stays NULL (never turned into 0)
  * spouse_of_employee dropped (99.99% null)

Run:  python -m pipeline.make_clean_parquet
Out:  processed/train_clean.parquet, processed/test_clean.parquet
"""
import os
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from santander.column_map import COLUMN_MAP, PRODUCT_COLUMNS

OUT = "processed"
os.makedirs(OUT, exist_ok=True)
STR_IN = {"age": str, "antiguedad": str, "indrel_1mes": str, "conyuemp": str,
          "ult_fec_cli_1t": str}
STRING_COLS = ["employee_index", "residence_country", "sex", "residence_index",
               "foreigner_index", "join_channel", "deceased_index", "province_name",
               "segment", "relation_type_month_start"]
INT8_COLS = ["new_customer", "primary_customer_index", "address_type", "province_code",
             "activity_index"]


def num(s):
    return pd.to_numeric(s.str.strip(), errors="coerce")


def clean(raw):
    ch = raw.rename(columns=COLUMN_MAP)
    d = ch["snapshot_date"]
    o = {
        "customer_id": ch["customer_id"].astype("int32"),
        "snapshot_date": pd.to_datetime(d),
        "month_idx": ((d.str[:4].astype(int) - 2015) * 12 + d.str[5:7].astype(int) - 1)
        .astype("int8"),
    }
    for c in STRING_COLS:
        o[c] = ch[c].astype("string")
    age = num(ch["age"])
    o["age"] = age.where((age >= 2) & (age <= 110)).astype("Int16")
    sen = num(ch["seniority_months"])
    o["seniority_months_raw"] = sen.where(sen >= 0).astype("Int16")
    o["first_contract_date"] = pd.to_datetime(ch["first_contract_date"], errors="coerce")
    snap, fc = o["snapshot_date"], o["first_contract_date"]
    tenure = (snap.dt.year - fc.dt.year) * 12 + (snap.dt.month - fc.dt.month)
    o["tenure_months"] = tenure.astype("Int16")
    o["is_new_6m"] = (tenure <= 5).where(tenure.notna()).astype("Int8")
    o["demographics_stale"] = (o["month_idx"] <= 6).astype("int8")
    o["last_primary_date"] = pd.to_datetime(ch["last_primary_date"], errors="coerce")
    o["customer_type_month_start"] = (ch["customer_type_month_start"].astype("string")
                                      .str.strip().str.replace(r"\.0$", "", regex=True))
    for c in INT8_COLS:
        o[c] = ch[c].astype("Int8")
    o["household_income"] = pd.to_numeric(ch["household_income"], errors="coerce")
    for c in PRODUCT_COLUMNS:                 # test file has no product columns
        if c in ch.columns:
            o[c] = ch[c].astype("Int8")
    return pd.DataFrame(o)


def convert(src, dst):
    t0 = time.time()
    writer, schema, rows = None, None, 0
    for k, raw in enumerate(pd.read_csv(src, chunksize=500_000, dtype=STR_IN, low_memory=False)):
        tbl = pa.Table.from_pandas(clean(raw), schema=schema, preserve_index=False)
        if writer is None:
            schema = tbl.schema
            writer = pq.ParquetWriter(dst, schema, compression="zstd")
        writer.write_table(tbl)
        rows += len(tbl)
        if k % 5 == 0:
            print(f"  {os.path.basename(src)}: {rows:,} rows ({time.time() - t0:.0f}s)", flush=True)
    writer.close()
    mb = lambda p: os.path.getsize(p) / 1e6
    print(f"{src}: {mb(src):,.0f} MB CSV -> {dst}: {mb(dst):,.0f} MB parquet "
          f"({mb(src) / mb(dst):.1f}x smaller), {rows:,} rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    convert("train_ver2.csv", os.path.join(OUT, "train_clean.parquet"))
    convert("test_ver2.csv", os.path.join(OUT, "test_clean.parquet"))
    for name in ["train_clean", "test_clean"]:
        f = pq.ParquetFile(os.path.join(OUT, f"{name}.parquet"))
        print(f"\n{name}: {f.metadata.num_rows:,} rows, {f.metadata.num_columns} columns, "
              f"{f.metadata.num_row_groups} row groups")
        print(f.schema_arrow.to_string(show_field_metadata=False)[:1500])
