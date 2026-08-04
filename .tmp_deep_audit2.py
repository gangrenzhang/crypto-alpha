"""深挖: NFP/CPI 双行与缺失、ECB FF 历史时刻、FOMC 17 行/年。"""
import pandas as pd
from crypto_alpha.data.macro_calendar_merge import canonical_macro_name

df = pd.read_parquet("data/macro_calendar/events.parquet", engine="pyarrow")
df["_canon"] = df["name"].map(canonical_macro_name)
cols = ["name", "source", "print_kind", "schedule_source", "released_at", "previous", "forecast", "actual"]

print("=== A. NFP 2025-05~2025-08 全部行 (查 2025-07 缺失/2025-06 双行) ===")
m = df[(df["_canon"] == "Nonfarm Payrolls") & (df["released_at"] >= "2025-05-01") & (df["released_at"] < "2025-09-01")]
print(m[cols].to_string(index=False))

print("\n=== B. NFP 2026-02~2026-06 (查 2026-04 缺失/2026-03 双行) ===")
m = df[(df["_canon"] == "Nonfarm Payrolls") & (df["released_at"] >= "2026-02-01") & (df["released_at"] < "2026-07-01")]
print(m[cols].to_string(index=False))

print("\n=== C. NFP 2020-01 双行样例 ===")
m = df[(df["_canon"] == "Nonfarm Payrolls") & (df["released_at"] >= "2020-01-01") & (df["released_at"] < "2020-02-01")]
print(m[cols].to_string(index=False))

print("\n=== D. CPI 2025-10~2025-12 (查 2025-11 缺失) ===")
m = df[(df["_canon"] == "CPI YoY") & (df["released_at"] >= "2025-10-01") & (df["released_at"] < "2026-01-01")]
print(m[cols].to_string(index=False))

print("\n=== E. CPI 2021-03 双行样例 ===")
m = df[(df["_canon"] == "CPI YoY") & (df["released_at"] >= "2021-03-01") & (df["released_at"] < "2021-04-01")]
print(m[cols].to_string(index=False))

print("\n=== F. ECB FF 行 2022-07 之后 (改点后应 13:15/12:15 UTC) ===")
ecb_ff = df[(df["country"] == "EU") & (df["_canon"] == "ECB Rate Decision") & (df["source"] == "forexfactory_hist")]
late = ecb_ff[ecb_ff["released_at"] >= "2022-07-21"]
for _, r in late.iterrows():
    hm = r["released_at"].strftime("%H:%M")
    flag = "OK" if hm in ("13:15", "12:15") else "❌"
    print(f"  {flag} {r['released_at']} {r['name']}")

print("\n=== G. FOMC 2022 全部 17 行 ===")
fomc = df[(df["_canon"] == "FOMC Rate Decision") & (df["released_at"].dt.year == 2022)]
print(fomc[["name", "source", "print_kind", "released_at", "actual"]].sort_values("released_at").to_string(index=False))

print("\n=== H. ADP 与 NFP 同刻的 5 例 ===")
adp = df[df["_canon"] == "ADP Nonfarm Employment Change"]
nfp = df[df["_canon"] == "Nonfarm Payrolls"]
for ts in adp["released_at"]:
    hit = nfp[(nfp["released_at"] - ts).abs() < pd.Timedelta("1min")]
    if len(hit):
        r = adp[adp["released_at"] == ts].iloc[0]
        print(f"  {ts} ADP:'{r['name']}'({r['source']}) vs NFP:'{hit.iloc[0]['name']}'({hit.iloc[0]['source']})")
