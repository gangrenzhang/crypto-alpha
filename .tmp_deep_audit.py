"""修复后深度审计: 10 维度核查 events.parquet。"""
import pandas as pd
from crypto_alpha.data.macro_calendar_merge import canonical_macro_name

df = pd.read_parquet("data/macro_calendar/events.parquet", engine="pyarrow")
df["_canon"] = df["name"].map(canonical_macro_name)
ok = []

# 1) NFP 逐月覆盖与重复 (2020-01 ~ 2026-06 已公布月份)
nfp = df[(df["_canon"] == "Nonfarm Payrolls") & (df["country"] == "US")]
nfp_m = nfp.groupby(nfp["released_at"].dt.to_period("M")).size()
dup_m = nfp_m[nfp_m > 1]
missing = [p for p in pd.period_range("2020-01", "2026-06", freq="M") if p not in nfp_m.index]
print(f"1) NFP: {len(nfp)} 行, 覆盖 {len(nfp_m)} 月, 同月多行: {dict(dup_m)}, 缺失月份: {missing}")
ok.append(len(dup_m) == 0 and len(missing) == 0)

# 2) CPI YoY 逐月覆盖
cpi = df[(df["_canon"] == "CPI YoY") & (df["country"] == "US")]
cpi_m = cpi.groupby(cpi["released_at"].dt.to_period("M")).size()
dup_c = cpi_m[cpi_m > 1]
missing_c = [p for p in pd.period_range("2020-01", "2026-06", freq="M") if p not in cpi_m.index]
print(f"2) CPI: {len(cpi)} 行, 覆盖 {len(cpi_m)} 月, 同月多行: {dict(dup_c)}, 缺失月份: {missing_c}")
ok.append(len(dup_c) == 0 and len(missing_c) == 0)

# 3) ADP 与 NFP 未误合并
adp = df[df["_canon"] == "ADP Nonfarm Employment Change"]
same_hour = 0
for ts in adp["released_at"]:
    if ((nfp["released_at"] - ts).abs() < pd.Timedelta("1min")).any():
        same_hour += 1
print(f"3) ADP 独立行: {len(adp)}, 与 NFP 同刻行: {same_hour} (应 0, ADP 周三 vs NFP 周五)")
ok.append(same_hour == 0)

# 4) ECB 全部行时刻 (14:15 CET = 冬 13:15 / 夏 12:15 UTC)
ecb = df[(df["country"] == "EU") & (df["_canon"] == "ECB Rate Decision")].sort_values("released_at")
bad_ecb = []
for _, r in ecb.iterrows():
    hm = r["released_at"].strftime("%H:%M")
    if hm not in ("13:15", "12:15"):
        bad_ecb.append((str(r["released_at"]), r["source"], r["name"]))
print(f"4) ECB 行: {len(ecb)} (手工 {len(ecb[ecb.source=='centralbank_historical'])}), 时刻异常: {bad_ecb[:5]}")

# 5) BLS 补充表 10 对生效核查
supp = [("Nonfarm Payrolls", "2024-01-05 13:30"), ("CPI YoY", "2024-01-11 13:30"),
        ("Nonfarm Payrolls", "2024-04-05 12:30"), ("CPI YoY", "2024-04-10 12:30"),
        ("CPI YoY", "2025-07-15 12:30"), ("Nonfarm Payrolls", "2026-02-06 13:30"),
        ("CPI YoY", "2026-02-11 13:30"), ("CPI YoY", "2026-04-10 12:30"),
        ("Nonfarm Payrolls", "2026-05-01 12:30"), ("CPI YoY", "2026-05-12 12:30")]
bad_supp = []
for canon, expect in supp:
    hit = df[(df["_canon"] == canon) & (df["released_at"] == pd.Timestamp(expect + ":00Z"))]
    if len(hit) != 1:
        bad_supp.append((canon, expect, len(hit)))
print(f"5) BLS 补充表生效: {10 - len(bad_supp)}/10, 异常: {bad_supp}")
ok.append(len(bad_supp) == 0)

# 6) FOMC 决议年度覆盖 + 同刻单行
fomc = df[df["_canon"] == "FOMC Rate Decision"]
per_y = fomc.groupby(fomc["released_at"].dt.year).size()
dup_f = fomc.groupby(fomc["released_at"].dt.floor("h")).size()
dup_f = dup_f[dup_f > 1]
print(f"6) FOMC: {len(fomc)} 行, 年度分布 {per_y.to_dict()}, 同刻多行: {len(dup_f)}")
ok.append(len(dup_f) == 0)

# 7) 新闻发布会未误并入决议
press = df[df["name"].astype(str).str.contains("Press Conference", case=False)]
press_canon = press["_canon"].value_counts().to_dict()
print(f"7) Press Conference 行: {len(press)}, canonical 分布: {press_canon}")
ok.append("FOMC Rate Decision" not in press_canon and "ECB Rate Decision" not in press_canon)

# 8) ff_week 行明细
ffw = df[df["source"] == "forexfactory_week"]
n_ffw_num = int(ffw[["previous", "forecast", "actual"]].notna().any(axis=1).sum())
print(f"8) ff_week: {len(ffw)} 行 ({n_ffw_num} 带数值), {ffw['released_at'].min()} ~ {ffw['released_at'].max()}")

# 9) PIT 与残留
neg = df[df["released_at"] < df["scheduled_at"]]
heur = df[df["schedule_source"] == "heuristic"]
test_src = df[df["source"] == "test"]
print(f"9) released<scheduled: {len(neg)}, heuristic: {len(heur)}, source=test: {len(test_src)}")
ok.append(len(neg) == 0 and len(heur) == 0 and len(test_src) == 0)

# 10) 全局 canonical 同刻残留 (数值类事件)
num = df[df[["previous", "forecast", "actual"]].notna().any(axis=1)]
g = num.groupby([num["country"], num["_canon"], num["released_at"].dt.floor("h")]).size()
resid = g[g > 1]
print(f"10) 数值事件 canonical+hour 残留多行组: {len(resid)}")
if len(resid):
    print(resid.head(5))
ok.append(len(resid) == 0)

print(f"\n== 硬断言通过: {sum(ok)}/{len(ok)} ==")
