"""阶段9: 历史新闻回填(多年期回测) -> 追加去重到原始语料库 -> 重建各币种新闻面板。

用途: 打通"几年期回测"所需的历史语料。原始库(data/news_raw/corpus.parquet)与聚合
面板分离, 抓取可增量、可续跑; 面板随时可用当前库重建。

数据源(config.news.history.providers):
- synthetic:      离线兜底, 在 [start,end] 生成多年合成语料(无需网络), 打通全链路。
- cryptocompare:  用 lTs 向历史方向分页(需 $env:CRYPTOCOMPARE_KEY, 免费额度)。
- gdelt:          把日期区间切窗抓取(免费, 无需 key, 请遵守限速)。

用法(PowerShell):
    python scripts/09_backfill_news.py                         # 用 config 的区间/来源
    python scripts/09_backfill_news.py --start 2021-01-01 --end 2024-12-31
    python scripts/09_backfill_news.py --providers gdelt cryptocompare
    python scripts/09_backfill_news.py --rebuild-panels        # 回填后重建面板(默认开启)

推荐稳健入口(满额切分/续跑战役对齐/429长冷却/结束后校验):
    PYTHONUNBUFFERED=1 PYTHONPATH=src python -u scripts/run_news_backfill_robust.py \\
        --start 2020-01-01T00:00:00Z --providers gdelt
    PYTHONPATH=src python scripts/validate_news_alignment.py

回填后需把 news.use_history 设为 true(或本脚本自动重建的面板)供后续专家消费。
校验通过后再开 news.as_feature。cryptocompare 需环境变量 CRYPTOCOMPARE_KEY。
"""
import argparse

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data import backfill_news, build_news_panel, save_news_panel


def main():
    ap = argparse.ArgumentParser(description="历史新闻回填 + 重建面板")
    ap.add_argument("--start", default=None, help="回填起点(ISO, 覆盖 config)")
    ap.add_argument("--end", default=None, help="回填终点(ISO, 覆盖 config)")
    ap.add_argument("--providers", nargs="*", default=None,
                    help="来源子集: synthetic cryptocompare gdelt")
    ap.add_argument("--no-rebuild-panels", action="store_true",
                    help="仅回填原始库, 不重建各币种面板")
    args = ap.parse_args()

    cfg = Config.load()
    print(f"[start] 历史回填 providers={args.providers or cfg['news'].get('history', {}).get('providers')}")
    stats = backfill_news(cfg, start=args.start, end=args.end, providers=args.providers)
    for k, v in stats.items():
        if not k.startswith("_"):
            print(f"  - {k}: 抓取 {v} 条")
    print(f"[ok] 原始库新增 {stats['_added']} 条, 总量 {stats['_total']} 条 -> "
          f"{cfg.root / cfg['news'].get('history', {}).get('raw_dir', 'data/news_raw')}/corpus.parquet")

    if args.no_rebuild_panels:
        print("[skip] 未重建面板; 记得手动重跑 08 或设置 news.use_history=true。")
        return

    # 用历史库重建面板(临时强制 use_history=true 走历史路径)
    cfg["news"]["use_history"] = True
    for symbol in cfg["data"]["symbols"]:
        df = build_news_panel(cfg, symbol)
        if len(df) == 0:
            print(f"[warn] {symbol}: 历史库中无相关新闻。")
            continue
        path = save_news_panel(cfg, symbol, df)
        span = f"{df.index.min().date()}~{df.index.max().date()}" if len(df) else "-"
        print(f"[ok] {symbol}: {len(df)} 桶, 区间 {span}, 平均情绪 {df['sentiment'].mean():+.3f} -> {path}")


if __name__ == "__main__":
    main()
