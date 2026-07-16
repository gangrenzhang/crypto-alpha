"""阶段8: 采集/合成多源新闻 -> 权威加权去重摘要 -> 落盘 parquet(供 LLM 专家)。

- use_synthetic=true(默认): 生成离线合成新闻, 无需网络/API key 即可跑通。
- use_synthetic=false: 从 config.news.sources 抓取真实来源(RSS/CryptoPanic/
  CryptoCompare/GDELT), 缺 key 或失败的源自动跳过。API key 走环境变量:
    $env:CRYPTOPANIC_KEY="..."; $env:CRYPTOCOMPARE_KEY="..."

产物: data/news/<SYMBOL>.parquet, 列: text/sentiment/n_items/max_authority/corroboration。
"""
import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data import build_news_panel, save_news_panel


def main():
    cfg = Config.load()
    mode = "合成" if cfg["news"].get("use_synthetic", True) else "真实抓取"
    for symbol in cfg["data"]["symbols"]:
        df = build_news_panel(cfg, symbol)
        if len(df) == 0:
            print(f"[warn] {symbol}: 未获得新闻(检查网络/key 或改用合成)。")
            continue
        path = save_news_panel(cfg, symbol, df)
        sent = df["sentiment"].mean()
        print(f"[ok] {symbol} ({mode}): {len(df)} 个时间桶, 平均情绪={sent:+.3f} -> {path}")
        print(f"     示例摘要: {df['text'].iloc[-1][:100]}")


if __name__ == "__main__":
    main()
