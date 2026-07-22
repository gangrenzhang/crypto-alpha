"""阶段2: 构建特征矩阵(技术指标 + 分数阶差分 + 多周期上下文), 落盘并打印概览。"""
import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data import load_symbol_data, save_parquet
from crypto_alpha.features.build import build_feature_matrix, feature_columns, mtf_columns
from crypto_alpha.features.news_features import add_news_features
from crypto_alpha.features.macro_calendar import add_macro_calendar_features


def main():
    cfg = Config.load()
    for symbol in cfg["data"]["symbols"]:
        raw = load_symbol_data(cfg, symbol)
        feat = build_feature_matrix(raw, cfg, symbol=symbol)
        feat["close"] = raw["close"]
        feat = add_news_features(feat, cfg, symbol)
        feat = add_macro_calendar_features(feat, cfg, symbol)
        fcols = feature_columns(feat)
        mcols = mtf_columns(feat)
        fname = symbol.replace("/", "_") + ".parquet"
        path = save_parquet(feat, cfg.data_dir / "features" / fname)
        print(f"[ok] {symbol}: {feat.shape[0]} 行 × {len(fcols)} 特征 -> {path}")
        print(f"     多周期特征 {len(mcols)} 列: {mcols[:6]}{' ...' if len(mcols) > 6 else ''}")
        print(f"     特征列示例: {fcols[:8]} ...")


if __name__ == "__main__":
    main()
