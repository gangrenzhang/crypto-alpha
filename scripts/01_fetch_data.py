"""阶段1: 采集(或合成) BTC/ETH 主周期 + 辅周期数据并落盘 parquet。"""
import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data import load_symbol_data, load_aux_timeframes, save_parquet


def main():
    cfg = Config.load()
    for symbol in cfg["data"]["symbols"]:
        df = load_symbol_data(cfg, symbol)
        fname = symbol.replace("/", "_") + ".parquet"
        path = save_parquet(df, cfg.data_dir / "raw" / fname)
        print(f"[ok] {symbol} {cfg['data']['timeframe']}: {len(df)} 行 -> {path}")

        aux = load_aux_timeframes(cfg, symbol, main_df=df)
        for tf, adf in aux.items():
            afname = symbol.replace("/", "_") + f"__{tf}.parquet"
            # 合成模式通常不落辅周期缓存; 真实模式 load 时已写入。此处统一再落一份便于检查。
            apath = save_parquet(adf, cfg.data_dir / "raw" / afname)
            print(f"[ok] {symbol} {tf}: {len(adf)} 行 -> {apath}")


if __name__ == "__main__":
    main()
