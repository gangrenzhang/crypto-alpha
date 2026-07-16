"""阶段1: 采集(或合成) BTC/ETH 数据并落盘 parquet。"""
import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data import load_symbol_data, save_parquet


def main():
    cfg = Config.load()
    for symbol in cfg["data"]["symbols"]:
        df = load_symbol_data(cfg, symbol)
        fname = symbol.replace("/", "_") + ".parquet"
        path = save_parquet(df, cfg.data_dir / "raw" / fname)
        print(f"[ok] {symbol}: {len(df)} 行 -> {path}")


if __name__ == "__main__":
    main()
