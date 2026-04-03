#!/usr/bin/env python3
"""
Download daily 1m futures kline data from Binance Public Data and append
to existing monthly data in binance_futures_1m/.

Usage:
    python download_binance_daily.py
"""

import sys
import time
import io
import zipfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NUM_WORKERS = 6
DATA_DIR = Path("datasets/binance_futures_1m")
BINANCE_DAILY_BASE = "https://data.binance.vision/data/futures/um/daily/klines"

# Download March 1 to yesterday
START_DATE = datetime(2026, 3, 1, tzinfo=timezone.utc)
END_DATE = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)

print_lock = threading.Lock()
counters = {"extended": 0, "no_data": 0, "failed": 0, "up_to_date": 0}
counter_lock = threading.Lock()


def get_symbols() -> list[str]:
    """Get symbols from whitelist."""
    whitelist_file = Path("approved_symbols.txt")
    if whitelist_file.exists():
        with open(whitelist_file) as f:
            return sorted({line.strip() for line in f if line.strip()})
    return sorted(f.stem.replace("_1m", "") for f in DATA_DIR.glob("*.csv"))


def convert_symbol_for_binance(symbol: str) -> str:
    mapping = {
        "1000BONKUSDT": "1000BONKUSDT",
        "1000FLOKIUSDT": "1000FLOKIUSDT",
        "1000LUNCUSDT": "1000LUNCUSDT",
        "1000PEPEUSDT": "1000PEPEUSDT",
        "1000RATSUSDT": "1000RATSUSDT",
        "1000SATSUSDT": "1000SATSUSDT",
        "1MBABYDOGEUSDT": "1MBABYDOGEUSDT",
        "LUNA2USDT": "LUNA2USDT",
    }
    return mapping.get(symbol, symbol)


def parse_binance_csv(f) -> pd.DataFrame:
    """Parse a Binance kline CSV, handling header/no-header."""
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    first_bytes = f.read(20)
    f.seek(0)
    has_header = b"open_time" in first_bytes or not first_bytes[:1].isdigit()
    if has_header:
        df = pd.read_csv(f, names=cols, header=0)
    else:
        df = pd.read_csv(f, names=cols, header=None)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])
    df["timestamp"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "quote_volume"]]
    df = df.rename(columns={"quote_volume": "volume"})
    return df


def process_symbol(symbol: str, total: int) -> None:
    """Download daily files for one symbol and append to existing CSV."""
    binance_sym = convert_symbol_for_binance(symbol)
    csv_path = DATA_DIR / f"{symbol}_1m.csv"

    # Read existing data to find where it ends
    if csv_path.exists():
        df_existing = pd.read_csv(str(csv_path))
        df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)
        last_ts = df_existing["timestamp"].iloc[-1]
        # Start from the day after last data
        fetch_start = (last_ts + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        df_existing = pd.DataFrame()
        fetch_start = START_DATE

    if fetch_start > END_DATE:
        with counter_lock:
            counters["up_to_date"] += 1
        return

    # Download daily files
    all_dfs = []
    current = fetch_start
    while current <= END_DATE:
        date_str = current.strftime("%Y-%m-%d")
        filename = f"{binance_sym}-1m-{date_str}.zip"
        url = f"{BINANCE_DAILY_BASE}/{binance_sym}/1m/{filename}"

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                current += timedelta(days=1)
                continue
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = parse_binance_csv(f)
                    all_dfs.append(df)
        except Exception:
            pass

        current += timedelta(days=1)

    if not all_dfs:
        with counter_lock:
            counters["no_data"] += 1
            done = sum(counters.values())
        with print_lock:
            print(f"  [{done}/{total}] {symbol}: no daily data")
        return

    df_new = pd.concat(all_dfs, ignore_index=True)

    if not df_existing.empty:
        combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        combined = df_new

    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    combined.to_csv(str(csv_path), index=False)

    with counter_lock:
        counters["extended"] += 1
        done = sum(counters.values())

    new_end = df_new["timestamp"].iloc[-1].strftime("%m/%d")
    with print_lock:
        print(f"  [{done}/{total}] {symbol}: +{len(df_new)} bars (now ends {new_end})")


def main():
    symbols = get_symbols()
    total = len(symbols)

    days = (END_DATE - START_DATE).days + 1
    print(f"Downloading daily data for {total} symbols")
    print(f"Date range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')} ({days} days)")
    print(f"Output: {DATA_DIR}/")
    print(f"Workers: {NUM_WORKERS} parallel threads\n")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_symbol, sym, total): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                future.result()
            except Exception as e:
                with print_lock:
                    print(f"  ERROR {sym}: {e}")
                with counter_lock:
                    counters["failed"] += 1

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Extended: {counters['extended']}")
    print(f"  Already up-to-date: {counters['up_to_date']}")
    print(f"  No daily data: {counters['no_data']}")
    print(f"  Failed: {counters['failed']}")


if __name__ == "__main__":
    main()
