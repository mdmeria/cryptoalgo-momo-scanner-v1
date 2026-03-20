#!/usr/bin/env python3
"""
Download 2 months of 1m candle data for all symbols in spot_mar2_mar15.

Fetches from Bitunix API (100 bars per request, paginated).
Output: datasets/spot_jan17_mar17/  (280 symbols, ~86k bars each)
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BITUNIX_BASE = "https://fapi.bitunix.com"
OUTPUT_DIR = Path("datasets/spot_jan17_mar17")
SOURCE_DIR = Path("datasets/spot_mar2_mar15")

# Date range: Jan 17 00:00 UTC to Mar 17 00:00 UTC (2 months)
START_TS = int(datetime(2026, 1, 17, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
END_TS = int(datetime(2026, 3, 17, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines_range(symbol: str, start_ms: int, end_ms: int,
                       interval: str = "1m") -> list[dict]:
    """Fetch all 1m candles between start_ms and end_ms, paginating forward."""
    all_rows = []
    cursor_start = start_ms
    request_count = 0
    max_requests = 1500  # Safety: 86400 bars / 100 per request = 864

    while cursor_start < end_ms and request_count < max_requests:
        cursor_end = min(cursor_start + 100 * 60 * 1000, end_ms)

        try:
            resp = requests.get(
                f"{BITUNIX_BASE}/api/v1/futures/market/kline",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": str(cursor_start),
                    "endTime": str(cursor_end),
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                break
            klines = data.get("data", [])
        except Exception as e:
            # Retry once after short wait
            time.sleep(1)
            try:
                resp = requests.get(
                    f"{BITUNIX_BASE}/api/v1/futures/market/kline",
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": str(cursor_start),
                        "endTime": str(cursor_end),
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                klines = data.get("data", []) if data.get("code") == 0 else []
            except Exception:
                break

        if not klines:
            # No data in this window, skip forward
            cursor_start = cursor_end + 1
            request_count += 1
            time.sleep(0.12)
            continue

        max_ts = 0
        for k in klines:
            ts = k.get("time")
            if ts is None:
                continue
            ts_int = int(ts)
            max_ts = max(max_ts, ts_int)
            all_rows.append({
                "timestamp": pd.Timestamp(ts_int, unit="ms", tz="UTC"),
                "open": float(k.get("open", 0)),
                "high": float(k.get("high", 0)),
                "low": float(k.get("low", 0)),
                "close": float(k.get("close", 0)),
                "volume": float(k.get("quoteVol", 0)),
            })

        # Move cursor past the last received bar
        cursor_start = max_ts + 60_000  # next minute after last bar
        request_count += 1
        time.sleep(0.12)  # Rate limit

    return all_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get symbol list from existing dataset
    csv_files = sorted(SOURCE_DIR.glob("*_1m.csv"))
    symbols = [f.stem.replace("_1m", "") for f in csv_files]
    print(f"Downloading 2 months of 1m data for {len(symbols)} symbols")
    print(f"Date range: 2026-01-17 to 2026-03-17")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Estimated time: ~8 hours\n")

    completed = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for i, sym in enumerate(symbols):
        out_path = OUTPUT_DIR / f"{sym}_1m.csv"

        # Skip if already downloaded
        if out_path.exists():
            existing = pd.read_csv(str(out_path))
            if len(existing) > 80000:
                skipped += 1
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"  [{i+1}/{len(symbols)}] {sym} skipped (already complete)")
                continue

        rows = fetch_klines_range(sym, START_TS, END_TS)

        if not rows:
            failed += 1
            print(f"  [{i+1}/{len(symbols)}] {sym} FAILED (no data)")
            continue

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df.to_csv(str(out_path), index=False)

        completed += 1
        elapsed = time.time() - start_time
        rate = elapsed / (completed + skipped) if (completed + skipped) > 0 else 0
        remaining = rate * (len(symbols) - i - 1)
        remaining_hrs = remaining / 3600

        if (i + 1) % 10 == 0 or i < 3:
            print(f"  [{i+1}/{len(symbols)}] {sym}: {len(df)} bars "
                  f"({df.iloc[0]['timestamp'].strftime('%m/%d')} to "
                  f"{df.iloc[-1]['timestamp'].strftime('%m/%d')}) "
                  f"| ETA: {remaining_hrs:.1f}h")

    elapsed_total = (time.time() - start_time) / 3600
    print(f"\nDone in {elapsed_total:.1f} hours")
    print(f"  Completed: {completed}")
    print(f"  Skipped (already had): {skipped}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
