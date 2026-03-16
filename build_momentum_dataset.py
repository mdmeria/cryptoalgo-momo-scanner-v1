#!/usr/bin/env python3
"""
Build a 1-minute OHLCV dataset for the backtest engine.

Fetches data from Binance Spot or Futures (perpetuals) API.

Usage:
  # Futures (perpetuals) data with extra symbols:
  python build_momentum_dataset.py \
      --futures \
      --universe top100_midcap \
      --extra-symbols BTCUSDT,ETHUSDT,SOLUSDT \
      --start 2026-03-02 \
      --out-dir datasets/perp_mar2_mar15

  # Spot data from existing manifest:
  python build_momentum_dataset.py \
      --source-manifest datasets/momo_1m_7d_top100_midcap_30d/dataset_manifest.csv \
      --start 2026-03-02 \
      --out-dir datasets/momo_1m_12d_mar2
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BINANCE_SPOT_API = "https://data-api.binance.vision/api/v3"
BINANCE_FUTURES_API = "https://fapi.binance.com/fapi/v1"
BITUNIX_FUTURES_API = "https://fapi.bitunix.com/api/v1/futures"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_1m_range(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                   api_base: str = BINANCE_SPOT_API) -> Optional[pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    end_ms = int(end_ts.timestamp() * 1000)
    start_ms = int(start_ts.timestamp() * 1000)

    cur_end_ms = end_ms
    while True:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "limit": 1000,
            "endTime": cur_end_ms,
        }
        r = requests.get(f"{api_base}/klines", params=params, timeout=20)
        if r.status_code != 200:
            return None
        raw = r.json()
        if not isinstance(raw, list) or len(raw) == 0:
            break

        for k in raw:
            ts_ms = int(k[0])
            if ts_ms < start_ms:
                continue
            rows.append({
                "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                "open":   float(k[1]),
                "high":   float(k[2]),
                "low":    float(k[3]),
                "close":  float(k[4]),
                "volume": float(k[5]),
            })

        earliest_ms = int(raw[0][0])
        if earliest_ms <= start_ms:
            break
        cur_end_ms = earliest_ms - 1
        time.sleep(0.05)

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp")
    df = df[df["timestamp"] >= start_ts]
    return df


def fetch_1m_range_bitunix(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Fetch 1m klines from Bitunix Futures API (100 bars per request, oldest first)."""
    rows: list[dict[str, Any]] = []
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    cur_start_ms = start_ms
    while cur_start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": str(cur_start_ms),
            "endTime": str(end_ms),
        }
        r = requests.get(f"{BITUNIX_FUTURES_API}/market/kline", params=params, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("code") != 0 or not data.get("data"):
            break

        raw = data["data"]
        if not isinstance(raw, list) or len(raw) == 0:
            break

        batch_timestamps = []
        for k in raw:
            ts_ms = int(k["time"])
            if ts_ms < start_ms or ts_ms > end_ms:
                continue
            batch_timestamps.append(ts_ms)
            rows.append({
                "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                "open":   float(k["open"]),
                "high":   float(k["high"]),
                "low":    float(k["low"]),
                "close":  float(k["close"]),
                "volume": float(k["quoteVol"]),  # quoteVol = quantity in base asset
            })

        if not batch_timestamps:
            break
        # Bitunix returns newest first; advance past the oldest in this batch
        oldest_ms = min(batch_timestamps)
        newest_ms = max(batch_timestamps)
        if newest_ms >= end_ms or len(raw) < 100:
            break
        cur_start_ms = newest_ms + 60000  # next minute after newest
        time.sleep(0.05)

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp")
    df = df[df["timestamp"] >= start_ts]
    return df


def fetch_24h_volumes(api_base: str = BINANCE_SPOT_API) -> dict[str, float]:
    """Return {symbol: quoteVolume} for all USDT pairs."""
    url = f"{api_base}/ticker/24hr"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return {
        item["symbol"]: float(item.get("quoteVolume", 0))
        for item in r.json()
        if item["symbol"].endswith("USDT")
    }


def top_midcap_symbols(skip_top: int = 20, count: int = 100,
                       api_base: str = BINANCE_SPOT_API) -> list[str]:
    """Return midcap USDT symbols: skip the top N by volume, take next `count`."""
    vols = fetch_24h_volumes(api_base)
    sorted_syms = sorted(vols, key=lambda s: vols[s], reverse=True)
    mid = sorted_syms[skip_top: skip_top + count]
    return mid


def fetch_futures_symbols() -> list[str]:
    """Return all actively trading USDT perpetual symbols on Binance Futures."""
    r = requests.get(f"{BINANCE_FUTURES_API}/exchangeInfo", timeout=20)
    r.raise_for_status()
    symbols = []
    for s in r.json().get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
        ):
            symbols.append(s["symbol"])
    return sorted(set(symbols))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build 1m OHLCV dataset for backtest")
    parser.add_argument("--start", required=True,
                        help="Start date (UTC), e.g. 2026-03-02")
    parser.add_argument("--end", default=None,
                        help="End date (UTC), e.g. 2026-03-14. Defaults to now.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for CSV files and manifest")
    parser.add_argument("--source-manifest", default=None,
                        help="Use symbols from an existing manifest CSV")
    parser.add_argument("--universe", default=None, choices=["top100_midcap"],
                        help="Auto-build symbol universe (alternative to --source-manifest)")
    parser.add_argument("--midcap-skip-top", type=int, default=20,
                        help="Skip top N symbols by volume when building midcap universe")
    parser.add_argument("--midcap-count", type=int, default=100,
                        help="Number of midcap symbols to fetch")
    parser.add_argument("--extra-symbols", default=None,
                        help="Comma-separated extra symbols to add to the universe")
    parser.add_argument("--extra-symbols-file", default=None,
                        help="File with extra symbols (one per line or comma-separated)")
    parser.add_argument("--futures", action="store_true",
                        help="Use Binance Futures (perpetuals) API instead of Spot")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel fetch workers")
    args = parser.parse_args()

    api_base = BINANCE_FUTURES_API if args.futures else BINANCE_SPOT_API
    source_label = "binance_futures_perp" if args.futures else "binance_data_api_spot"

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts   = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.utcnow()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"API        : {'Futures (perps)' if args.futures else 'Spot'}")
    print(f"Date range : {start_ts.date()} -> {end_ts.date()}")

    # --- Build symbol list ---
    if args.source_manifest:
        mf = pd.read_csv(args.source_manifest)
        symbols = mf[mf["ok"].astype(str).str.lower() == "true"]["symbol"].tolist()
        print(f"Symbols    : {len(symbols)} from {args.source_manifest}")
    elif args.universe == "top100_midcap":
        print("Fetching top midcap universe...")
        symbols = top_midcap_symbols(skip_top=args.midcap_skip_top,
                                     count=args.midcap_count, api_base=api_base)
        print(f"Symbols    : {len(symbols)}")
    else:
        symbols = []

    # --- Add extra symbols ---
    extra = set()
    if args.extra_symbols:
        for s in args.extra_symbols.split(","):
            s = s.strip()
            if s:
                extra.add(s)
    if args.extra_symbols_file:
        with open(args.extra_symbols_file) as f:
            for line in f:
                for s in line.strip().split(","):
                    s = s.strip()
                    if s:
                        extra.add(s)

    if extra:
        existing = set(symbols)
        added = extra - existing
        symbols = symbols + sorted(added)
        print(f"Extra syms : {len(added)} added ({len(extra)} requested, {len(extra - added)} already present)")

    if not symbols:
        parser.error("No symbols to fetch. Provide --source-manifest, --universe, or --extra-symbols")

    # If futures mode, validate symbols exist on futures exchange
    if args.futures:
        print("Validating symbols against Futures exchange...")
        valid_futures = set(fetch_futures_symbols())
        invalid = [s for s in symbols if s not in valid_futures]
        if invalid:
            print(f"  Removing {len(invalid)} symbols not on Futures: {', '.join(invalid[:20])}{'...' if len(invalid)>20 else ''}")
        symbols = [s for s in symbols if s in valid_futures]
        print(f"Valid syms : {len(symbols)}")

    # --- Fetch data in parallel ---
    manifest_rows = []

    def fetch_one(sym):
        out_path = out_dir / f"{sym}_1m.csv"
        try:
            df = fetch_1m_range(sym, start_ts, end_ts, api_base=api_base)
            if df is None or df.empty:
                return sym, False, "no_data", 0, str(out_path)
            df.to_csv(str(out_path), index=False)
            return sym, True, "ok", len(df), str(out_path)
        except Exception as e:
            return sym, False, str(e)[:80], 0, str(out_path)

    print(f"Fetching {len(symbols)} symbols with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            sym, ok, reason, bars, path = fut.result()
            status = "OK" if ok else "SKIP"
            print(f"  [{i:3d}/{len(symbols)}] {sym:20s} {status}  {bars} bars")
            manifest_rows.append({
                "symbol": sym, "ok": ok, "reason": reason,
                "bars": bars, "path": path, "source": source_label
            })

    # --- Write manifest ---
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "dataset_manifest.csv"
    manifest_df.to_csv(str(manifest_path), index=False)

    ok_count = manifest_df["ok"].sum()
    print(f"\nDone. {ok_count}/{len(symbols)} symbols fetched.")
    print(f"Manifest   : {manifest_path}")
    print(f"Output dir : {out_dir}")


if __name__ == "__main__":
    main()
