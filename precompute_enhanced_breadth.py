#!/usr/bin/env python3
"""Precompute enhanced market breadth: breakout count from liquid coins."""
import sys, os, time, pandas as pd, numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_DIR = Path("datasets/binance_futures_1m")
EVAL_INTERVAL = 120  # every 120 bars
MIN_VOL_5M = 500_000
OUTPUT_CSV = "enhanced_breadth_cache.csv"


def main():
    t0 = time.time()

    # Load BTC for timestamp reference
    btc = pd.read_csv(str(DATASET_DIR / "BTCUSDT_1m.csv"), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    btc_ts = btc["timestamp"].values
    n_btc = len(btc_ts)
    print(f"BTC bars: {n_btc}")

    # Load all symbols: close, high, low, volume arrays + timestamps
    syms = sorted([f.replace("_1m.csv", "") for f in os.listdir(DATASET_DIR) if f.endswith("_1m.csv")])
    print(f"Loading {len(syms)} symbols...")

    # Load symbols in memory-efficient way: only keep arrays, not DataFrames
    # Process in batches to avoid OOM
    BATCH_SIZE = 50
    checkpoints = list(range(400, n_btc, EVAL_INTERVAL))
    n_cp = len(checkpoints)
    print(f"Computing {n_cp} checkpoints across {len(syms)} symbols in batches of {BATCH_SIZE}")

    # Initialize accumulator arrays
    liquid_count = np.zeros(n_cp, dtype=int)
    bullish_breakouts = np.zeros(n_cp, dtype=int)
    bearish_breakouts = np.zeros(n_cp, dtype=int)
    up_5pct = np.zeros(n_cp, dtype=int)
    down_5pct = np.zeros(n_cp, dtype=int)

    for batch_start in range(0, len(syms), BATCH_SIZE):
        batch_syms = syms[batch_start:batch_start + BATCH_SIZE]
        batch_data = {}
        for sym in batch_syms:
            fpath = DATASET_DIR / f"{sym}_1m.csv"
            try:
                df = pd.read_csv(str(fpath), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                if len(df) < 400:
                    continue
                batch_data[sym] = (
                    df["timestamp"].values,
                    df["close"].values.astype(np.float32),
                    df["high"].values.astype(np.float32),
                    df["low"].values.astype(np.float32),
                    df["volume"].values.astype(np.float32),
                )
            except Exception:
                continue

        # Process all checkpoints for this batch
        for ci, btc_i in enumerate(checkpoints):
            current_ts = btc_ts[btc_i]
            for sym, (ts, c, h, l, v) in batch_data.items():
                idx = int(np.searchsorted(ts, current_ts, side="right")) - 1
                if idx < 361 or idx >= len(c):
                    continue
                vol_5m = v[max(0, idx-4):idx+1].sum()
                if vol_5m < MIN_VOL_5M:
                    continue
                liquid_count[ci] += 1
                hi_6h = np.max(h[idx-360:idx])
                lo_6h = np.min(l[idx-360:idx])
                if c[idx] > hi_6h:
                    bullish_breakouts[ci] += 1
                if c[idx] < lo_6h:
                    bearish_breakouts[ci] += 1
                if idx >= 1440:
                    day_change = (c[idx] - c[idx-1440]) / c[idx-1440] * 100
                    if day_change >= 5:
                        up_5pct[ci] += 1
                    elif day_change <= -5:
                        down_5pct[ci] += 1

        del batch_data
        print(f"  Batch {batch_start//BATCH_SIZE + 1}/{(len(syms)-1)//BATCH_SIZE + 1} done ({time.time()-t0:.0f}s)", flush=True)

    results = []
    for ci, btc_i in enumerate(checkpoints):
        results.append({
            "timestamp": btc_ts[btc_i],
            "liquid_coins": int(liquid_count[ci]),
            "bullish_breakouts": int(bullish_breakouts[ci]),
            "bearish_breakouts": int(bearish_breakouts[ci]),
            "up_5pct": int(up_5pct[ci]),
            "down_5pct": int(down_5pct[ci]),
        })

    rdf = pd.DataFrame(results)
    rdf.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone in {time.time()-t0:.0f}s — {len(rdf)} checkpoints saved to {OUTPUT_CSV}")

    # Quick stats
    print(f"\nStats:")
    print(f"  Avg liquid coins: {rdf['liquid_coins'].mean():.1f}")
    print(f"  Avg bullish breakouts: {rdf['bullish_breakouts'].mean():.1f}")
    print(f"  Avg bearish breakouts: {rdf['bearish_breakouts'].mean():.1f}")
    print(f"  Max bullish: {rdf['bullish_breakouts'].max()}")
    print(f"  Max bearish: {rdf['bearish_breakouts'].max()}")


if __name__ == "__main__":
    main()
