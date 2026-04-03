#!/usr/bin/env python3
"""
Precompute 4-component market condition score every 120 bars.

Components:
  1. BTC vs previous day high/low: +1 above, -1 below, 0 between
  2. BTC SMMA30 slope (120-bar): +1 rising, -1 falling, 0 flat
  3. BTC vs session VWAP: +1 above, -1 below
  4. Enhanced breadth (breakout counts): +1 if 2+ bullish BOs, -1 if 2+ bearish BOs

Range: -4 to +4
"""
import sys, os, time
import pandas as pd, numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_DIR = Path("datasets/binance_futures_1m")
EVAL_INTERVAL = 120
MIN_VOL_5M = 500_000
OUTPUT_CSV = "enhanced_market_cache.csv"


def main():
    t0 = time.time()

    # ── Load BTC ──
    btc = pd.read_csv(str(DATASET_DIR / "BTCUSDT_1m.csv"),
                       parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    btc_c = btc["close"].values.astype(float)
    btc_h = btc["high"].values.astype(float)
    btc_l = btc["low"].values.astype(float)
    btc_v = btc["volume"].values.astype(float)
    btc_ts = btc["timestamp"].values
    n = len(btc_c)
    print(f"BTC bars: {n}")

    # ── Precompute BTC indicators ──

    # SMMA30
    btc_smma30 = pd.Series(btc_c).ewm(alpha=1.0 / 30, adjust=False).mean().values

    # Previous day high/low (calendar day UTC)
    dates = pd.to_datetime(btc_ts).normalize()
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    current_date = None
    day_start = 0
    prev_hi = np.nan
    prev_lo = np.nan
    for j in range(n):
        d = dates[j]
        if d != current_date:
            # New day — previous day is the range we just finished
            if current_date is not None:
                prev_hi = np.max(btc_h[day_start:j])
                prev_lo = np.min(btc_l[day_start:j])
            day_start = j
            current_date = d
        prev_day_high[j] = prev_hi
        prev_day_low[j] = prev_lo

    # Session VWAP
    btc_vol_base = btc_v / np.where(btc_c > 0, btc_c, 1.0)  # approx base volume
    tp = (btc_h + btc_l + btc_c) / 3.0
    vwap_arr = np.full(n, np.nan)
    cum_tp_vol = 0.0
    cum_vol = 0.0
    current_date = None
    for j in range(n):
        d = dates[j]
        if d != current_date:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            current_date = d
        cum_tp_vol += tp[j] * btc_vol_base[j]
        cum_vol += btc_vol_base[j]
        vwap_arr[j] = cum_tp_vol / cum_vol if cum_vol > 0 else btc_c[j]

    # ── Load enhanced breadth cache ──
    breadth_path = Path("enhanced_breadth_cache.csv")
    if breadth_path.exists():
        breadth_df = pd.read_csv(str(breadth_path), parse_dates=["timestamp"])
        breadth_ts = breadth_df["timestamp"].values
        breadth_bull = breadth_df["bullish_breakouts"].values
        breadth_bear = breadth_df["bearish_breakouts"].values
        print(f"Enhanced breadth: {len(breadth_df)} checkpoints loaded")
    else:
        print("WARNING: enhanced_breadth_cache.csv not found — breadth will be 0")
        breadth_ts = None

    # ── Compute score at each checkpoint ──
    checkpoints = list(range(400, n, EVAL_INTERVAL))
    print(f"Computing {len(checkpoints)} checkpoints...")

    results = []
    for btc_i in checkpoints:
        # 1. BTC vs previous day high/low
        if not np.isnan(prev_day_high[btc_i]) and not np.isnan(prev_day_low[btc_i]):
            if btc_c[btc_i] > prev_day_high[btc_i]:
                btc_prevday = 1
            elif btc_c[btc_i] < prev_day_low[btc_i]:
                btc_prevday = -1
            else:
                btc_prevday = 0
        else:
            btc_prevday = 0

        # 2. BTC SMMA30 slope (120-bar)
        if btc_i >= 121:
            slope_pct = (btc_smma30[btc_i] - btc_smma30[btc_i - 120]) / btc_smma30[btc_i - 120] * 100
            btc_smma_sig = 1 if slope_pct > 0.01 else (-1 if slope_pct < -0.01 else 0)
        else:
            btc_smma_sig = 0

        # 3. BTC vs session VWAP
        if not np.isnan(vwap_arr[btc_i]):
            btc_vwap_sig = 1 if btc_c[btc_i] > vwap_arr[btc_i] else -1
        else:
            btc_vwap_sig = 0

        # 4. Enhanced breadth (breakout counts)
        breadth_sig = 0
        if breadth_ts is not None:
            b_idx = int(np.searchsorted(breadth_ts, btc_ts[btc_i], side="right")) - 1
            if 0 <= b_idx < len(breadth_bull):
                bull_bos = int(breadth_bull[b_idx])
                bear_bos = int(breadth_bear[b_idx])
                if bull_bos >= 2:
                    breadth_sig += 1
                if bear_bos >= 2:
                    breadth_sig -= 1
            else:
                bull_bos = 0
                bear_bos = 0
        else:
            bull_bos = 0
            bear_bos = 0

        total = btc_prevday + btc_smma_sig + btc_vwap_sig + breadth_sig

        results.append({
            "timestamp": btc_ts[btc_i],
            "score": total,
            "btc_prevday": btc_prevday,
            "btc_smma": btc_smma_sig,
            "btc_vwap": btc_vwap_sig,
            "breadth": breadth_sig,
            "bull_bos": bull_bos,
            "bear_bos": bear_bos,
        })

    rdf = pd.DataFrame(results)
    rdf.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s — {len(rdf)} checkpoints saved to {OUTPUT_CSV}")

    # Distribution
    print(f"\nScore distribution:")
    for s in sorted(rdf["score"].unique()):
        cnt = (rdf["score"] == s).sum()
        pct = cnt / len(rdf) * 100
        print(f"  Score {s:+d}: {cnt:5d} ({pct:5.1f}%)")

    print(f"\nComponent stats:")
    for col in ["btc_prevday", "btc_smma", "btc_vwap", "breadth"]:
        vals = rdf[col]
        print(f"  {col:15s}: +1={int((vals==1).sum()):5d}  0={int((vals==0).sum()):5d}  -1={int((vals==-1).sum()):5d}")


if __name__ == "__main__":
    main()
