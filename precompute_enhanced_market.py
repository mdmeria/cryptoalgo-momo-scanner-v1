#!/usr/bin/env python3
"""
Enhanced Market Condition Scorer — based on ZCT daily bias framework.
Memory-optimized: loads only close+volume arrays aligned to BTC timestamps.

1. RETURN BUCKETS (Direction) — 4h return distribution → dir_score -2 to +2
2. SPAGHETTI R² (Strategy Type) — top mover linearity → strat_score -2 to +2
3. ACTIVITY (Volume modifier) — vol vs 7d baseline → act_mod -1 to +1
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os, time, gc
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_DIR = Path("datasets/binance_futures_1m")
OUTPUT_CSV = DATASET_DIR / "enhanced_market_cache.csv"
CHECKPOINT_INTERVAL = 30   # every 30 bars (30 min)
RET_WINDOW = 240           # 4h for returns
VOL_BASELINE = 10080       # 7 days in minutes
MIN_SYMBOLS = 30


def main():
    t0 = time.time()

    # Step 1: Load BTC as reference timeline
    btc_path = DATASET_DIR / "BTCUSDT_1m.csv"
    btc = pd.read_csv(str(btc_path), parse_dates=["timestamp"]).sort_values("timestamp")
    ref_ts = btc["timestamp"].values
    n_ref = len(ref_ts)
    print(f"Reference: BTC {n_ref} bars, {ref_ts[0]} to {ref_ts[-1]}")

    # Step 2: Load close & volume for each symbol using timestamp dict for alignment
    files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith("_1m.csv")])
    print(f"Loading {len(files)} symbols (close+volume only)...")

    # Use searchsorted for fast alignment
    ref_ts_i8 = ref_ts.astype("datetime64[ns]").astype(np.int64)

    close_matrix = {}
    vol_matrix = {}

    loaded = 0
    for f in files:
        sym = f.replace("_1m.csv", "")
        try:
            df = pd.read_csv(str(DATASET_DIR / f), usecols=["timestamp", "close", "volume"],
                            parse_dates=["timestamp"]).sort_values("timestamp")
            if len(df) < 500:
                continue

            sym_ts_i8 = df["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
            # Find matching indices in ref_ts
            idxs = np.searchsorted(ref_ts_i8, sym_ts_i8)
            # Only keep exact matches (within 1 minute = 60e9 nanoseconds)
            valid_mask = (idxs < n_ref) & (np.abs(ref_ts_i8[np.clip(idxs, 0, n_ref-1)] - sym_ts_i8) < 60_000_000_000)

            c_arr = np.full(n_ref, np.nan, dtype=np.float32)
            v_arr = np.full(n_ref, np.nan, dtype=np.float32)
            good_idxs = idxs[valid_mask]
            c_arr[good_idxs] = df["close"].values[valid_mask].astype(np.float32)
            v_arr[good_idxs] = (df["volume"].values[valid_mask] * df["close"].values[valid_mask]).astype(np.float32)

            n_valid = np.sum(~np.isnan(c_arr))
            if n_valid > n_ref * 0.3:
                close_matrix[sym] = c_arr
                vol_matrix[sym] = v_arr
                loaded += 1
        except Exception:
            continue

        if loaded % 50 == 0 and loaded > 0:
            elapsed = time.time() - t0
            print(f"  Loaded {loaded} symbols... ({elapsed:.0f}s)")

    gc.collect()
    syms = list(close_matrix.keys())
    print(f"  Total: {loaded} symbols loaded in {time.time()-t0:.0f}s")

    # Step 3: Compute at checkpoints
    start_idx = VOL_BASELINE + RET_WINDOW
    checkpoints = list(range(start_idx, n_ref, CHECKPOINT_INTERVAL))
    print(f"  {len(checkpoints)} checkpoints")

    results = []

    for ci, i in enumerate(checkpoints):
        # ── OBS 1: RETURN BUCKETS (4h returns) ──
        i_4h = i - RET_WINDOW
        if i_4h < 0:
            continue

        returns = []
        for sym in syms:
            c = close_matrix[sym]
            curr = c[i]
            past = c[i_4h]
            if np.isnan(curr) or np.isnan(past) or past <= 0:
                continue
            returns.append((curr - past) / past * 100)

        if len(returns) < MIN_SYMBOLS:
            continue

        ret_arr = np.array(returns, dtype=np.float32)
        n_total = len(ret_arr)
        pct_up = np.sum(ret_arr > 0) / n_total * 100
        pct_dn = np.sum(ret_arr < 0) / n_total * 100
        pct_up5 = np.sum(ret_arr > 5) / n_total * 100
        pct_dn5 = np.sum(ret_arr < -5) / n_total * 100
        avg_ret = np.mean(ret_arr)

        if pct_up > 80 or pct_up5 > 15:
            dir_score = 2
        elif pct_up > 65 or avg_ret > 1.5:
            dir_score = 1
        elif pct_dn > 80 or pct_dn5 > 15:
            dir_score = -2
        elif pct_dn > 65 or avg_ret < -1.5:
            dir_score = -1
        else:
            dir_score = 0

        # ── OBS 2: SPAGHETTI R² (top movers) ──
        abs_ret = np.abs(ret_arr)
        mover_thresh = max(np.percentile(abs_ret, 90), 2.0)

        r2_vals = []
        sym_idx = 0
        for sym in syms:
            c = close_matrix[sym]
            curr = c[i]
            past = c[i_4h]
            if np.isnan(curr) or np.isnan(past) or past <= 0:
                continue
            ret = abs(curr - past) / past * 100
            if ret < mover_thresh:
                continue

            # Get 4h window
            window = c[i_4h:i+1]
            valid_mask = ~np.isnan(window)
            if np.sum(valid_mask) < 120:
                continue
            window = window[valid_mask]

            # R² of linear fit
            x = np.arange(len(window))
            try:
                slope, intercept = np.polyfit(x, window, 1)
                pred = slope * x + intercept
                ss_res = np.sum((window - pred) ** 2)
                ss_tot = np.sum((window - np.mean(window)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                r2_vals.append(r2)
            except Exception:
                continue

        if len(r2_vals) >= 5:
            avg_r2 = float(np.mean(r2_vals))
            pct_clean = float(np.sum(np.array(r2_vals) > 0.7) / len(r2_vals) * 100)
            if avg_r2 > 0.75 or pct_clean > 70:
                strat_score = 2
            elif avg_r2 > 0.60 or pct_clean > 50:
                strat_score = 1
            elif avg_r2 < 0.30 or pct_clean < 20:
                strat_score = -2
            elif avg_r2 < 0.45 or pct_clean < 30:
                strat_score = -1
            else:
                strat_score = 0
        else:
            avg_r2 = 0.0; pct_clean = 0.0; strat_score = 0

        # ── OBS 3: ACTIVITY (vol vs 7d baseline) ──
        i_7d = i - VOL_BASELINE
        i_1d = i - 1440
        curr_vol = 0.0; base_vol = 0.0; vol_n = 0

        for sym in syms[:100]:
            v = vol_matrix[sym]
            cv = np.nansum(v[i_4h:i+1])
            bv = np.nansum(v[max(0,i_7d):i_1d])
            n_base_bars = max(1, i_1d - max(0, i_7d))
            base_avg_4h = bv / n_base_bars * RET_WINDOW if n_base_bars > 0 else 0

            if cv > 0 and base_avg_4h > 0:
                curr_vol += cv
                base_vol += base_avg_4h
                vol_n += 1

        if vol_n > 20 and base_vol > 0:
            vol_ratio = curr_vol / base_vol
            act_mod = 1 if vol_ratio > 1.5 else (-1 if vol_ratio < 0.6 else 0)
        else:
            vol_ratio = 1.0; act_mod = 0

        # ── QUADRANT ──
        if strat_score >= 1:
            if dir_score >= 1: quad = "momo_long"
            elif dir_score <= -1: quad = "momo_short"
            else: quad = "momo_neutral"
        elif strat_score <= -1:
            if dir_score >= 1: quad = "mr_long"
            elif dir_score <= -1: quad = "mr_short"
            else: quad = "mr_neutral"
        else:
            if dir_score >= 1: quad = "neutral_long"
            elif dir_score <= -1: quad = "neutral_short"
            else: quad = "neutral"

        results.append({
            "timestamp": str(pd.Timestamp(ref_ts[i])),
            "direction_score": dir_score, "strategy_score": strat_score,
            "activity_mod": act_mod, "quadrant": quad,
            "pct_up": round(float(pct_up), 1),
            "pct_strong_up": round(float(pct_up5), 1),
            "pct_strong_dn": round(float(pct_dn5), 1),
            "avg_return": round(float(avg_ret), 3),
            "avg_r2": round(avg_r2, 3),
            "pct_clean_staircase": round(pct_clean, 1),
            "n_top_movers": len(r2_vals),
            "vol_ratio": round(float(vol_ratio), 3),
        })

        if (ci + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{ci+1}/{len(checkpoints)}] {str(pd.Timestamp(ref_ts[i]))[:19]} "
                  f"dir={dir_score:+d} strat={strat_score:+d} act={act_mod:+d} "
                  f"quad={quad} | {elapsed:.0f}s")
            # Incremental save
            pd.DataFrame(results).to_csv(str(OUTPUT_CSV), index=False)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(str(OUTPUT_CSV), index=False)
        print(f"\nSaved {len(df)} checkpoints to {OUTPUT_CSV}")
        print(f"\nDirection: {df['direction_score'].value_counts().sort_index().to_dict()}")
        print(f"Strategy:  {df['strategy_score'].value_counts().sort_index().to_dict()}")
        print(f"Activity:  {df['activity_mod'].value_counts().sort_index().to_dict()}")
        print(f"Quadrant:  {df['quadrant'].value_counts().to_dict()}")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
