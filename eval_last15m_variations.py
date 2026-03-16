#!/usr/bin/env python3
"""
Evaluate last-15min gate variations on the existing trade set.

Computes all last-15min metrics (dir, net, abvMA, HL structure, pullback,
max consecutive run, volume trend) at each trade's entry timestamp, then
filters by various threshold combinations to find the best gate parameters.
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from backtest_momo_vwap_grind15_full import load_symbol_df, prepare_features


def compute_last15m_metrics(df: pd.DataFrame, entry_ts: pd.Timestamp, side: str) -> dict:
    """Compute all last-15min metrics at a specific entry timestamp."""
    idx = df.index.get_indexer([entry_ts], method="nearest")[0]
    nan_result = {
        "l15_dir_pct": np.nan, "l15_net_move": np.nan, "l15_abv_ma_pct": np.nan,
        "l15_hl_count": np.nan, "l15_max_pullback": np.nan, "l15_max_run": np.nan,
        "l15_vol_slope": np.nan, "l15_vol_ratio": np.nan, "l15_vol_steady": np.nan,
    }
    if idx < 20:
        return nan_result

    tail = df.iloc[idx - 14 : idx + 1]  # 15 bars ending at entry
    if len(tail) < 15:
        return nan_result

    is_long = side == "long"
    closes = tail["close"].to_numpy(dtype=float)
    opens = tail["open"].to_numpy(dtype=float)
    highs = tail["high"].to_numpy(dtype=float)
    lows = tail["low"].to_numpy(dtype=float)
    smma30 = tail["smma30"].to_numpy(dtype=float)
    volumes = tail["volume"].to_numpy(dtype=float)

    # Dir consistency
    if is_long:
        dir_pct = np.sum(closes > opens) / 15 * 100.0
    else:
        dir_pct = np.sum(closes < opens) / 15 * 100.0

    # Net move
    net = (closes[-1] - closes[0]) / max(abs(closes[0]), 1e-12) * 100.0
    net_dir = net if is_long else -net

    # AbvMA
    valid = ~np.isnan(smma30)
    if valid.sum() > 0:
        if is_long:
            abv_pct = np.sum(closes[valid] > smma30[valid]) / valid.sum() * 100.0
        else:
            abv_pct = np.sum(closes[valid] < smma30[valid]) / valid.sum() * 100.0
    else:
        abv_pct = 100.0

    # Higher-lows (long) or lower-highs (short)
    if is_long:
        hl_count = int(np.sum(lows[1:] > lows[:-1]))
    else:
        hl_count = int(np.sum(highs[1:] < highs[:-1]))

    # Max pullback
    if is_long:
        running_high = np.maximum.accumulate(closes)
        pullbacks = (running_high - lows) / np.maximum(running_high, 1e-12) * 100.0
    else:
        running_low = np.minimum.accumulate(closes)
        pullbacks = (highs - running_low) / np.maximum(running_low, 1e-12) * 100.0
    max_pullback = float(np.max(pullbacks))

    # Max consecutive same-direction
    dirs = closes > opens if is_long else closes < opens
    max_run = 0
    cur_run = 0
    for d in dirs:
        if d:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0

    # --- Volume metrics (ZCT Increasing/Steady Volume) ---
    # 1. Volume slope: linear regression slope of volume over 15 bars (normalized)
    vol_x = np.arange(15, dtype=float)
    vol_mean = np.mean(volumes)
    if vol_mean > 0 and not np.all(volumes == volumes[0]):
        vol_slope_raw, _ = np.polyfit(vol_x, volumes, 1)
        vol_slope = vol_slope_raw / vol_mean  # normalized: positive = increasing
    else:
        vol_slope = 0.0

    # 2. Volume ratio: last 5 bars avg / first 5 bars avg (>1 = increasing)
    first5_vol = np.mean(volumes[:5]) if np.mean(volumes[:5]) > 0 else 1e-12
    last5_vol = np.mean(volumes[-5:])
    vol_ratio = last5_vol / first5_vol

    # 3. Volume steadiness: what % of bars have volume >= 50% of the 15-bar avg
    #    High = steady participation, Low = sporadic spikes
    vol_threshold = vol_mean * 0.5
    vol_steady = np.sum(volumes >= vol_threshold) / 15 * 100.0

    return {
        "l15_dir_pct": dir_pct,
        "l15_net_move": net_dir,
        "l15_abv_ma_pct": abv_pct,
        "l15_hl_count": hl_count,
        "l15_max_pullback": max_pullback,
        "l15_max_run": max_run,
        "l15_vol_slope": vol_slope,
        "l15_vol_ratio": vol_ratio,
        "l15_vol_steady": vol_steady,
    }


def process_symbol(args):
    symbol, csv_path, trades_for_sym = args
    p = Path(csv_path)
    if not p.exists():
        return []
    try:
        raw_df = load_symbol_df(str(p))
        df = prepare_features(raw_df)
    except Exception:
        return []

    results = []
    for _, trade in trades_for_sym.iterrows():
        ts = pd.Timestamp(trade["timestamp"], tz="UTC")
        metrics = compute_last15m_metrics(df, ts, trade["side"])
        row = {
            "symbol": trade["symbol"],
            "timestamp": trade["timestamp"],
            "side": trade["side"],
            "outcome": trade["outcome"],
            "pnl_pct": trade["pnl_pct"],
        }
        row.update(metrics)
        results.append(row)
    return results


def stats(subset, label=""):
    n = len(subset)
    if n == 0:
        return
    tp = (subset["outcome"] == "TP").sum()
    sl = (subset["outcome"] == "SL").sum()
    op = (subset["outcome"] == "OPEN").sum()
    wr = tp / n * 100 if n > 0 else 0
    total_pnl = subset["pnl_pct"].sum()
    avg_pnl = subset["pnl_pct"].mean()
    print(f"  {label:50s} | {n:4d} trades | WR {wr:5.1f}% | TP {tp:3d} SL {sl:3d} OPEN {op:3d} | PnL {total_pnl:+7.2f}% (avg {avg_pnl:+.3f}%)")


def main():
    dataset_dir = Path("datasets/spot_mar2_mar15")
    manifest = pd.read_csv(dataset_dir / "dataset_manifest.csv")
    manifest = manifest[manifest["ok"].astype(str).str.lower() == "true"]

    trades = pd.read_csv("momo_run_trade_list.csv")
    print(f"Loaded {len(trades)} trades from momo_run_trade_list.csv")

    path_map = dict(zip(manifest["symbol"], manifest["path"]))
    work_items = []
    for sym in trades["symbol"].unique():
        csv_path = path_map.get(sym)
        if csv_path:
            sym_trades = trades[trades["symbol"] == sym]
            work_items.append((sym, csv_path, sym_trades))

    print(f"Processing {len(work_items)} symbols...")

    all_results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_symbol, item): item[0] for item in work_items}
        for fut in as_completed(futures):
            all_results.extend(fut.result())

    rdf = pd.DataFrame(all_results)
    rdf.to_csv("momo_run_last15m_metrics.csv", index=False)
    print(f"\nComputed metrics for {len(rdf)} trades")
    print(f"Saved to momo_run_last15m_metrics.csv\n")

    valid = rdf.dropna(subset=["l15_dir_pct"])

    # =========================================================================
    print("=" * 120)
    print("BASELINE")
    print("=" * 120)
    stats(valid, "All trades")

    # =========================================================================
    print(f"\n{'='*120}")
    print("BY LAST-15M DIRECTIONAL %")
    print("=" * 120)
    for lo, hi in [(0, 47), (47, 53), (53, 60), (60, 67), (67, 80), (80, 101)]:
        subset = valid[(valid["l15_dir_pct"] >= lo) & (valid["l15_dir_pct"] < hi)]
        stats(subset, f"Dir {lo}-{hi}%")

    # =========================================================================
    print(f"\n{'='*120}")
    print("BY LAST-15M HIGHER-LOW / LOWER-HIGH COUNT (out of 14)")
    print("=" * 120)
    for lo, hi in [(0, 6), (6, 8), (8, 10), (10, 12), (12, 15)]:
        subset = valid[(valid["l15_hl_count"] >= lo) & (valid["l15_hl_count"] < hi)]
        stats(subset, f"HL {lo}-{hi}")

    # =========================================================================
    print(f"\n{'='*120}")
    print("BY LAST-15M MAX PULLBACK %")
    print("=" * 120)
    for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.75), (0.75, 999)]:
        hi_l = f"{hi:.2f}" if hi < 900 else ">0.75"
        subset = valid[(valid["l15_max_pullback"] >= lo) & (valid["l15_max_pullback"] < hi)]
        stats(subset, f"Pullback {lo:.2f}-{hi_l}%")

    # =========================================================================
    print(f"\n{'='*120}")
    print("BY LAST-15M MAX CONSECUTIVE RUN")
    print("=" * 120)
    for r in range(2, 10):
        subset = valid[valid["l15_max_run"] == r]
        stats(subset, f"MaxRun = {r}")

    # =========================================================================
    print(f"\n{'='*120}")
    print("BY LAST-15M VOLUME TREND (slope, ratio, steadiness)")
    print("=" * 120)
    print("\n  --- Volume Slope (normalized) ---")
    for lo, hi in [(-999, -0.05), (-0.05, 0), (0, 0.05), (0.05, 0.15), (0.15, 999)]:
        lo_l = f"{lo:+.2f}" if lo > -900 else "<-0.05"
        hi_l = f"{hi:+.2f}" if hi < 900 else ">0.15"
        subset = valid[(valid["l15_vol_slope"] >= lo) & (valid["l15_vol_slope"] < hi)]
        stats(subset, f"VolSlope {lo_l} to {hi_l}")

    print("\n  --- Volume Ratio (last5 / first5) ---")
    for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 999)]:
        hi_l = f"{hi:.1f}" if hi < 900 else ">2.0"
        subset = valid[(valid["l15_vol_ratio"] >= lo) & (valid["l15_vol_ratio"] < hi)]
        stats(subset, f"VolRatio {lo:.1f}-{hi_l}")

    print("\n  --- Volume Steadiness (% bars >= 50% avg) ---")
    for lo, hi in [(0, 60), (60, 73), (73, 87), (87, 101)]:
        subset = valid[(valid["l15_vol_steady"] >= lo) & (valid["l15_vol_steady"] < hi)]
        stats(subset, f"VolSteady {lo}-{hi}%")

    # =========================================================================
    print(f"\n{'='*120}")
    print("COMBINED GATE VARIATIONS (staircase structure + original metrics)")
    print("=" * 120)

    variations = [
        # (dir_min, net_min, abv_min, hl_min, pb_max, run_max, label)
        # Original metrics only
        (60, -999, 0,  0, 999, 99, "Dir>=60%"),
        (53, -999, 0,  0, 999, 99, "Dir>=53%"),

        # Staircase structure only
        (0, -999, 0,  8, 0.4, 5, "HL>=8, PB<=0.4%, Run<=5"),
        (0, -999, 0,  7, 0.5, 5, "HL>=7, PB<=0.5%, Run<=5"),
        (0, -999, 0,  8, 0.5, 6, "HL>=8, PB<=0.5%, Run<=6"),
        (0, -999, 0, 10, 0.4, 5, "HL>=10, PB<=0.4%, Run<=5"),

        # PENGU-like combos (dir + structure)
        (53, -999, 0,  8, 0.4, 5, "Dir>=53%, HL>=8, PB<=0.4%, Run<=5"),
        (60, -999, 0,  8, 0.4, 5, "Dir>=60%, HL>=8, PB<=0.4%, Run<=5"),
        (53, -999, 0,  7, 0.5, 5, "Dir>=53%, HL>=7, PB<=0.5%, Run<=5"),
        (60, -999, 0,  7, 0.5, 5, "Dir>=60%, HL>=7, PB<=0.5%, Run<=5"),
        (53, -999, 0,  8, 0.5, 5, "Dir>=53%, HL>=8, PB<=0.5%, Run<=5"),
        (60, -999, 0,  8, 0.5, 5, "Dir>=60%, HL>=8, PB<=0.5%, Run<=5"),
        (53, -999, 80, 8, 0.4, 5, "Dir>=53%, AbvMA>=80%, HL>=8, PB<=0.4%, Run<=5"),
        (60, -999, 80, 8, 0.4, 5, "Dir>=60%, AbvMA>=80%, HL>=8, PB<=0.4%, Run<=5"),

        # Looser structure
        (53, -999, 0,  6, 0.5, 6, "Dir>=53%, HL>=6, PB<=0.5%, Run<=6"),
        (60, -999, 0,  6, 0.5, 6, "Dir>=60%, HL>=6, PB<=0.5%, Run<=6"),
        (53, -999, 0,  7, 0.6, 6, "Dir>=53%, HL>=7, PB<=0.6%, Run<=6"),
        (60, -999, 0,  7, 0.6, 6, "Dir>=60%, HL>=7, PB<=0.6%, Run<=6"),

        # Tighter structure
        (60, -999, 0,  9, 0.4, 5, "Dir>=60%, HL>=9, PB<=0.4%, Run<=5"),
        (60, -999, 0, 10, 0.35, 4, "Dir>=60%, HL>=10, PB<=0.35%, Run<=4"),

        # With net move
        (53, 0.2, 0,  8, 0.4, 5, "Dir>=53%, Net>=0.2%, HL>=8, PB<=0.4%, Run<=5"),
        (60, 0.2, 0,  8, 0.4, 5, "Dir>=60%, Net>=0.2%, HL>=8, PB<=0.4%, Run<=5"),
        (53, 0.5, 0,  7, 0.5, 5, "Dir>=53%, Net>=0.5%, HL>=7, PB<=0.5%, Run<=5"),
        (60, 0.5, 0,  7, 0.5, 5, "Dir>=60%, Net>=0.5%, HL>=7, PB<=0.5%, Run<=5"),
    ]

    for dir_min, net_min, abv_min, hl_min, pb_max, run_max, label in variations:
        mask = pd.Series(True, index=valid.index)
        if dir_min > 0:
            mask &= valid["l15_dir_pct"] >= dir_min
        if net_min > -900:
            mask &= valid["l15_net_move"] >= net_min
        if abv_min > 0:
            mask &= valid["l15_abv_ma_pct"] >= abv_min
        if hl_min > 0:
            mask &= valid["l15_hl_count"] >= hl_min
        if pb_max < 900:
            mask &= valid["l15_max_pullback"] <= pb_max
        if run_max < 90:
            mask &= valid["l15_max_run"] <= run_max
        subset = valid[mask]
        stats(subset, label)

    # =========================================================================
    # Print full metrics for reference trades
    print(f"\n{'='*120}")
    print("REFERENCE TRADE DETAILS (all metrics)")
    print("=" * 120)
    for _, row in valid.iterrows():
        if row["outcome"] == "TP":
            print(f"  {row['symbol']:18s} {row['timestamp']:28s} {row['side']:5s} TP PnL={row['pnl_pct']:+.3f}% | "
                  f"Dir={row['l15_dir_pct']:.0f}% Net={row['l15_net_move']:+.3f}% HL={row['l15_hl_count']:.0f} "
                  f"PB={row['l15_max_pullback']:.3f}% Run={row['l15_max_run']:.0f} | "
                  f"VolSlope={row['l15_vol_slope']:+.3f} VolRatio={row['l15_vol_ratio']:.2f} VolSteady={row['l15_vol_steady']:.0f}%")


if __name__ == "__main__":
    main()
