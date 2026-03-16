#!/usr/bin/env python3
"""
Backtest Choppy Range MR setups from scanner output.

Reads mr_choppy_setups.csv and simulates trades using 1m candle data.
"""

from __future__ import annotations

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def simulate_trade(df_1m: pd.DataFrame, entry_ts: pd.Timestamp,
                   side: str, entry_price: float,
                   sl: float, tp: float,
                   max_bars: int = 120) -> dict:
    """Simulate a trade on 1m data. Returns outcome dict."""
    mask = df_1m["timestamp"] >= entry_ts
    bars = df_1m.loc[mask].head(max_bars)

    if len(bars) == 0:
        return {"outcome": "NO_DATA", "bars_held": 0, "exit_price": entry_price}

    for idx, bar in bars.iterrows():
        if side == "long":
            if bar["low"] <= sl:
                return {"outcome": "SL", "bars_held": idx - bars.index[0] + 1,
                        "exit_price": sl, "exit_ts": str(bar["timestamp"])}
            if bar["high"] >= tp:
                return {"outcome": "TP", "bars_held": idx - bars.index[0] + 1,
                        "exit_price": tp, "exit_ts": str(bar["timestamp"])}
        else:
            if bar["high"] >= sl:
                return {"outcome": "SL", "bars_held": idx - bars.index[0] + 1,
                        "exit_price": sl, "exit_ts": str(bar["timestamp"])}
            if bar["low"] <= tp:
                return {"outcome": "TP", "bars_held": idx - bars.index[0] + 1,
                        "exit_price": tp, "exit_ts": str(bar["timestamp"])}

    # Still open after max_bars
    last_close = float(bars.iloc[-1]["close"])
    return {"outcome": "OPEN", "bars_held": len(bars), "exit_price": last_close,
            "exit_ts": str(bars.iloc[-1]["timestamp"])}


def main():
    setups_file = "mr_choppy_setups.csv"
    dataset_dir = Path("datasets/momo_1m_mar2_mar14")

    if not os.path.exists(setups_file):
        print(f"Error: {setups_file} not found. Run scan_mean_reversion.py first.")
        return

    setups = pd.read_csv(setups_file)
    print(f"Loaded {len(setups)} setups from {setups_file}")

    # Filter by DPS confidence
    for min_dps in [0, 3, 4, 5, 6]:
        subset = setups[setups["dps_total"] >= min_dps]
        print(f"  DPS >= {min_dps}: {len(subset)} setups")

    # Run backtest on DPS >= 3 (low confidence and above)
    bt_setups = setups[setups["dps_total"] >= 3].copy()
    print(f"\nBacktesting {len(bt_setups)} setups with DPS >= 3...")

    # Cache loaded dataframes
    df_cache = {}
    results = []

    for i, (_, setup) in enumerate(bt_setups.iterrows()):
        sym = setup["symbol"]

        if sym not in df_cache:
            fpath = dataset_dir / f"{sym}_1m.csv"
            if not fpath.exists():
                continue
            df_cache[sym] = pd.read_csv(str(fpath), parse_dates=["timestamp"]).sort_values("timestamp")

        df_1m = df_cache[sym]
        entry_ts = pd.Timestamp(setup["timestamp"])

        result = simulate_trade(
            df_1m, entry_ts,
            side=setup["side"],
            entry_price=setup["entry"],
            sl=setup["sl"],
            tp=setup["tp"],
            max_bars=120,
        )

        result["symbol"] = sym
        result["side"] = setup["side"]
        result["entry"] = setup["entry"]
        result["sl"] = setup["sl"]
        result["tp"] = setup["tp"]
        result["sl_pct"] = setup["sl_pct"]
        result["tp_pct"] = setup["tp_pct"]
        result["rr"] = setup.get("rr", 1.0)
        result["dps_total"] = setup["dps_total"]
        result["dps_confidence"] = setup["dps_confidence"]
        result["noise_level"] = setup["noise_level"]
        result["touches"] = setup["touches"]
        result["break_pct"] = setup.get("break_pct", 0)
        result["range_duration_hrs"] = setup["range_duration_hrs"]
        result["range_width_pct"] = setup["range_width_pct"]
        result["pre_chop_trend"] = setup.get("pre_chop_trend", "unclear")
        result["dps_v2_label"] = setup["dps_v2_label"]
        result["dps_v3_vol_trend"] = setup["dps_v3_vol_trend"]
        result["timestamp"] = setup["timestamp"]
        results.append(result)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(bt_setups)}] processed...")

    rdf = pd.DataFrame(results)
    rdf.to_csv("mr_choppy_backtest.csv", index=False)

    print(f"\n{'='*70}")
    print(f"CHOPPY RANGE MR BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"Total trades simulated: {len(rdf)}")

    # Overall stats
    tp_n = (rdf["outcome"] == "TP").sum()
    sl_n = (rdf["outcome"] == "SL").sum()
    op_n = (rdf["outcome"] == "OPEN").sum()
    nd_n = (rdf["outcome"] == "NO_DATA").sum()
    closed = tp_n + sl_n
    wr = tp_n / closed * 100 if closed > 0 else 0
    print(f"\nOverall: {tp_n} TP / {sl_n} SL / {op_n} OPEN / {nd_n} NO_DATA")
    print(f"Win Rate: {wr:.1f}%")

    # PnL calculation
    pnl_list = []
    for _, r in rdf.iterrows():
        if r["outcome"] == "TP":
            pnl_list.append(r["tp_pct"])
        elif r["outcome"] == "SL":
            pnl_list.append(-r["sl_pct"])
        else:
            if r["side"] == "long":
                pnl_list.append((r["exit_price"] - r["entry"]) / r["entry"] * 100)
            else:
                pnl_list.append((r["entry"] - r["exit_price"]) / r["entry"] * 100)
    rdf["pnl_pct"] = pnl_list

    print(f"Total PnL: {rdf['pnl_pct'].sum():.2f}%")
    print(f"Avg PnL/trade: {rdf['pnl_pct'].mean():.3f}%")
    print(f"Avg bars held: {rdf['bars_held'].mean():.0f}")

    # By DPS score
    print(f"\n--- By DPS Score ---")
    print(f"{'DPS':>4s}  {'Trades':>6s}  {'TP':>4s}  {'SL':>4s}  {'OPEN':>4s}  "
          f"{'WR%':>6s}  {'TotPnL':>8s}  {'AvgPnL':>8s}")
    print("-" * 65)
    for dps in sorted(rdf["dps_total"].unique()):
        sub = rdf[rdf["dps_total"] == dps]
        tp = (sub["outcome"] == "TP").sum()
        sl = (sub["outcome"] == "SL").sum()
        op = (sub["outcome"] == "OPEN").sum()
        wr = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        tot = sub["pnl_pct"].sum()
        avg = sub["pnl_pct"].mean()
        print(f"{dps:4d}  {len(sub):6d}  {tp:4d}  {sl:4d}  {op:4d}  "
              f"{wr:6.1f}  {tot:8.2f}  {avg:8.4f}")

    # By side
    print(f"\n--- By Side ---")
    for side in ["long", "short"]:
        sub = rdf[rdf["side"] == side]
        tp = (sub["outcome"] == "TP").sum()
        sl = (sub["outcome"] == "SL").sum()
        wr = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        tot = sub["pnl_pct"].sum()
        print(f"  {side:5s}: {len(sub)} trades, {tp} TP / {sl} SL, WR={wr:.1f}%, PnL={tot:.2f}%")

    # By noise level
    print(f"\n--- By Noise Level ---")
    for noise in ["low", "medium", "high"]:
        sub = rdf[rdf["noise_level"] == noise]
        if len(sub) == 0:
            continue
        tp = (sub["outcome"] == "TP").sum()
        sl = (sub["outcome"] == "SL").sum()
        wr = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        tot = sub["pnl_pct"].sum()
        print(f"  {noise:6s}: {len(sub)} trades, {tp} TP / {sl} SL, WR={wr:.1f}%, PnL={tot:.2f}%")

    # By approach quality
    print(f"\n--- By Approach ---")
    for app in ["spike", "unclear", "grind"]:
        sub = rdf[rdf["dps_v2_label"] == app]
        if len(sub) == 0:
            continue
        tp = (sub["outcome"] == "TP").sum()
        sl = (sub["outcome"] == "SL").sum()
        wr = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        tot = sub["pnl_pct"].sum()
        print(f"  {app:8s}: {len(sub)} trades, {tp} TP / {sl} SL, WR={wr:.1f}%, PnL={tot:.2f}%")

    # By volume trend
    print(f"\n--- By Volume Trend ---")
    for vol in ["flat", "decreasing", "increasing"]:
        sub = rdf[rdf["dps_v3_vol_trend"] == vol]
        if len(sub) == 0:
            continue
        tp = (sub["outcome"] == "TP").sum()
        sl = (sub["outcome"] == "SL").sum()
        wr = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        tot = sub["pnl_pct"].sum()
        print(f"  {vol:11s}: {len(sub)} trades, {tp} TP / {sl} SL, WR={wr:.1f}%, PnL={tot:.2f}%")

    # By range duration
    print(f"\n--- By Range Duration ---")
    for dur_min, dur_max, label in [(2, 3, "2-3h"), (3, 5, "3-5h"), (5, 9, "5-8h")]:
        sub = rdf[(rdf["range_duration_hrs"] >= dur_min) & (rdf["range_duration_hrs"] < dur_max)]
        if len(sub) == 0:
            continue
        tp = (sub["outcome"] == "TP").sum()
        sl = (sub["outcome"] == "SL").sum()
        wr = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        tot = sub["pnl_pct"].sum()
        print(f"  {label:5s}: {len(sub)} trades, {tp} TP / {sl} SL, WR={wr:.1f}%, PnL={tot:.2f}%")

    # By pre-chop trend
    if "pre_chop_trend" in rdf.columns:
        print(f"\n--- By Pre-Chop Trend ---")
        for trend in ["up", "down", "unclear"]:
            sub = rdf[rdf["pre_chop_trend"] == trend]
            if len(sub) == 0:
                continue
            tp = (sub["outcome"] == "TP").sum()
            sl = (sub["outcome"] == "SL").sum()
            wr = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
            tot = sub["pnl_pct"].sum()
            print(f"  {trend:7s}: {len(sub)} trades, {tp} TP / {sl} SL, WR={wr:.1f}%, PnL={tot:.2f}%")

    # TP/SL/RR stats
    print(f"\n--- TP/SL/RR Distribution ---")
    print(f"  SL%:  mean={rdf['sl_pct'].mean():.2f}  median={rdf['sl_pct'].median():.2f}  "
          f"min={rdf['sl_pct'].min():.2f}  max={rdf['sl_pct'].max():.2f}")
    print(f"  TP%:  mean={rdf['tp_pct'].mean():.2f}  median={rdf['tp_pct'].median():.2f}  "
          f"min={rdf['tp_pct'].min():.2f}  max={rdf['tp_pct'].max():.2f}")
    if "rr" in rdf.columns:
        print(f"  RR:   mean={rdf['rr'].mean():.2f}  median={rdf['rr'].median():.2f}  "
              f"min={rdf['rr'].min():.2f}  max={rdf['rr'].max():.2f}")

    # By level integrity (break %)
    if "break_pct" in rdf.columns:
        print(f"\n--- By Level Integrity (break %) ---")
        for bmax, label in [(1, "0-1%"), (3, "1-3%"), (5, "3-5%")]:
            bmin = 0 if bmax == 1 else (1 if bmax == 3 else 3)
            sub = rdf[(rdf["break_pct"] >= bmin) & (rdf["break_pct"] < bmax)]
            if len(sub) == 0:
                continue
            tp = (sub["outcome"] == "TP").sum()
            sl = (sub["outcome"] == "SL").sum()
            wr = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
            tot = sub["pnl_pct"].sum()
            print(f"  {label:5s}: {len(sub)} trades, {tp} TP / {sl} SL, WR={wr:.1f}%, PnL={tot:.2f}%")

    print(f"\nOutput: mr_choppy_backtest.csv")


if __name__ == "__main__":
    main()
