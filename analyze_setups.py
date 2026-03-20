#!/usr/bin/env python3
"""
Setup Analyzer — Comprehensive scan of MR and Momo setups over historical data.

Scans all symbols in a dataset, finds every setup that would have triggered,
simulates the trade, and categorizes results by multiple dimensions to help
identify which setup types perform best.

Usage:
  python analyze_setups.py [--dataset datasets/spot_mar2_mar15] [--days 7]
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scan_mean_reversion import (
    MRSettings,
    check_mr_gates_at_bar,
    check_strict_mr_gates_at_bar,
)
from backtest_momo_vwap_grind15_full import (
    GateSettings as MomoGateSettings,
    check_momo_gates_at_bar,
    prepare_features,
)

MAX_BARS = 120  # timeout


def simulate_trade(df, entry_idx, side, entry_price, sl, tp, max_bars=MAX_BARS):
    end_idx = min(entry_idx + max_bars, len(df))
    for i in range(entry_idx, end_idx):
        bar = df.iloc[i]
        if side == "long":
            if bar["low"] <= sl:
                return {"outcome": "SL", "bars_held": i - entry_idx + 1,
                        "exit_price": sl, "exit_ts": str(bar["timestamp"])}
            if bar["high"] >= tp:
                return {"outcome": "TP", "bars_held": i - entry_idx + 1,
                        "exit_price": tp, "exit_ts": str(bar["timestamp"])}
        else:
            if bar["high"] >= sl:
                return {"outcome": "SL", "bars_held": i - entry_idx + 1,
                        "exit_price": sl, "exit_ts": str(bar["timestamp"])}
            if bar["low"] <= tp:
                return {"outcome": "TP", "bars_held": i - entry_idx + 1,
                        "exit_price": tp, "exit_ts": str(bar["timestamp"])}
    last = df.iloc[end_idx - 1]
    return {"outcome": "TIMEOUT", "bars_held": end_idx - entry_idx,
            "exit_price": float(last["close"]), "exit_ts": str(last["timestamp"])}


def calc_pnl(outcome, side, entry, exit_price, sl_pct, tp_pct):
    if outcome == "TP":
        return tp_pct
    elif outcome == "SL":
        return -sl_pct
    else:
        if side == "long":
            return (exit_price - entry) / entry * 100
        else:
            return (entry - exit_price) / entry * 100


def scan_mr(df, cfg, scan_start, scan_end, symbol, cooldown_bars=60):
    trades = []
    cooldown_reg = -1
    cooldown_strict = -1

    for i in range(scan_start, scan_end):
        # Regular MR
        if i > cooldown_reg:
            r = check_mr_gates_at_bar(df, i, cfg)
            if r["passed"]:
                sim = simulate_trade(df, i, r["side"], r["entry"], r["sl"], r["tp"])
                pnl = calc_pnl(sim["outcome"], r["side"], r["entry"],
                               sim["exit_price"], r["sl_pct"], r["tp_pct"])
                trades.append({
                    "symbol": symbol,
                    "strategy": "MR",
                    "timestamp": str(df.iloc[i]["timestamp"]),
                    "hour": df.iloc[i]["timestamp"].hour,
                    "side": r["side"],
                    "entry": r["entry"],
                    "sl_pct": r["sl_pct"],
                    "tp_pct": r["tp_pct"],
                    "rr": r.get("rr", 1.0),
                    "dps_total": r["dps_total"],
                    "dps_confidence": r["dps_confidence"],
                    "dps_v1_duration": r.get("dps_v1", 0),
                    "dps_v2_approach": r.get("dps_v2_label", "?"),
                    "dps_v3_vol": r.get("dps_v3_vol_trend", "?"),
                    "noise_level": r.get("noise_level", "?"),
                    "pre_chop_trend": r.get("pre_chop_trend", "?"),
                    "touches": r.get("touches", 0),
                    "break_pct": r.get("break_pct", 0),
                    "range_width_pct": r.get("range_width_pct", 0),
                    "range_duration_hrs": r.get("range_duration_hrs", 0),
                    "smma30_slope": r.get("smma30_slope", 0),
                    "smma120_slope": r.get("smma120_slope", 0),
                    "outcome": sim["outcome"],
                    "pnl_pct": pnl,
                    "bars_held": sim["bars_held"],
                })
                cooldown_reg = i + cooldown_bars

        # Strict MR
        if i > cooldown_strict:
            r = check_strict_mr_gates_at_bar(df, i, cfg)
            if r["passed"]:
                sim = simulate_trade(df, i, r["side"], r["entry"], r["sl"], r["tp"])
                pnl = calc_pnl(sim["outcome"], r["side"], r["entry"],
                               sim["exit_price"], r["sl_pct"], r["tp_pct"])
                trades.append({
                    "symbol": symbol,
                    "strategy": "Strict_MR",
                    "timestamp": str(df.iloc[i]["timestamp"]),
                    "hour": df.iloc[i]["timestamp"].hour,
                    "side": r["side"],
                    "entry": r["entry"],
                    "sl_pct": r["sl_pct"],
                    "tp_pct": r["tp_pct"],
                    "rr": r.get("rr", 1.0),
                    "dps_total": r["dps_total"],
                    "dps_confidence": r["dps_confidence"],
                    "dps_v1_duration": r.get("dps_v1", 0),
                    "dps_v2_approach": r.get("dps_v2_label", "?"),
                    "dps_v3_vol": r.get("dps_v3_vol_trend", "?"),
                    "noise_level": r.get("noise_level", "?"),
                    "pre_chop_trend": r.get("pre_chop_trend", "?"),
                    "touches": r.get("touches", 0),
                    "break_pct": r.get("break_pct", 0),
                    "range_width_pct": r.get("range_width_pct", 0),
                    "range_duration_hrs": r.get("range_duration_hrs", 0),
                    "smma30_slope": r.get("smma30_slope", 0),
                    "smma120_slope": r.get("smma120_slope", 0),
                    "outcome": sim["outcome"],
                    "pnl_pct": pnl,
                    "bars_held": sim["bars_held"],
                })
                cooldown_strict = i + cooldown_bars

    return trades


def scan_momo(df_prepped, cfg, symbol):
    trades = []
    cooldown = -1

    for i in range(len(df_prepped)):
        if i <= cooldown:
            continue
        slice_df = df_prepped.iloc[:i + 1]
        if len(slice_df) < 200:
            continue

        for side in ["long", "short"]:
            r = check_momo_gates_at_bar(slice_df, side, cfg)
            if not r["passed"]:
                continue

            # Simulate using integer-indexed df
            entry_price = r["entry"]
            sim = simulate_trade(
                df_prepped.reset_index(), i, side,
                entry_price, r["sl"], r["tp"])
            pnl = calc_pnl(sim["outcome"], side, entry_price,
                           sim["exit_price"], r["sl_pct"], r["tp_pct"])

            ts = df_prepped.index[i]
            trades.append({
                "symbol": symbol,
                "strategy": "Momo",
                "timestamp": str(ts),
                "hour": ts.hour,
                "side": side,
                "entry": entry_price,
                "sl_pct": r["sl_pct"],
                "tp_pct": r["tp_pct"],
                "rr": r.get("rr", 1.0),
                "dps_total": r["dps_total"],
                "dps_confidence": r["dps_confidence"],
                "dps_v1_duration": r.get("dps_v1", 0),
                "dps_v2_approach": r.get("approach", "?"),
                "dps_v3_vol": r.get("vol_trend", "?"),
                "noise_level": "n/a",
                "pre_chop_trend": "n/a",
                "touches": 0,
                "break_pct": 0,
                "range_width_pct": 0,
                "range_duration_hrs": r.get("duration_hrs", 0),
                "smma30_slope": 0,
                "smma120_slope": 0,
                "outcome": sim["outcome"],
                "pnl_pct": pnl,
                "bars_held": sim["bars_held"],
            })
            cooldown = i + 60
            break  # one direction per bar

    return trades


def print_breakdown(df, field, label, top_n=20):
    """Print WR and PnL breakdown by a categorical field."""
    vals = df[field].unique()
    if len(vals) > top_n:
        # Show top by trade count
        vals = df[field].value_counts().head(top_n).index
    rows = []
    for v in sorted(vals, key=str):
        sub = df[df[field] == v]
        tp = (sub["outcome"] == "TP").sum()
        sl = (sub["outcome"] == "SL").sum()
        to = (sub["outcome"] == "TIMEOUT").sum()
        wr = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        pnl = sub["pnl_pct"].sum()
        rows.append((v, len(sub), tp, sl, to, wr, pnl))

    print(f"\n  By {label}:")
    print(f"    {'Value':20s} {'Trades':>6s} {'TP':>4s} {'SL':>4s} {'TO':>4s} {'WR':>6s} {'PnL':>8s}")
    print(f"    {'-'*55}")
    for v, n, tp, sl, to, wr, pnl in sorted(rows, key=lambda x: -x[6]):
        marker = " <-- best" if pnl == max(r[6] for r in rows) else ""
        print(f"    {str(v):20s} {n:6d} {tp:4d} {sl:4d} {to:4d} {wr:5.1f}% {pnl:+8.2f}{marker}")


def print_report(all_trades):
    df = pd.DataFrame(all_trades)
    if len(df) == 0:
        print("No trades found.")
        return

    print(f"\n{'='*80}")
    print(f"SETUP ANALYSIS — {len(df)} total trades")
    print(f"{'='*80}")

    for strat in sorted(df["strategy"].unique()):
        sdf = df[df["strategy"] == strat].copy()
        tp = (sdf["outcome"] == "TP").sum()
        sl = (sdf["outcome"] == "SL").sum()
        to = (sdf["outcome"] == "TIMEOUT").sum()
        wr = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        pnl = sdf["pnl_pct"].sum()
        avg = sdf["pnl_pct"].mean()

        print(f"\n{'#'*70}")
        print(f"  {strat} — {len(sdf)} trades, {tp} TP / {sl} SL / {to} TO")
        print(f"  WR={wr:.1f}%, PnL={pnl:+.2f}%, Avg={avg:+.3f}%/trade")
        print(f"{'#'*70}")

        # By side
        print_breakdown(sdf, "side", "Side")

        # By DPS
        print_breakdown(sdf, "dps_total", "DPS Score")

        # By DPS confidence
        print_breakdown(sdf, "dps_confidence", "DPS Confidence")

        # By approach (V2)
        print_breakdown(sdf, "dps_v2_approach", "Approach (V2)")

        # By volume trend (V3)
        print_breakdown(sdf, "dps_v3_vol", "Volume Trend (V3)")

        if strat in ("MR", "Strict_MR"):
            # By pre-chop trend
            print_breakdown(sdf, "pre_chop_trend", "Pre-Chop Trend")

            # By noise level
            print_breakdown(sdf, "noise_level", "Noise Level")

            # By touch count
            print_breakdown(sdf, "touches", "Touch Count")

            # By range width bucket
            sdf["width_bucket"] = pd.cut(sdf["range_width_pct"],
                                          bins=[0, 1.5, 2.0, 2.5, 3.0, 4.0],
                                          labels=["1.0-1.5%", "1.5-2.0%", "2.0-2.5%",
                                                  "2.5-3.0%", "3.0-4.0%"])
            print_breakdown(sdf, "width_bucket", "Range Width")

            # By range duration bucket
            sdf["dur_bucket"] = pd.cut(sdf["range_duration_hrs"],
                                        bins=[0, 2, 4, 6, 8, 12, 24],
                                        labels=["0-2h", "2-4h", "4-6h", "6-8h", "8-12h", "12-24h"])
            print_breakdown(sdf, "dur_bucket", "Range Duration")

        # By hour of day
        sdf["hour_label"] = sdf["hour"].apply(lambda h: f"{h:02d}:00")
        print_breakdown(sdf, "hour_label", "Hour (UTC)")

        # Combined: DPS + Side
        sdf["dps_side"] = sdf.apply(lambda r: f"DPS{r['dps_total']}_{r['side']}", axis=1)
        print_breakdown(sdf, "dps_side", "DPS x Side")

        if strat in ("MR", "Strict_MR"):
            # Combined: pre_chop_trend + side
            sdf["trend_side"] = sdf.apply(
                lambda r: f"{r['pre_chop_trend']}_{r['side']}", axis=1)
            print_breakdown(sdf, "trend_side", "Pre-Chop Trend x Side")

            # Combined: approach + pre_chop_trend
            sdf["approach_trend"] = sdf.apply(
                lambda r: f"{r['dps_v2_approach']}_{r['pre_chop_trend']}", axis=1)
            print_breakdown(sdf, "approach_trend", "Approach x Pre-Chop Trend")

            # Combined: noise + DPS
            sdf["noise_dps"] = sdf.apply(
                lambda r: f"{r['noise_level']}_DPS{r['dps_total']}", axis=1)
            print_breakdown(sdf, "noise_dps", "Noise x DPS")

        # Top/bottom symbols
        sym_pnl = sdf.groupby("symbol")["pnl_pct"].agg(["sum", "count"]).sort_values("sum")
        print(f"\n  Top 5 symbols:")
        for sym, row in sym_pnl.tail(5).iloc[::-1].iterrows():
            print(f"    {sym:20s} {int(row['count'])} trades, PnL={row['sum']:+.2f}%")
        print(f"  Bottom 5 symbols:")
        for sym, row in sym_pnl.head(5).iterrows():
            print(f"    {sym:20s} {int(row['count'])} trades, PnL={row['sum']:+.2f}%")

    # Save to CSV
    df.to_csv("analyze_setups_results.csv", index=False)
    print(f"\nSaved {len(df)} trades to analyze_setups_results.csv")


def main():
    parser = argparse.ArgumentParser(description="Setup Analyzer")
    parser.add_argument("--dataset", type=str, default="datasets/spot_mar2_mar15",
                        help="Dataset directory (default: datasets/spot_mar2_mar15)")
    parser.add_argument("--days", type=int, default=7,
                        help="Analyze last N days of data (default: 7)")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    mr_cfg = MRSettings()
    momo_cfg = MomoGateSettings.from_json("momo_gate_settings.json")
    warmup = max(mr_cfg.range_max_bars, mr_cfg.noise_lookback_bars, 720)

    # Find all symbol files
    files = sorted(dataset.glob("*_1m.csv"))
    print(f"Dataset: {dataset} ({len(files)} symbols)")
    print(f"Analyzing last {args.days} days")
    print(f"Strategies: MR, Strict_MR, Momo")

    all_trades = []

    for fi, fpath in enumerate(files):
        symbol = fpath.stem.replace("_1m", "")
        df = pd.read_csv(str(fpath), parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        if len(df) < warmup + 200:
            continue

        # Last N days
        max_ts = df["timestamp"].max()
        cutoff = max_ts - pd.Timedelta(days=args.days)
        cutoff_idx = df[df["timestamp"] >= cutoff].index
        if len(cutoff_idx) == 0:
            continue
        scan_start = max(warmup, cutoff_idx[0])
        scan_end = len(df) - 1

        if scan_start >= scan_end:
            continue

        # Scan MR + Strict MR
        mr_trades = scan_mr(df, mr_cfg, scan_start, scan_end, symbol)
        all_trades.extend(mr_trades)

        # Scan Momo
        try:
            df_idx = df.set_index("timestamp").copy()
            df_idx.index = pd.to_datetime(df_idx.index, utc=True)
            df_prepped = prepare_features(df_idx)
            # Only scan last N days
            cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
            prepped_cutoff = df_prepped.index.searchsorted(cutoff_ts)
            if prepped_cutoff < len(df_prepped):
                momo_slice = df_prepped.iloc[prepped_cutoff:]
                if len(momo_slice) >= 200:
                    momo_trades = scan_momo(momo_slice, momo_cfg, symbol)
                    all_trades.extend(momo_trades)
        except Exception as e:
            pass

        if (fi + 1) % 20 == 0 or fi == len(files) - 1:
            mr_n = sum(1 for t in all_trades if t["strategy"] == "MR")
            smr_n = sum(1 for t in all_trades if t["strategy"] == "Strict_MR")
            momo_n = sum(1 for t in all_trades if t["strategy"] == "Momo")
            print(f"  [{fi+1}/{len(files)}] {symbol:20s}  MR={mr_n} Strict={smr_n} Momo={momo_n}")

    print_report(all_trades)


if __name__ == "__main__":
    main()
