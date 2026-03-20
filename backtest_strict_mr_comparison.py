#!/usr/bin/env python3
"""
Backtest: Strict MR vs Regular MR comparison.

Runs both check_mr_gates_at_bar() and check_strict_mr_gates_at_bar()
on a fixed symbol list over the last 7 days of the dataset (Mar 8-15),
simulates trades with identical SL/TP/timeout logic, and compares results.
"""
from __future__ import annotations

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scan_mean_reversion import (
    MRSettings,
    check_mr_gates_at_bar,
    check_strict_mr_gates_at_bar,
)

DATASET_DIR = Path("datasets/spot_mar2_mar15")
MAX_BARS = 120  # timeout after 120 bars

SYMBOLS = [
    "1000SATS", "1MBABYDOGE", "A2Z", "AAVE", "ACE", "AGLD", "AGT", "AI",
    "ATA", "ATOM", "AVNT", "AWE", "AXS", "B2", "BAND", "BEAT", "BIO",
    "BLESS", "BNB", "BOB", "BROCCOLIF3B", "BR", "BTR", "B", "CHZ", "COAI",
    "COOKIE", "DIA", "DOGE", "DOLO", "DOT", "DRIFT", "ETC", "ETH", "EVAA",
    "FET", "FHE", "GAS", "GLMR", "HAEDAL", "HANA", "HEI", "HMSTR", "HYPE",
    "ICNT", "IDOL", "IOTX", "IR", "IRYS", "JCT", "JELLYJELLY", "KAIA",
    "KAS", "KERNEL", "LAYER", "LIGHT", "LINK", "LISTA", "LIT", "LRC",
    "LSK", "LTC", "LUNA2", "MANTA", "METIS", "MOVR", "NEAR", "NEIRO",
    "NEO", "NMR", "OG", "OM", "OXT", "PENGU", "PHB", "PIEVERSE", "PIPPIN",
    "PLAY", "POWER", "PROMPT", "PTB", "PUMP", "Q", "RDNT", "RECALL",
    "RIVER", "RLC", "SAHARA", "SOL", "SOON", "SPK", "STBL", "STO", "STRK",
    "SXT", "TAKE", "TA", "TNSR", "TRADOOR", "TREE", "TRUST", "TST",
    "WLFI", "XTZ", "ZBT", "ZEC", "ZORA",
]


def simulate_trade(df_1m: pd.DataFrame, entry_idx: int,
                   side: str, entry_price: float,
                   sl: float, tp: float,
                   max_bars: int = MAX_BARS) -> dict:
    """Simulate a trade starting from entry_idx (bar-indexed)."""
    bars = df_1m.iloc[entry_idx:entry_idx + max_bars]
    if len(bars) == 0:
        return {"outcome": "NO_DATA", "bars_held": 0, "exit_price": entry_price}

    for offset, (idx, bar) in enumerate(bars.iterrows()):
        if side == "long":
            if bar["low"] <= sl:
                return {"outcome": "SL", "bars_held": offset + 1,
                        "exit_price": sl, "exit_ts": str(bar["timestamp"])}
            if bar["high"] >= tp:
                return {"outcome": "TP", "bars_held": offset + 1,
                        "exit_price": tp, "exit_ts": str(bar["timestamp"])}
        else:
            if bar["high"] >= sl:
                return {"outcome": "SL", "bars_held": offset + 1,
                        "exit_price": sl, "exit_ts": str(bar["timestamp"])}
            if bar["low"] <= tp:
                return {"outcome": "TP", "bars_held": offset + 1,
                        "exit_price": tp, "exit_ts": str(bar["timestamp"])}

    last_close = float(bars.iloc[-1]["close"])
    return {"outcome": "OPEN", "bars_held": len(bars), "exit_price": last_close,
            "exit_ts": str(bars.iloc[-1]["timestamp"])}


def calc_pnl_pct(outcome, side, entry, exit_price, sl_pct, tp_pct):
    if outcome == "TP":
        return tp_pct
    elif outcome == "SL":
        return -sl_pct
    else:
        if side == "long":
            return (exit_price - entry) / entry * 100
        else:
            return (entry - exit_price) / entry * 100


def main():
    cfg = MRSettings()
    warmup = max(cfg.range_max_bars, cfg.noise_lookback_bars, 720)

    # Deduplicate symbols
    seen = set()
    unique_symbols = []
    for s in SYMBOLS:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)

    # Check which files exist
    valid_symbols = []
    for sym in unique_symbols:
        fpath = DATASET_DIR / f"{sym}USDT_1m.csv"
        if fpath.exists():
            valid_symbols.append(sym)
    missing = set(unique_symbols) - set(valid_symbols)
    print(f"Symbols: {len(unique_symbols)} requested, {len(valid_symbols)} found in dataset")
    if missing:
        print(f"Missing ({len(missing)}): {', '.join(sorted(missing))}")

    regular_trades = []
    strict_trades = []
    filtered_by_strict = []  # trades in regular but NOT in strict

    for si, sym in enumerate(valid_symbols):
        fpath = DATASET_DIR / f"{sym}USDT_1m.csv"
        df = pd.read_csv(str(fpath), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        if len(df) < warmup + 100:
            continue

        # Last 7 days: find the max timestamp and subtract 7 days
        max_ts = df["timestamp"].max()
        cutoff = max_ts - pd.Timedelta(days=7)

        # We need warmup bars BEFORE the cutoff, so we load the full df
        # but only generate signals for bars where timestamp >= cutoff
        cutoff_idx = df[df["timestamp"] >= cutoff].index[0] if len(df[df["timestamp"] >= cutoff]) > 0 else len(df)
        # Make sure we start after warmup
        scan_start = max(warmup, cutoff_idx)
        scan_end = len(df) - 1  # leave 1 bar for trade simulation

        if scan_start >= scan_end:
            continue

        symbol_name = f"{sym}USDT"
        cooldown_regular = -1
        cooldown_strict = -1

        for i in range(scan_start, scan_end):
            # --- Regular MR ---
            if i > cooldown_regular:
                result_reg = check_mr_gates_at_bar(df, i, cfg)
                if result_reg["passed"]:
                    trade = simulate_trade(df, i, result_reg["side"],
                                           result_reg["entry"],
                                           result_reg["sl"], result_reg["tp"])
                    trade["symbol"] = symbol_name
                    trade["side"] = result_reg["side"]
                    trade["entry"] = result_reg["entry"]
                    trade["sl"] = result_reg["sl"]
                    trade["tp"] = result_reg["tp"]
                    trade["sl_pct"] = result_reg["sl_pct"]
                    trade["tp_pct"] = result_reg["tp_pct"]
                    trade["rr"] = result_reg.get("rr", 1.0)
                    trade["dps_total"] = result_reg["dps_total"]
                    trade["dps_confidence"] = result_reg["dps_confidence"]
                    trade["touches"] = result_reg["touches"]
                    trade["timestamp"] = str(df.iloc[i]["timestamp"])
                    trade["pnl_pct"] = calc_pnl_pct(
                        trade["outcome"], trade["side"], trade["entry"],
                        trade["exit_price"], trade["sl_pct"], trade["tp_pct"])
                    regular_trades.append(trade)
                    cooldown_regular = i + cfg.cooldown_bars

            # --- Strict MR ---
            if i > cooldown_strict:
                result_str = check_strict_mr_gates_at_bar(df, i, cfg)
                if result_str["passed"]:
                    trade = simulate_trade(df, i, result_str["side"],
                                           result_str["entry"],
                                           result_str["sl"], result_str["tp"])
                    trade["symbol"] = symbol_name
                    trade["side"] = result_str["side"]
                    trade["entry"] = result_str["entry"]
                    trade["sl"] = result_str["sl"]
                    trade["tp"] = result_str["tp"]
                    trade["sl_pct"] = result_str["sl_pct"]
                    trade["tp_pct"] = result_str["tp_pct"]
                    trade["rr"] = result_str.get("rr", 1.0)
                    trade["dps_total"] = result_str["dps_total"]
                    trade["dps_confidence"] = result_str["dps_confidence"]
                    trade["touches"] = result_str["touches"]
                    trade["timestamp"] = str(df.iloc[i]["timestamp"])
                    trade["pnl_pct"] = calc_pnl_pct(
                        trade["outcome"], trade["side"], trade["entry"],
                        trade["exit_price"], trade["sl_pct"], trade["tp_pct"])
                    strict_trades.append(trade)
                    cooldown_strict = i + cfg.cooldown_bars

        if (si + 1) % 10 == 0 or si == len(valid_symbols) - 1:
            print(f"  [{si+1}/{len(valid_symbols)}] {symbol_name:20s}  "
                  f"regular={len(regular_trades)} strict={len(strict_trades)}")

    # Build DataFrames
    reg_df = pd.DataFrame(regular_trades)
    str_df = pd.DataFrame(strict_trades)

    # Find which regular trades were filtered by strict
    if len(reg_df) > 0 and len(str_df) > 0:
        reg_keys = set(zip(reg_df["symbol"], reg_df["timestamp"]))
        str_keys = set(zip(str_df["symbol"], str_df["timestamp"]))
        filtered_keys = reg_keys - str_keys
        filtered_df = reg_df[reg_df.apply(
            lambda r: (r["symbol"], r["timestamp"]) in filtered_keys, axis=1)]
    elif len(reg_df) > 0:
        filtered_df = reg_df.copy()
    else:
        filtered_df = pd.DataFrame()

    # ===================== REPORT =====================
    print(f"\n{'='*70}")
    print(f"STRICT MR vs REGULAR MR — COMPARISON RESULTS")
    print(f"Dataset: {DATASET_DIR} (last 7 days)")
    print(f"{'='*70}")

    for label, tdf in [("REGULAR MR", reg_df), ("STRICT MR", str_df)]:
        print(f"\n--- {label} ---")
        if len(tdf) == 0:
            print("  No trades.")
            continue

        tp_n = (tdf["outcome"] == "TP").sum()
        sl_n = (tdf["outcome"] == "SL").sum()
        op_n = (tdf["outcome"] == "OPEN").sum()
        closed = tp_n + sl_n
        wr = tp_n / closed * 100 if closed > 0 else 0
        total_pnl = tdf["pnl_pct"].sum()
        avg_pnl = tdf["pnl_pct"].mean()
        avg_bars = tdf["bars_held"].mean()

        print(f"  Total trades:   {len(tdf)}")
        print(f"  TP / SL / OPEN: {tp_n} / {sl_n} / {op_n}")
        print(f"  Win Rate:       {wr:.1f}%")
        print(f"  Total PnL:      {total_pnl:.2f}%")
        print(f"  Avg PnL/trade:  {avg_pnl:.4f}%")
        print(f"  Avg bars held:  {avg_bars:.0f}")

        # By side
        for side in ["long", "short"]:
            sub = tdf[tdf["side"] == side]
            if len(sub) == 0:
                continue
            tp = (sub["outcome"] == "TP").sum()
            sl = (sub["outcome"] == "SL").sum()
            wr_s = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
            print(f"    {side:5s}: {len(sub)} trades, {tp} TP / {sl} SL, "
                  f"WR={wr_s:.1f}%, PnL={sub['pnl_pct'].sum():.2f}%")

        # By DPS
        print(f"  By DPS:")
        for dps in sorted(tdf["dps_total"].unique()):
            sub = tdf[tdf["dps_total"] == dps]
            tp = (sub["outcome"] == "TP").sum()
            sl = (sub["outcome"] == "SL").sum()
            wr_d = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
            print(f"    DPS {dps}: {len(sub)} trades, WR={wr_d:.1f}%, "
                  f"PnL={sub['pnl_pct'].sum():.2f}%")

    # Filtered trades analysis
    print(f"\n--- FILTERED BY STRICT (in Regular but NOT in Strict) ---")
    if len(filtered_df) > 0:
        tp_n = (filtered_df["outcome"] == "TP").sum()
        sl_n = (filtered_df["outcome"] == "SL").sum()
        op_n = (filtered_df["outcome"] == "OPEN").sum()
        closed = tp_n + sl_n
        wr = tp_n / closed * 100 if closed > 0 else 0
        total_pnl = filtered_df["pnl_pct"].sum()
        print(f"  {len(filtered_df)} trades filtered out")
        print(f"  TP / SL / OPEN: {tp_n} / {sl_n} / {op_n}")
        print(f"  Win Rate:       {wr:.1f}%")
        print(f"  Total PnL:      {total_pnl:.2f}%")
        print(f"  (Positive = Strict correctly filtered losers, "
              f"Negative = Strict filtered winners)")

        # Show each filtered trade
        print(f"\n  Individual filtered trades:")
        print(f"  {'Symbol':20s} {'Side':5s} {'Outcome':7s} {'PnL%':>8s} {'DPS':>4s} {'Timestamp'}")
        print(f"  {'-'*80}")
        for _, r in filtered_df.iterrows():
            print(f"  {r['symbol']:20s} {r['side']:5s} {r['outcome']:7s} "
                  f"{r['pnl_pct']:8.3f} {r['dps_total']:4.0f} {r['timestamp']}")
    else:
        print("  No trades were filtered (both strategies produced identical signals)")

    # Strict-only trades (in strict but not regular) — should be rare/impossible
    # since strict is a subset, but check anyway
    if len(str_df) > 0 and len(reg_df) > 0:
        str_keys = set(zip(str_df["symbol"], str_df["timestamp"]))
        reg_keys = set(zip(reg_df["symbol"], reg_df["timestamp"]))
        strict_only = str_keys - reg_keys
        if strict_only:
            print(f"\n  NOTE: {len(strict_only)} trades appeared in Strict but NOT Regular")
            print(f"  (This can happen due to independent cooldowns)")

    # Summary verdict
    print(f"\n{'='*70}")
    print(f"VERDICT:")
    reg_pnl = reg_df["pnl_pct"].sum() if len(reg_df) > 0 else 0
    str_pnl = str_df["pnl_pct"].sum() if len(str_df) > 0 else 0
    reg_n = len(reg_df)
    str_n = len(str_df)
    reg_wr = (reg_df["outcome"] == "TP").sum() / max(1, ((reg_df["outcome"] == "TP").sum() + (reg_df["outcome"] == "SL").sum())) * 100 if len(reg_df) > 0 else 0
    str_wr = (str_df["outcome"] == "TP").sum() / max(1, ((str_df["outcome"] == "TP").sum() + (str_df["outcome"] == "SL").sum())) * 100 if len(str_df) > 0 else 0
    print(f"  Regular MR: {reg_n} trades, WR={reg_wr:.1f}%, PnL={reg_pnl:.2f}%")
    print(f"  Strict  MR: {str_n} trades, WR={str_wr:.1f}%, PnL={str_pnl:.2f}%")
    print(f"  Trades filtered: {reg_n - str_n}")
    if reg_pnl != 0:
        improvement = str_pnl - reg_pnl
        print(f"  PnL difference: {improvement:+.2f}% ({'Strict better' if improvement > 0 else 'Regular better'})")
    print(f"{'='*70}")

    # Save full trade lists to CSV
    if len(reg_df) > 0:
        reg_df.to_csv("backtest_regular_mr_trades.csv", index=False)
        print(f"\nSaved {len(reg_df)} regular MR trades to backtest_regular_mr_trades.csv")
    if len(str_df) > 0:
        str_df.to_csv("backtest_strict_mr_trades.csv", index=False)
        print(f"Saved {len(str_df)} strict MR trades to backtest_strict_mr_trades.csv")


if __name__ == "__main__":
    main()
