#!/usr/bin/env python3
"""
Backtest: Depth-of-Book and Depth-Bounce strategies.

Replays depth snapshots from datasets/live/depth/{date}/*.jsonl,
runs check_depth_setup() and check_depth_bounce_setup() at each snapshot,
simulates trades forward on 1m candle data, and reports results.

Usage:
  python backtest_depth.py
"""
from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from strategy_depth import (
    DEPTH_EXCLUDED_SYMBOLS,
    DepthStrategySettings,
    check_depth_setup,
)
from strategy_depth_bounce import (
    DepthBounceSettings,
    check_depth_bounce_setup,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEPTH_DIR = Path("datasets/live/depth")
CANDLES_DIR = Path("datasets/live/candles_1m")
MAX_BARS = 120          # timeout after 120 bars (~2h)
COOLDOWN_SNAPS = 30     # ~1 hour cooldown between trades per symbol

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_depth_snapshots(symbol: str) -> list[dict]:
    """Load all depth snapshots for a symbol across all date folders."""
    snapshots = []
    if not DEPTH_DIR.exists():
        return snapshots
    for date_dir in sorted(DEPTH_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        fpath = date_dir / f"{symbol}_depth.jsonl"
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snap = json.loads(line)
                    snapshots.append(snap)
                except json.JSONDecodeError:
                    continue
    # Sort by timestamp
    snapshots.sort(key=lambda s: s.get("timestamp", ""))
    return snapshots


def load_candles(symbol: str) -> pd.DataFrame:
    """Load 1m candle CSV for a symbol."""
    fpath = CANDLES_DIR / f"{symbol}_1m.csv"
    if not fpath.exists():
        return pd.DataFrame()
    df = pd.read_csv(str(fpath), parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Ensure timezone-aware
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df


def find_candle_idx(df: pd.DataFrame, ts: pd.Timestamp) -> int:
    """Find the candle bar index closest to (but not after) a given timestamp."""
    if len(df) == 0:
        return -1
    # Binary search for the closest bar <= ts
    idx = df["timestamp"].searchsorted(ts, side="right") - 1
    if idx < 0:
        idx = 0
    return int(idx)


def simulate_trade(df: pd.DataFrame, entry_idx: int,
                   side: str, entry_price: float,
                   sl: float, tp: float,
                   max_bars: int = MAX_BARS) -> dict:
    """Simulate a trade starting from entry_idx, checking TP/SL bar by bar."""
    bars = df.iloc[entry_idx:entry_idx + max_bars]
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
    return {"outcome": "TIMEOUT", "bars_held": len(bars), "exit_price": last_close,
            "exit_ts": str(bars.iloc[-1]["timestamp"])}


def calc_pnl_pct(outcome: str, side: str, entry: float,
                 exit_price: float, sl_pct: float, tp_pct: float) -> float:
    if outcome == "TP":
        return tp_pct
    elif outcome == "SL":
        return -sl_pct
    else:
        if side == "long":
            return (exit_price - entry) / entry * 100
        else:
            return (entry - exit_price) / entry * 100


def get_depth_symbols() -> list[str]:
    """Get all unique symbols with depth data across all date folders."""
    symbols = set()
    if not DEPTH_DIR.exists():
        return []
    for date_dir in DEPTH_DIR.iterdir():
        if not date_dir.is_dir():
            continue
        for fpath in date_dir.glob("*_depth.jsonl"):
            sym = fpath.name.replace("_depth.jsonl", "")
            if sym not in DEPTH_EXCLUDED_SYMBOLS:
                symbols.add(sym)
    return sorted(symbols)


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest():
    depth_cfg = DepthStrategySettings()
    bounce_cfg = DepthBounceSettings()

    symbols = get_depth_symbols()
    print(f"Found {len(symbols)} symbols with depth data (excluding majors/commodities)")
    print(f"Excluded: {', '.join(sorted(DEPTH_EXCLUDED_SYMBOLS))}")

    depth_trades = []
    bounce_trades = []
    skipped_no_candles = []

    for si, symbol in enumerate(symbols):
        # Load candle data
        df = load_candles(symbol)
        if len(df) < 10:
            skipped_no_candles.append(symbol)
            continue

        # Load depth snapshots
        snapshots = load_depth_snapshots(symbol)
        if not snapshots:
            continue

        # Track cooldowns (by snapshot index)
        depth_cooldown = -COOLDOWN_SNAPS
        bounce_cooldown = -COOLDOWN_SNAPS

        for snap_idx, snap in enumerate(snapshots):
            asks = snap.get("asks", [])
            bids = snap.get("bids", [])
            if not asks or not bids:
                continue

            # Current price = midpoint of best bid/ask
            current_price = (float(asks[0][0]) + float(bids[0][0])) / 2

            # Parse snapshot timestamp
            snap_ts_str = snap.get("timestamp", "")
            try:
                snap_ts = pd.Timestamp(snap_ts_str).tz_convert("UTC")
            except Exception:
                try:
                    snap_ts = pd.Timestamp(snap_ts_str, tz="UTC")
                except Exception:
                    continue

            # Find matching candle index
            candle_idx = find_candle_idx(df, snap_ts)
            if candle_idx < 0:
                continue

            # Need at least some bars ahead for trade simulation
            if candle_idx >= len(df) - 5:
                continue

            # ============ DEPTH STRATEGY ============
            if snap_idx - depth_cooldown >= COOLDOWN_SNAPS:
                depth_data = {
                    "asks": asks,
                    "bids": bids,
                    "ask_walls": snap.get("ask_walls", []),
                    "bid_walls": snap.get("bid_walls", []),
                }

                result = check_depth_setup(depth_data, current_price, depth_cfg)

                if result.get("passed"):
                    # Simulate trade forward from next bar
                    trade_result = simulate_trade(
                        df, candle_idx + 1,
                        result["side"], result["entry"],
                        result["sl"], result["tp"], MAX_BARS)

                    if trade_result["outcome"] == "NO_DATA":
                        continue

                    pnl = calc_pnl_pct(
                        trade_result["outcome"], result["side"],
                        result["entry"], trade_result["exit_price"],
                        result["sl_pct"], result["tp_pct"])

                    trade = {
                        "strategy": "depth",
                        "symbol": symbol,
                        "side": result["side"],
                        "entry": result["entry"],
                        "sl": result["sl"],
                        "tp": result["tp"],
                        "sl_pct": result["sl_pct"],
                        "tp_pct": result["tp_pct"],
                        "rr": result["rr"],
                        "outcome": trade_result["outcome"],
                        "bars_held": trade_result["bars_held"],
                        "exit_price": trade_result["exit_price"],
                        "pnl_pct": round(pnl, 4),
                        "timestamp": snap_ts_str,
                        "exit_ts": trade_result.get("exit_ts", ""),
                        "date": snap_ts.strftime("%Y-%m-%d"),
                        "sl_wall_usd": result.get("sl_wall_usd", 0),
                        "sl_wall_strength": result.get("sl_wall_strength", 0),
                        "tp_wall_usd": result.get("tp_wall_usd", 0),
                        "tp_wall_strength": result.get("tp_wall_strength", 0),
                        "imbalance_1pct": result.get("imbalance_1pct", 0),
                        "imbalance_2pct": result.get("imbalance_2pct", 0),
                    }
                    depth_trades.append(trade)
                    depth_cooldown = snap_idx

            # ============ DEPTH BOUNCE STRATEGY ============
            if snap_idx - bounce_cooldown >= COOLDOWN_SNAPS:
                depth_data = {
                    "asks": asks,
                    "bids": bids,
                    "ask_walls": snap.get("ask_walls", []),
                    "bid_walls": snap.get("bid_walls", []),
                }

                # Bounce needs a candle DataFrame — pass the candles up to this point
                # Need at least 2 bars for the bounce check
                if candle_idx >= 1:
                    df_slice = df.iloc[:candle_idx + 1].copy()
                    bounce_result = check_depth_bounce_setup(
                        depth_data, df_slice, bounce_cfg)

                    if bounce_result.get("passed"):
                        trade_result = simulate_trade(
                            df, candle_idx + 1,
                            bounce_result["side"], bounce_result["entry"],
                            bounce_result["sl"], bounce_result["tp"], MAX_BARS)

                        if trade_result["outcome"] == "NO_DATA":
                            continue

                        pnl = calc_pnl_pct(
                            trade_result["outcome"], bounce_result["side"],
                            bounce_result["entry"], trade_result["exit_price"],
                            bounce_result["sl_pct"], bounce_result["tp_pct"])

                        trade = {
                            "strategy": "depth_bounce",
                            "symbol": symbol,
                            "side": bounce_result["side"],
                            "entry": bounce_result["entry"],
                            "sl": bounce_result["sl"],
                            "tp": bounce_result["tp"],
                            "sl_pct": bounce_result["sl_pct"],
                            "tp_pct": bounce_result["tp_pct"],
                            "rr": bounce_result["rr"],
                            "outcome": trade_result["outcome"],
                            "bars_held": trade_result["bars_held"],
                            "exit_price": trade_result["exit_price"],
                            "pnl_pct": round(pnl, 4),
                            "timestamp": snap_ts_str,
                            "exit_ts": trade_result.get("exit_ts", ""),
                            "date": snap_ts.strftime("%Y-%m-%d"),
                            "entry_wall_usd": bounce_result.get("entry_wall_usd", 0),
                            "entry_wall_strength": bounce_result.get("entry_wall_strength", 0),
                            "tp_wall_usd": bounce_result.get("tp_wall_usd", 0),
                            "tp_wall_strength": bounce_result.get("tp_wall_strength", 0),
                            "bounce_type": bounce_result.get("bounce_type", ""),
                            "wall_price": bounce_result.get("wall_price", 0),
                        }
                        bounce_trades.append(trade)
                        bounce_cooldown = snap_idx

        if (si + 1) % 20 == 0 or si == len(symbols) - 1:
            print(f"  [{si+1}/{len(symbols)}] {symbol:20s}  "
                  f"depth={len(depth_trades)} bounce={len(bounce_trades)}")

    # ===================== BUILD DATAFRAMES =====================
    depth_df = pd.DataFrame(depth_trades)
    bounce_df = pd.DataFrame(bounce_trades)
    all_df = pd.concat([depth_df, bounce_df], ignore_index=True) if len(depth_df) + len(bounce_df) > 0 else pd.DataFrame()

    # ===================== REPORT =====================
    print(f"\n{'='*70}")
    print(f"DEPTH STRATEGY BACKTEST RESULTS")
    print(f"Data: {DEPTH_DIR} (dates: {', '.join(sorted(d.name for d in DEPTH_DIR.iterdir() if d.is_dir()))})")
    print(f"Symbols: {len(symbols)} tested, {len(skipped_no_candles)} skipped (no candle data)")
    if skipped_no_candles:
        print(f"  Skipped: {', '.join(skipped_no_candles[:20])}"
              + (f"... +{len(skipped_no_candles)-20} more" if len(skipped_no_candles) > 20 else ""))
    print(f"{'='*70}")

    for label, tdf in [("DEPTH (wall imbalance)", depth_df),
                        ("DEPTH BOUNCE (wall touch)", bounce_df)]:
        _print_strategy_report(label, tdf)

    # ===================== COMPARISON =====================
    if len(depth_df) > 0 and len(bounce_df) > 0:
        print(f"\n{'='*70}")
        print(f"COMPARISON: DEPTH vs DEPTH BOUNCE")
        print(f"{'='*70}")
        d_pnl = depth_df["pnl_pct"].sum()
        b_pnl = bounce_df["pnl_pct"].sum()
        d_wr = _win_rate(depth_df)
        b_wr = _win_rate(bounce_df)
        print(f"  Depth:        {len(depth_df):3d} trades, WR={d_wr:.1f}%, PnL={d_pnl:+.2f}%")
        print(f"  Depth Bounce: {len(bounce_df):3d} trades, WR={b_wr:.1f}%, PnL={b_pnl:+.2f}%")
        better = "Depth" if d_pnl > b_pnl else "Depth Bounce"
        print(f"  Winner: {better}")

    # ===================== SAVE CSV =====================
    if len(all_df) > 0:
        all_df.to_csv("backtest_depth_trades.csv", index=False)
        print(f"\nSaved {len(all_df)} trades to backtest_depth_trades.csv")
        print(f"  ({len(depth_df)} depth + {len(bounce_df)} bounce)")
    else:
        print("\nNo trades generated.")


def _win_rate(tdf: pd.DataFrame) -> float:
    if len(tdf) == 0:
        return 0.0
    tp = (tdf["outcome"] == "TP").sum()
    sl = (tdf["outcome"] == "SL").sum()
    return tp / (tp + sl) * 100 if (tp + sl) > 0 else 0.0


def _print_strategy_report(label: str, tdf: pd.DataFrame):
    print(f"\n--- {label} ---")
    if len(tdf) == 0:
        print("  No trades.")
        return

    tp_n = (tdf["outcome"] == "TP").sum()
    sl_n = (tdf["outcome"] == "SL").sum()
    to_n = (tdf["outcome"] == "TIMEOUT").sum()
    closed = tp_n + sl_n
    wr = tp_n / closed * 100 if closed > 0 else 0
    total_pnl = tdf["pnl_pct"].sum()
    avg_pnl = tdf["pnl_pct"].mean()
    avg_bars = tdf["bars_held"].mean()

    print(f"  Total trades:      {len(tdf)}")
    print(f"  TP / SL / Timeout: {tp_n} / {sl_n} / {to_n}")
    print(f"  Win Rate:          {wr:.1f}%")
    print(f"  Total PnL:         {total_pnl:+.2f}%")
    print(f"  Avg PnL/trade:     {avg_pnl:+.4f}%")
    print(f"  Avg bars held:     {avg_bars:.0f}")

    # By side
    print(f"  By side:")
    for side in ["long", "short"]:
        sub = tdf[tdf["side"] == side]
        if len(sub) == 0:
            continue
        tp = (sub["outcome"] == "TP").sum()
        sl = (sub["outcome"] == "SL").sum()
        to = (sub["outcome"] == "TIMEOUT").sum()
        wr_s = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        print(f"    {side:5s}: {len(sub):3d} trades, {tp} TP / {sl} SL / {to} TO, "
              f"WR={wr_s:.1f}%, PnL={sub['pnl_pct'].sum():+.2f}%")

    # By day
    if "date" in tdf.columns:
        print(f"  By day:")
        for date in sorted(tdf["date"].unique()):
            sub = tdf[tdf["date"] == date]
            tp = (sub["outcome"] == "TP").sum()
            sl = (sub["outcome"] == "SL").sum()
            to = (sub["outcome"] == "TIMEOUT").sum()
            wr_d = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
            print(f"    {date}: {len(sub):3d} trades, {tp} TP / {sl} SL / {to} TO, "
                  f"WR={wr_d:.1f}%, PnL={sub['pnl_pct'].sum():+.2f}%")

    # By hour of day
    if "timestamp" in tdf.columns:
        print(f"  By hour of day:")
        tdf_hour = tdf.copy()
        tdf_hour["hour"] = pd.to_datetime(tdf_hour["timestamp"]).dt.hour
        hour_stats = []
        for hour in sorted(tdf_hour["hour"].unique()):
            sub = tdf_hour[tdf_hour["hour"] == hour]
            tp = (sub["outcome"] == "TP").sum()
            sl = (sub["outcome"] == "SL").sum()
            wr_h = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
            pnl_h = sub["pnl_pct"].sum()
            hour_stats.append((hour, len(sub), wr_h, pnl_h))
            print(f"    {hour:02d}:00  {len(sub):3d} trades, "
                  f"WR={wr_h:.1f}%, PnL={pnl_h:+.2f}%")
        if hour_stats:
            best = max(hour_stats, key=lambda x: x[3])
            worst = min(hour_stats, key=lambda x: x[3])
            print(f"    Best hour:  {best[0]:02d}:00 (PnL={best[3]:+.2f}%)")
            print(f"    Worst hour: {worst[0]:02d}:00 (PnL={worst[3]:+.2f}%)")

    # Top/bottom symbols
    if len(tdf) >= 5:
        sym_pnl = tdf.groupby("symbol")["pnl_pct"].agg(["sum", "count"]).sort_values("sum")
        print(f"  Top 5 symbols:")
        for sym, row in sym_pnl.tail(5).iloc[::-1].iterrows():
            print(f"    {sym:20s}  {row['count']:.0f} trades  PnL={row['sum']:+.2f}%")
        print(f"  Bottom 5 symbols:")
        for sym, row in sym_pnl.head(5).iterrows():
            print(f"    {sym:20s}  {row['count']:.0f} trades  PnL={row['sum']:+.2f}%")


if __name__ == "__main__":
    run_backtest()
