#!/usr/bin/env python3
"""
Live Dummy Trader

Scans active coins in real-time using Momentum and Mean Reversion strategies,
enters dummy trades with TP/SL, tracks positions, and logs everything including
depth-based TP/SL alternatives for comparison.

Usage:
  python live_dummy_trader.py [--interval 60] [--min-vol 200000] [--top-n 30]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

handler = logging.StreamHandler(stream=sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s",
                                        datefmt="%H:%M:%S"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Imports from existing modules
# ---------------------------------------------------------------------------
from live_data_collector import (
    fetch_orion_active_coins,
    fetch_bitunix_depth,
    fetch_bitunix_klines,
    analyze_depth,
    LIVE_DIR,
    BITUNIX_BASE,
)
from scan_mean_reversion import (
    MRSettings,
    check_mr_gates_at_bar,
)
from backtest_momo_vwap_grind15_full import (
    GateSettings as MomoGateSettings,
    check_momo_gates_at_bar,
)

# Load momentum gate config (same as backtest)
_momo_gate_cfg = MomoGateSettings.from_json("momo_gate_settings.json")
from depth_tp_sl_analyzer import compute_depth_tp_sl

import requests

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
TRADER_DIR = LIVE_DIR / "dummy_trader"
TRADES_LOG = TRADER_DIR / "trades.csv"
POSITIONS_FILE = TRADER_DIR / "open_positions.json"
DEPTH_COMPARISON_LOG = TRADER_DIR / "depth_comparison.csv"
CYCLE_LOG = TRADER_DIR / "cycle_log.csv"
CLOSED_TRADES_LOG = TRADER_DIR / "closed_trades.csv"

# How many 1m bars we need for each strategy
MR_WARMUP_BARS = 800    # MR scanner needs up to 720 + buffer
MOMO_WARMUP_BARS = 400  # Momo needs ~370 bars

# Dummy account settings
DUMMY_BALANCE = 10000.0  # $10k starting
STANDARD_RISK_PCT = 1.0  # 1% risk per trade
DUMMY_RISK_PCT = 0.1     # 0.1% for low-confidence


# ---------------------------------------------------------------------------
# Bitunix paginated kline fetch
# ---------------------------------------------------------------------------

def fetch_klines_paginated(symbol: str, n_bars: int = 800,
                            interval: str = "1m") -> Optional[pd.DataFrame]:
    """
    Fetch up to n_bars of 1m candles from Bitunix by paginating
    (API returns max 100 per request).
    """
    all_rows = []
    now_ms = int(time.time() * 1000)
    end_ms = now_ms

    bars_remaining = n_bars
    max_requests = 12  # Safety limit

    for _ in range(max_requests):
        if bars_remaining <= 0:
            break

        batch_size = min(bars_remaining, 100)
        start_ms = end_ms - batch_size * 60 * 1000

        try:
            resp = requests.get(
                f"{BITUNIX_BASE}/api/v1/futures/market/kline",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": str(start_ms),
                    "endTime": str(end_ms),
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                break
            klines = data.get("data", [])
        except Exception:
            break

        if not klines:
            break

        for k in klines:
            ts = k.get("time")
            if ts is None:
                continue
            all_rows.append({
                "timestamp": pd.Timestamp(int(ts), unit="ms", tz="UTC"),
                "open": float(k.get("open", 0)),
                "high": float(k.get("high", 0)),
                "low": float(k.get("low", 0)),
                "close": float(k.get("close", 0)),
                "volume": float(k.get("quoteVol", 0)),
            })

        bars_remaining -= len(klines)
        # Move window back
        end_ms = start_ms - 1

        time.sleep(0.12)  # Rate limit

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------

class PositionManager:
    """Track open dummy positions and check for TP/SL hits."""

    def __init__(self, positions_file: Path = POSITIONS_FILE):
        self.positions_file = positions_file
        self.positions: list[dict] = []
        self._load()

    def _load(self):
        if self.positions_file.exists():
            with open(self.positions_file) as f:
                self.positions = json.load(f)

    def _save(self):
        self.positions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.positions_file, "w") as f:
            json.dump(self.positions, f, indent=2, default=str)

    def has_position(self, symbol: str, strategy: str) -> bool:
        """Check if we already have an open position for this symbol+strategy."""
        return any(
            p["symbol"] == symbol and p["strategy"] == strategy
            for p in self.positions
        )

    def open_position(self, trade: dict):
        """Add a new open position."""
        self.positions.append(trade)
        self._save()

    def check_tp_sl(self, symbol: str, current_high: float, current_low: float,
                     current_close: float, current_ts: str) -> list[dict]:
        """
        Check if any positions hit TP or SL.
        Returns list of closed trades.
        """
        closed = []
        remaining = []

        for pos in self.positions:
            if pos["symbol"] != symbol:
                remaining.append(pos)
                continue

            hit = None
            exit_price = None

            if pos["side"] == "long":
                if current_low <= pos["sl"]:
                    hit = "SL"
                    exit_price = pos["sl"]
                elif current_high >= pos["tp"]:
                    hit = "TP"
                    exit_price = pos["tp"]
            else:  # short
                if current_high >= pos["sl"]:
                    hit = "SL"
                    exit_price = pos["sl"]
                elif current_low <= pos["tp"]:
                    hit = "TP"
                    exit_price = pos["tp"]

            if hit:
                pos["outcome"] = hit
                pos["exit_price"] = exit_price
                pos["exit_ts"] = current_ts
                pos["bars_held"] = pos.get("bars_held", 0) + 1

                if pos["side"] == "long":
                    pos["pnl_pct"] = (exit_price - pos["entry"]) / pos["entry"] * 100
                else:
                    pos["pnl_pct"] = (pos["entry"] - exit_price) / pos["entry"] * 100

                pos["pnl_usd"] = pos["pnl_pct"] / 100 * pos.get("position_usd", 0)
                closed.append(pos)
            else:
                pos["bars_held"] = pos.get("bars_held", 0) + 1

                # Check max hold time (2 hours = 120 bars)
                if pos.get("bars_held", 0) >= 120:
                    pos["outcome"] = "TIMEOUT"
                    pos["exit_price"] = current_close
                    pos["exit_ts"] = current_ts
                    if pos["side"] == "long":
                        pos["pnl_pct"] = (current_close - pos["entry"]) / pos["entry"] * 100
                    else:
                        pos["pnl_pct"] = (pos["entry"] - current_close) / pos["entry"] * 100
                    pos["pnl_usd"] = pos["pnl_pct"] / 100 * pos.get("position_usd", 0)
                    closed.append(pos)
                else:
                    remaining.append(pos)

        self.positions = remaining
        self._save()
        return closed

    def get_all_symbols(self) -> set:
        """Get all symbols with open positions."""
        return {p["symbol"] for p in self.positions}


# ---------------------------------------------------------------------------
# MR Setup Detection (live, latest bar)
# ---------------------------------------------------------------------------

def detect_mr_setup_live(df: pd.DataFrame, symbol: str,
                          cfg: MRSettings) -> Optional[dict]:
    """
    Check if the LATEST bar has an MR setup.
    Uses the shared check_mr_gates_at_bar() for identical logic
    between backtest and live trading.
    """
    if len(df) < max(cfg.range_max_bars, cfg.noise_lookback_bars, 720) + 10:
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)
    i = len(df) - 1  # Latest bar

    result = check_mr_gates_at_bar(df, i, cfg)
    if not result["passed"]:
        return None

    # Only take DPS >= 3 (low confidence and above)
    if result["dps_total"] < 3:
        return None

    entry_ts = str(df.iloc[i]["timestamp"])

    return {
        "symbol": symbol,
        "strategy": "mean_reversion",
        "timestamp": entry_ts,
        "side": result["side"],
        "entry": result["entry"],
        "sl": result["sl"],
        "tp": result["tp"],
        "sl_pct": result["sl_pct"],
        "tp_pct": result["tp_pct"],
        "rr": result["rr"],
        "dps_total": result["dps_total"],
        "dps_confidence": result["dps_confidence"],
        "noise_level": result["noise_level"],
        "touches": result["touches"],
        "break_pct": result["break_pct"],
        "range_width_pct": result["range_width_pct"],
        "range_duration_hrs": result["range_duration_hrs"],
        "pre_chop_trend": result["pre_chop_trend"],
        "approach": result["dps_v2_label"],
        "vol_trend": result["dps_v3_vol_trend"],
        "range_upper": result["range_upper"],
        "range_lower": result["range_lower"],
        "touch_timestamps": result["touch_timestamps"],
    }


# ---------------------------------------------------------------------------
# Momentum Setup Detection (live, latest bar)
# ---------------------------------------------------------------------------

def detect_momo_setup_live(df: pd.DataFrame, symbol: str) -> Optional[dict]:
    """
    Check if the LATEST bars show a momentum setup.

    Uses the same gate-based system as the backtest (check_momo_gates_at_bar)
    to ensure identical trade detection between backtest and live.
    DPS scoring is computed inside the shared gate function.
    """
    if len(df) < 500:
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Need timestamp-indexed DataFrame for the gate functions
    df_indexed = df.set_index("timestamp").copy()
    df_indexed.index = pd.to_datetime(df_indexed.index, utc=True)

    # Check both directions using the shared gate system
    for direction in ["long", "short"]:
        result = check_momo_gates_at_bar(df_indexed, direction, _momo_gate_cfg)

        if not result["passed"]:
            continue

        # DPS filter: only take DPS >= 3
        if result["dps_total"] < 3:
            continue

        entry_ts = str(df.iloc[-1]["timestamp"])

        return {
            "symbol": symbol,
            "strategy": "momentum",
            "timestamp": entry_ts,
            "side": direction,
            "entry": round(result["entry"], 8),
            "sl": round(result["sl"], 8),
            "tp": round(result["tp"], 8),
            "sl_pct": result["sl_pct"],
            "tp_pct": result["tp_pct"],
            "rr": result["rr"],
            "dps_total": result["dps_total"],
            "dps_confidence": result["dps_confidence"],
            "approach": result["approach"],
            "vol_trend": result["vol_trend"],
            "duration_hrs": result["duration_hrs"],
        }

    return None


# ---------------------------------------------------------------------------
# Depth Comparison
# ---------------------------------------------------------------------------

def get_depth_alternative(symbol: str, depth_data: dict, current_price: float,
                           side: str, strategy: str) -> Optional[dict]:
    """
    Compute what depth-based TP/SL would suggest for comparison.
    """
    strat_key = "momentum" if strategy == "momentum" else "mean_reversion"
    result = compute_depth_tp_sl(
        depth_data, current_price,
        side=side, strategy=strat_key,
        min_tp_pct=1.0, min_rr=1.0,
    )

    best = result.get("best_combo")
    if not best:
        return None

    return {
        "depth_tp": best["tp"]["price"],
        "depth_tp_pct": best["tp"]["dist_pct"],
        "depth_tp_reason": best["tp"]["reason"],
        "depth_sl": best["sl"]["price"],
        "depth_sl_pct": best["sl"]["dist_pct"],
        "depth_sl_reason": best["sl"]["reason"],
        "depth_rr": best["rr"],
        "depth_score": best["score"],
        "depth_sl_wall_usd": best["sl"].get("wall_usd", 0),
        "depth_imbalance_1pct": result.get("depth_imbalance_1pct", 0),
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_trade_entry(trade: dict, depth_alt: Optional[dict] = None):
    """Log a new trade to CSV."""
    TRADER_DIR.mkdir(parents=True, exist_ok=True)

    # Main trade log
    write_header = not TRADES_LOG.exists()
    with open(TRADES_LOG, "a", encoding="utf-8") as f:
        if write_header:
            cols = list(trade.keys())
            f.write(",".join(cols) + "\n")
        vals = []
        for v in trade.values():
            s = str(v) if v is not None else ""
            if "," in s:
                s = f'"{s}"'
            vals.append(s)
        f.write(",".join(vals) + "\n")

    # Depth comparison log
    if depth_alt:
        comparison = {
            "timestamp": trade["timestamp"],
            "symbol": trade["symbol"],
            "strategy": trade["strategy"],
            "side": trade["side"],
            "entry": trade["entry"],
            # Strategy TP/SL
            "strat_tp": trade["tp"],
            "strat_tp_pct": trade["tp_pct"],
            "strat_sl": trade["sl"],
            "strat_sl_pct": trade["sl_pct"],
            "strat_rr": trade["rr"],
            # Depth TP/SL
            **depth_alt,
        }
        write_header = not DEPTH_COMPARISON_LOG.exists()
        with open(DEPTH_COMPARISON_LOG, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(comparison.keys()) + "\n")
            vals = []
            for v in comparison.values():
                s = str(v) if v is not None else ""
                if "," in s:
                    s = f'"{s}"'
                vals.append(s)
            f.write(",".join(vals) + "\n")


def log_closed_trade(trade: dict):
    """Log a closed trade."""
    TRADER_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CLOSED_TRADES_LOG.exists()
    with open(CLOSED_TRADES_LOG, "a", encoding="utf-8") as f:
        if write_header:
            cols = list(trade.keys())
            f.write(",".join(cols) + "\n")
        vals = []
        for v in trade.values():
            s = str(v) if v is not None else ""
            if "," in s:
                s = f'"{s}"'
            vals.append(s)
        f.write(",".join(vals) + "\n")


def log_cycle(cycle_num: int, n_coins: int, n_mr_setups: int, n_momo_setups: int,
              n_open: int, n_closed: int, total_pnl: float):
    """Log cycle summary."""
    TRADER_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    write_header = not CYCLE_LOG.exists()
    with open(CYCLE_LOG, "a", encoding="utf-8") as f:
        if write_header:
            f.write("timestamp,cycle,coins_scanned,mr_setups,momo_setups,"
                    "open_positions,closed_this_cycle,total_pnl\n")
        f.write(f"{ts},{cycle_num},{n_coins},{n_mr_setups},{n_momo_setups},"
                f"{n_open},{n_closed},{total_pnl:.4f}\n")


# ---------------------------------------------------------------------------
# Main Trading Cycle
# ---------------------------------------------------------------------------

def trading_cycle(pos_mgr: PositionManager, mr_cfg: MRSettings,
                  min_vol: float, top_n: int, cycle_num: int,
                  candle_cache: dict) -> dict:
    """
    Run one trading cycle:
    1. Get active coins from Orion
    2. Fetch candles for each
    3. Check for MR and Momo setups
    4. Enter dummy trades
    5. Check open positions for TP/SL hits
    6. Log depth comparison
    """
    stats = {"coins": 0, "mr_setups": 0, "momo_setups": 0, "closed": 0, "pnl": 0.0}

    # Step 1: Get active coins
    coins = fetch_orion_active_coins(min_vol_5m=min_vol, top_n=top_n)
    if not coins:
        logger.warning("No coins from Orion")
        return stats

    # Also include symbols with open positions
    open_syms = pos_mgr.get_all_symbols()
    coin_symbols = {c["symbol"] for c in coins}
    all_symbols = coin_symbols | open_syms

    stats["coins"] = len(all_symbols)
    logger.info("Cycle %d: %d coins (%d active + %d with positions)",
                cycle_num, len(all_symbols), len(coin_symbols), len(open_syms - coin_symbols))

    for sym in sorted(all_symbols):
        # Step 2: Fetch candles (paginated for enough history)
        # Use cache to avoid re-fetching entire history each cycle
        if sym in candle_cache:
            cached_df = candle_cache[sym]
            # Just fetch latest 5 bars to append
            new_klines = fetch_bitunix_klines(sym, interval="1m", limit=5)
            if new_klines:
                new_rows = []
                for k in new_klines:
                    ts = k.get("time")
                    if ts is None:
                        continue
                    new_rows.append({
                        "timestamp": pd.Timestamp(int(ts), unit="ms", tz="UTC"),
                        "open": float(k.get("open", 0)),
                        "high": float(k.get("high", 0)),
                        "low": float(k.get("low", 0)),
                        "close": float(k.get("close", 0)),
                        "volume": float(k.get("quoteVol", 0)),
                    })
                if new_rows:
                    new_df = pd.DataFrame(new_rows)
                    df = pd.concat([cached_df, new_df]).drop_duplicates(
                        subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                    # Keep last MR_WARMUP_BARS bars
                    if len(df) > MR_WARMUP_BARS + 50:
                        df = df.tail(MR_WARMUP_BARS + 50).reset_index(drop=True)
                    candle_cache[sym] = df
                else:
                    df = cached_df
            else:
                df = cached_df
        else:
            df = fetch_klines_paginated(sym, n_bars=MR_WARMUP_BARS)
            if df is None or len(df) < 200:
                time.sleep(0.1)
                continue
            candle_cache[sym] = df

        current_close = float(df.iloc[-1]["close"])
        current_high = float(df.iloc[-1]["high"])
        current_low = float(df.iloc[-1]["low"])
        current_ts = str(df.iloc[-1]["timestamp"])

        # Step 3: Check open positions for TP/SL hits
        closed_trades = pos_mgr.check_tp_sl(sym, current_high, current_low,
                                             current_close, current_ts)
        for ct in closed_trades:
            log_closed_trade(ct)
            stats["closed"] += 1
            stats["pnl"] += ct.get("pnl_pct", 0)
            outcome = ct["outcome"]
            pnl = ct.get("pnl_pct", 0)
            logger.info("  CLOSED %s %s %s: %s @ %.6g -> %.6g  PnL=%.2f%%",
                        ct["strategy"].upper(), ct["side"].upper(), sym,
                        outcome, ct["entry"], ct["exit_price"], pnl)

        # Step 4: Scan for new setups (skip if already have position)
        # --- Mean Reversion ---
        mr_setup = None
        if not pos_mgr.has_position(sym, "mean_reversion") and len(df) >= 750:
            try:
                mr_setup = detect_mr_setup_live(df, sym, mr_cfg)
            except Exception as e:
                logger.debug("MR scan error %s: %s", sym, e)

        if mr_setup:
            stats["mr_setups"] += 1

            # Determine position size
            # Force dummy trade if pre-chop trend is unclear (no directional conviction)
            if mr_setup.get("pre_chop_trend") == "unclear":
                risk_pct = DUMMY_RISK_PCT
            else:
                risk_pct = STANDARD_RISK_PCT if mr_setup["dps_confidence"] in ("max", "high") else DUMMY_RISK_PCT
            position_usd = DUMMY_BALANCE * risk_pct / 100

            mr_setup["position_usd"] = round(position_usd, 2)
            mr_setup["risk_pct"] = risk_pct
            mr_setup["bars_held"] = 0

            # Fetch depth for comparison
            depth_alt = None
            depth_data = fetch_bitunix_depth(sym, limit="max")
            if depth_data:
                depth_alt = get_depth_alternative(
                    sym, depth_data, current_close,
                    mr_setup["side"], "mean_reversion")

            pos_mgr.open_position(mr_setup)
            log_trade_entry(mr_setup, depth_alt)

            logger.info("  NEW MR %s %s: entry=%.6g tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) "
                        "RR=%.2f DPS=%d [%s]",
                        mr_setup["side"].upper(), sym,
                        mr_setup["entry"], mr_setup["tp"], mr_setup["tp_pct"],
                        mr_setup["sl"], mr_setup["sl_pct"],
                        mr_setup["rr"], mr_setup["dps_total"],
                        mr_setup["dps_confidence"])
            if depth_alt:
                logger.info("    DEPTH alt: tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) RR=%.2f wall=$%s",
                            depth_alt["depth_tp"], depth_alt["depth_tp_pct"],
                            depth_alt["depth_sl"], depth_alt["depth_sl_pct"],
                            depth_alt["depth_rr"],
                            f"{depth_alt['depth_sl_wall_usd']:,.0f}")

        # --- Momentum ---
        momo_setup = None
        if not pos_mgr.has_position(sym, "momentum") and len(df) >= 370:
            try:
                momo_setup = detect_momo_setup_live(df, sym)
            except Exception as e:
                logger.debug("Momo scan error %s: %s", sym, e)

        if momo_setup:
            stats["momo_setups"] += 1

            risk_pct = STANDARD_RISK_PCT if momo_setup["dps_confidence"] in ("max", "high") else DUMMY_RISK_PCT
            position_usd = DUMMY_BALANCE * risk_pct / 100

            momo_setup["position_usd"] = round(position_usd, 2)
            momo_setup["risk_pct"] = risk_pct
            momo_setup["bars_held"] = 0

            depth_alt = None
            depth_data = fetch_bitunix_depth(sym, limit="max")
            if depth_data:
                depth_alt = get_depth_alternative(
                    sym, depth_data, current_close,
                    momo_setup["side"], "momentum")

            pos_mgr.open_position(momo_setup)
            log_trade_entry(momo_setup, depth_alt)

            logger.info("  NEW MOMO %s %s: entry=%.6g tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) "
                        "RR=%.2f DPS=%d [%s] tier=%s",
                        momo_setup["side"].upper(), sym,
                        momo_setup["entry"], momo_setup["tp"], momo_setup["tp_pct"],
                        momo_setup["sl"], momo_setup["sl_pct"],
                        momo_setup["rr"], momo_setup["dps_total"],
                        momo_setup["dps_confidence"],
                        momo_setup.get("quality_tier", "?"))
            if depth_alt:
                logger.info("    DEPTH alt: tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) RR=%.2f wall=$%s",
                            depth_alt["depth_tp"], depth_alt["depth_tp_pct"],
                            depth_alt["depth_sl"], depth_alt["depth_sl_pct"],
                            depth_alt["depth_rr"],
                            f"{depth_alt['depth_sl_wall_usd']:,.0f}")

        time.sleep(0.12)  # Rate limit between symbols

    # Log cycle summary
    n_open = len(pos_mgr.positions)
    log_cycle(cycle_num, stats["coins"], stats["mr_setups"], stats["momo_setups"],
              n_open, stats["closed"], stats["pnl"])

    logger.info("Cycle %d done: %d MR setups, %d Momo setups, %d closed (PnL=%.2f%%), %d open",
                cycle_num, stats["mr_setups"], stats["momo_setups"],
                stats["closed"], stats["pnl"], n_open)

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live Dummy Trader")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between cycles (default: 60)")
    parser.add_argument("--min-vol", type=float, default=200_000,
                        help="Min 5-minute volume (default: 200000)")
    parser.add_argument("--top-n", type=int, default=60,
                        help="Max coins to scan (default: 60)")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show current positions and stats, then exit")
    args = parser.parse_args()

    TRADER_DIR.mkdir(parents=True, exist_ok=True)

    pos_mgr = PositionManager()
    mr_cfg = MRSettings()

    if args.status:
        print(f"\n{'=' * 60}")
        print(f"DUMMY TRADER STATUS")
        print(f"{'=' * 60}")
        print(f"Open positions: {len(pos_mgr.positions)}")
        for p in pos_mgr.positions:
            print(f"  {p['strategy']:15s} {p['side']:5s} {p['symbol']:15s} "
                  f"entry={p['entry']:.6g}  tp={p['tp']:.6g}  sl={p['sl']:.6g}  "
                  f"bars={p.get('bars_held', 0)}")

        if CLOSED_TRADES_LOG.exists():
            ct = pd.read_csv(str(CLOSED_TRADES_LOG))
            total_trades = len(ct)
            tp_count = (ct["outcome"] == "TP").sum()
            sl_count = (ct["outcome"] == "SL").sum()
            to_count = (ct["outcome"] == "TIMEOUT").sum()
            total_pnl = ct["pnl_pct"].sum()
            wr = tp_count / (tp_count + sl_count) * 100 if (tp_count + sl_count) > 0 else 0

            print(f"\nClosed trades: {total_trades}")
            print(f"  TP: {tp_count}  SL: {sl_count}  TIMEOUT: {to_count}")
            print(f"  Win Rate: {wr:.1f}%")
            print(f"  Total PnL: {total_pnl:+.2f}%")
            print(f"  Avg PnL/trade: {total_pnl / total_trades:+.3f}%")

            # By strategy
            for strat in ["momentum", "mean_reversion"]:
                sub = ct[ct["strategy"] == strat]
                if len(sub) == 0:
                    continue
                tp_s = (sub["outcome"] == "TP").sum()
                sl_s = (sub["outcome"] == "SL").sum()
                wr_s = tp_s / (tp_s + sl_s) * 100 if (tp_s + sl_s) > 0 else 0
                pnl_s = sub["pnl_pct"].sum()
                print(f"\n  {strat.upper()}:")
                print(f"    Trades: {len(sub)}  TP: {tp_s}  SL: {sl_s}  WR: {wr_s:.1f}%  PnL: {pnl_s:+.2f}%")

        if DEPTH_COMPARISON_LOG.exists():
            dc = pd.read_csv(str(DEPTH_COMPARISON_LOG))
            if len(dc) > 0:
                print(f"\nDepth Comparison ({len(dc)} trades):")
                print(f"  Avg Strategy RR: {dc['strat_rr'].mean():.2f}")
                print(f"  Avg Depth RR:    {dc['depth_rr'].mean():.2f}")
                print(f"  Avg Strategy TP%: {dc['strat_tp_pct'].mean():.2f}")
                print(f"  Avg Depth TP%:    {dc['depth_tp_pct'].mean():.2f}")
        return

    logger.info("Starting Live Dummy Trader")
    logger.info("  Interval: %ds | Min vol: $%s | Top N: %d",
                args.interval, f"{min_vol:,.0f}" if (min_vol := args.min_vol) else "0",
                args.top_n)
    logger.info("  Positions file: %s", POSITIONS_FILE)
    logger.info("  Open positions: %d", len(pos_mgr.positions))

    candle_cache = {}  # {symbol: DataFrame} - avoids re-fetching full history

    if args.once:
        trading_cycle(pos_mgr, mr_cfg, args.min_vol, args.top_n, 1, candle_cache)
        return

    cycle = 0
    while True:
        cycle += 1
        try:
            trading_cycle(pos_mgr, mr_cfg, args.min_vol, args.top_n, cycle, candle_cache)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error("Cycle error: %s", e, exc_info=True)

        logger.info("Sleeping %ds...", args.interval)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break


if __name__ == "__main__":
    main()
