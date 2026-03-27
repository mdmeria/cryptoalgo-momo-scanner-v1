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

_log_fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s",
                              datefmt="%H:%M:%S")

_console_handler = logging.StreamHandler(stream=sys.stderr)
_console_handler.setFormatter(_log_fmt)

_dummy_log_dir = Path("datasets/live/dummy_trader")
_dummy_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(
    _dummy_log_dir / "trader_log.txt", encoding="utf-8")
_file_handler.setFormatter(_log_fmt)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_console_handler)
logger.addHandler(_file_handler)

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
from strategies import (
    StrategyConfig,
    MRSettings,
    MomoGateSettings,
    check_mr_gates_at_bar,
    check_momo_gates_at_bar,
    detect_setups,
    get_risk_pct,
    check_75pct_tp_rule,
    MarketCondition,
)

# Load configs
_strategy_cfg = StrategyConfig.from_json()
_mr_cfg = MRSettings()
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
NEW_COINS_LOG = TRADER_DIR / "new_coins.csv"

# Approved symbols whitelist
APPROVED_SYMBOLS_FILE = Path("approved_symbols.txt")
_approved_symbols: set = set()
_alerted_new_coins: set = set()  # track already-alerted coins to avoid spam


def load_approved_symbols() -> set:
    """Load approved symbols from whitelist file."""
    if not APPROVED_SYMBOLS_FILE.exists():
        return set()  # no whitelist = allow all
    with open(APPROVED_SYMBOLS_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def log_new_coin(symbol: str, volume_5m: float = 0, trades_5m: int = 0):
    """Log a new coin that's not in the whitelist for manual review."""
    global _alerted_new_coins
    if symbol in _alerted_new_coins:
        return
    _alerted_new_coins.add(symbol)

    TRADER_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not NEW_COINS_LOG.exists()
    ts = datetime.now(timezone.utc).isoformat()
    with open(NEW_COINS_LOG, "a", encoding="utf-8") as f:
        if write_header:
            f.write("timestamp,symbol,volume_5m,trades_5m,status\n")
        f.write(f"{ts},{symbol},{volume_5m:.0f},{trades_5m},pending\n")

    logger.info("  NEW COIN not in whitelist: %s (vol=$%.0f, trades=%d) — logged for review",
                symbol, volume_5m, trades_5m)


# How many 1m bars we need for each strategy
MR_WARMUP_BARS = 800    # MR scanner needs up to 720 + buffer
MOMO_WARMUP_BARS = 150  # Momo needs smma30 + 120 bar staircase check

# Dummy account settings (loaded from strategy_config.json via _strategy_cfg)


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
            entry = pos["entry"]
            sl_orig = pos.get("sl_original", pos["sl"])
            tp = pos["tp"]
            bars = pos.get("bars_held", 0) + 1
            pos["bars_held"] = bars

            # Save original SL on first check (for R calculation)
            if "sl_original" not in pos:
                pos["sl_original"] = pos["sl"]

            # --- Trail rules: move SL to 0.1R when conditions met ---
            if not pos.get("sl_trailed", False):
                if pos["side"] == "long":
                    r_dist = entry - sl_orig          # 1R distance
                    target_09r = entry + r_dist * 0.9  # 0.9R onside
                    new_sl = entry + r_dist * 0.1      # 0.1R onside

                    # Rule 1: price reached 0.9R onside
                    if current_high >= target_09r:
                        pos["sl"] = round(new_sl, 8)
                        pos["sl_trailed"] = True

                    # Rule 2: 60+ bars held and price is onside
                    elif bars >= 60 and current_close > entry:
                        pos["sl"] = round(new_sl, 8)
                        pos["sl_trailed"] = True
                else:  # short
                    r_dist = sl_orig - entry           # 1R distance
                    target_09r = entry - r_dist * 0.9  # 0.9R onside
                    new_sl = entry - r_dist * 0.1      # 0.1R onside

                    # Rule 1: price reached 0.9R onside
                    if current_low <= target_09r:
                        pos["sl"] = round(new_sl, 8)
                        pos["sl_trailed"] = True

                    # Rule 2: 60+ bars held and price is onside
                    elif bars >= 60 and current_close < entry:
                        pos["sl"] = round(new_sl, 8)
                        pos["sl_trailed"] = True

            # --- Check TP/SL ---
            if pos["side"] == "long":
                if current_low <= pos["sl"]:
                    hit = "TRAIL_SL" if pos.get("sl_trailed") else "SL"
                    exit_price = pos["sl"]
                elif current_high >= pos["tp"]:
                    hit = "TP"
                    exit_price = pos["tp"]
            else:  # short
                if current_high >= pos["sl"]:
                    hit = "TRAIL_SL" if pos.get("sl_trailed") else "SL"
                    exit_price = pos["sl"]
                elif current_low <= pos["tp"]:
                    hit = "TP"
                    exit_price = pos["tp"]

            if hit:
                pos["outcome"] = hit
                pos["exit_price"] = exit_price
                pos["exit_ts"] = current_ts

                if pos["side"] == "long":
                    pos["pnl_pct"] = (exit_price - entry) / entry * 100
                else:
                    pos["pnl_pct"] = (entry - exit_price) / entry * 100

                pos["pnl_usd"] = pos["pnl_pct"] / 100 * pos.get("position_usd", 0)
                closed.append(pos)
            else:
                # No timeout — trail rules keep trade alive
                remaining.append(pos)

        self.positions = remaining
        self._save()
        return closed

    def close_position(self, symbol: str, strategy: str,
                       exit_price: float, exit_ts: str,
                       outcome: str) -> Optional[dict]:
        """Manually close a specific position. Returns the closed trade dict."""
        remaining = []
        closed = None
        for pos in self.positions:
            if closed is None and pos["symbol"] == symbol and pos["strategy"] == strategy:
                pos["outcome"] = outcome
                pos["exit_price"] = exit_price
                pos["exit_ts"] = exit_ts
                if pos["side"] == "long":
                    pos["pnl_pct"] = (exit_price - pos["entry"]) / pos["entry"] * 100
                else:
                    pos["pnl_pct"] = (pos["entry"] - exit_price) / pos["entry"] * 100
                pos["pnl_usd"] = pos["pnl_pct"] / 100 * pos.get("position_usd", 0)
                closed = pos
            else:
                remaining.append(pos)
        if closed:
            self.positions = remaining
            self._save()
        return closed

    def get_all_symbols(self) -> set:
        """Get all symbols with open positions."""
        return {p["symbol"] for p in self.positions}


# ---------------------------------------------------------------------------
# MR Setup Detection (live, latest bar)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Setup Detection — delegates to strategies.py
# ---------------------------------------------------------------------------

def detect_all_setups(df: pd.DataFrame, symbol: str,
                      depth_data: dict = None) -> list[dict]:
    """
    Run all enabled strategies on the latest bar.
    Returns list of setup dicts via the unified strategies module.
    """
    return detect_setups(df, symbol, _strategy_cfg, _mr_cfg, _momo_gate_cfg,
                         depth_data=depth_data)


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

def trading_cycle(pos_mgr: PositionManager,
                  min_vol: float, top_n: int, cycle_num: int,
                  candle_cache: dict, depth_cooldown: dict = None,
                  market_cond: MarketCondition = None,
                  pending_limits: dict = None) -> dict:
    """
    Run one trading cycle:
    1. Get active coins from Orion
    2. Fetch candles for each
    3. Check for MR and Momo setups
    4. Enter dummy trades (mr_chop uses limit orders)
    5. Check open positions for TP/SL hits
    6. Manage pending limit orders (3 min expiry)
    """
    if pending_limits is None:
        pending_limits = {}
    stats = {"coins": 0, "mr_setups": 0, "momo_setups": 0, "closed": 0, "pnl": 0.0}

    # Step 1: Get active coins
    coins = fetch_orion_active_coins(min_vol_5m=min_vol, top_n=top_n)
    if not coins:
        logger.warning("No coins from Orion")
        return stats

    # Also include symbols with open positions
    open_syms = pos_mgr.get_all_symbols()
    coin_symbols = {c["symbol"] for c in coins}

    # Filter by approved whitelist (if loaded) — log new coins for review
    if _approved_symbols:
        for c in coins:
            sym = c["symbol"]
            if sym not in _approved_symbols and sym not in open_syms:
                log_new_coin(sym, c.get("volume_5m", 0), c.get("trades_5m", 0))
        coin_symbols = {s for s in coin_symbols if s in _approved_symbols}

    all_symbols = coin_symbols | open_syms

    stats["coins"] = len(all_symbols)
    logger.info("Cycle %d: %d coins (%d active + %d with positions)",
                cycle_num, len(all_symbols), len(coin_symbols), len(open_syms - coin_symbols))

    # Step 2: Update market conditions every 120 cycles (~2 hours)
    if market_cond is not None:
        if cycle_num == 1 or cycle_num % 120 == 0:
            # Update BTC
            btc_sym = "BTCUSDT"
            btc_df = fetch_klines_paginated(btc_sym, n_bars=MR_WARMUP_BARS)
            if btc_df is not None and len(btc_df) >= 150:
                candle_cache[btc_sym] = btc_df
            if btc_sym in candle_cache:
                market_cond.update_btc(candle_cache[btc_sym])

            # Update breadth from cached candle data
            if len(candle_cache) > 10:
                market_cond.update_breadth(candle_cache)

            logger.info("  %s (updated)", market_cond.summary())
        else:
            # Log current score without recalculating
            if cycle_num % 10 == 0:  # log every 10 cycles to reduce noise
                logger.info("  %s", market_cond.summary())

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

        # Step 3b: Check 75% TP rule for depth_bounce positions
        for pos in list(pos_mgr.positions):
            if pos.get("symbol") != sym:
                continue
            if pos.get("strategy") != "depth_bounce":
                continue
            if check_75pct_tp_rule(pos, current_high, current_low):
                ct_75 = pos_mgr.close_position(
                    sym, "depth_bounce", current_close, current_ts, "75PCT_TP")
                if ct_75:
                    closed_trades.append(ct_75)

        for ct in closed_trades:
            log_closed_trade(ct)
            stats["closed"] += 1
            stats["pnl"] += ct.get("pnl_pct", 0)
            outcome = ct["outcome"]
            pnl = ct.get("pnl_pct", 0)
            logger.info("  CLOSED %s %s %s: %s @ %.6g -> %.6g  PnL=%.2f%%",
                        ct["strategy"].upper(), ct["side"].upper(), sym,
                        outcome, ct["entry"], ct["exit_price"], pnl)

        # Step 4: Fetch depth (used by depth strategy + comparison logging)
        depth_data = None
        if _strategy_cfg.enable_depth or True:  # always fetch for comparison
            depth_data = fetch_bitunix_depth(sym, limit="max")

        # Step 5: Scan for new setups via unified strategy module
        try:
            found_setups = detect_all_setups(df, sym, depth_data=depth_data)
        except Exception as e:
            logger.debug("Strategy scan error %s: %s", sym, e)
            found_setups = []

        for setup in found_setups:
            strat = setup["strategy"]
            if pos_mgr.has_position(sym, strat):
                continue

            # Depth/depth_bounce cooldown: skip if recently traded this symbol
            if strat in ("depth", "depth_bounce", "bouncy_ball", "mr_chop") and depth_cooldown is not None:
                cd_key = f"{sym}_{strat}"
                last_cycle = depth_cooldown.get(cd_key, -999)
                if cycle_num - last_cycle < 30:  # ~30 min cooldown
                    continue

            # Market condition filter
            if market_cond is not None:
                side = setup["side"]
                # Momo: also enforce DPS >= 4
                if strat == "momentum" and setup.get("dps_total", 0) < 4:
                    continue
                if not market_cond.is_allowed(strat, side):
                    logger.debug("  SKIPPED %s %s %s: market=%s",
                                 strat, side, sym, market_cond.summary())
                    continue

            if strat == "mean_reversion":
                stats["mr_setups"] += 1
            elif strat == "depth":
                stats["mr_setups"] += 1  # count depth in mr stats for now
            else:
                stats["momo_setups"] += 1

            risk_pct = get_risk_pct(setup, _strategy_cfg)
            position_usd = _strategy_cfg.dummy_balance * risk_pct / 100

            setup["position_usd"] = round(position_usd, 2)
            setup["risk_pct"] = risk_pct
            setup["market_score"] = market_cond.score if market_cond else 0
            setup["bars_held"] = 0

            # MR Chop: use limit order at level price
            if strat == "mr_chop":
                limit_key = f"{sym}_mr_chop"
                if limit_key not in pending_limits:
                    import time as _time
                    # Fetch depth data for logging
                    mr_depth_alt = None
                    if depth_data:
                        try:
                            mr_depth_alt = get_depth_alternative(
                                sym, depth_data, current_close,
                                setup["side"], strat)
                        except Exception:
                            pass
                    pending_limits[limit_key] = {
                        **setup,
                        "placed_at": _time.time(),
                        "limit_price": setup["entry"],
                        "depth_alt": mr_depth_alt,
                    }
                    variant = setup.get("strategy_variant", "mr_chop_v2")
                    logger.info("  LIMIT MR_CHOP(%s) %s %s: price=%.6g tp=%.6g(%.2f%%) "
                                "sl=%.6g(%.2f%%) RR=%.2f DPS=%d src=%s sw=%d (3min expiry)",
                                variant, setup["side"].upper(), sym,
                                setup["entry"], setup["tp"], setup["tp_pct"],
                                setup["sl"], setup["sl_pct"],
                                setup["rr"], setup["dps_total"],
                                setup.get("level_source", "?"),
                                setup.get("n_swings", 0))
                    if mr_depth_alt:
                        logger.info("    DEPTH alt: tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) RR=%.2f wall=$%s",
                                    mr_depth_alt["depth_tp"], mr_depth_alt["depth_tp_pct"],
                                    mr_depth_alt["depth_sl"], mr_depth_alt["depth_sl_pct"],
                                    mr_depth_alt["depth_rr"],
                                    f"{mr_depth_alt['depth_sl_wall_usd']:,.0f}")
                continue  # don't open position yet — wait for fill

            # Depth comparison (use already-fetched depth_data)
            depth_alt = None
            if depth_data and strat != "depth":
                depth_alt = get_depth_alternative(
                    sym, depth_data, current_close,
                    setup["side"], strat)

            pos_mgr.open_position(setup)
            log_trade_entry(setup, depth_alt)

            # Set depth/depth_bounce cooldown
            if strat in ("depth", "depth_bounce", "bouncy_ball", "mr_chop") and depth_cooldown is not None:
                depth_cooldown[f"{sym}_{strat}"] = cycle_num

            strat_labels = {"mean_reversion": "MR", "strict_mr": "STRICT_MR",
                           "momentum": "MOMO", "depth": "DEPTH",
                           "depth_bounce": "DEPTH_BOUNCE",
                           "bouncy_ball": "BB_MR",
                           "mr_chop": "MR_CHOP"}
            strat_label = strat_labels.get(strat, strat.upper())
            if strat in ("depth", "depth_bounce"):
                zct_align = setup.get("zct_alignment", "unclear")
                zct_mr = setup.get("zct_mr_reason", "?")
                zct_momo = setup.get("zct_momo_reason", "?")
                logger.info("  NEW %s %s %s: entry=%.6g tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) "
                            "RR=%.2f align=%s DPS=%d [%s]",
                            strat_label, setup["side"].upper(), sym,
                            setup["entry"], setup["tp"], setup["tp_pct"],
                            setup["sl"], setup["sl_pct"],
                            setup["rr"], zct_align, setup["dps_total"],
                            setup["dps_confidence"])
                if strat == "depth_bounce":
                    logger.info("    bounce: %s wall=%.6g wick=%.6g | "
                                "walls: SL=$%s(%.1fx) TP=$%s(%.1fx) | MR:%s Momo:%s",
                                setup.get("bounce_type", "?"),
                                setup.get("wall_price", 0),
                                setup.get("wick_price", 0),
                                f"{setup.get('sl_wall_usd', 0):,.0f}",
                                setup.get("sl_wall_strength", 0),
                                f"{setup.get('tp_wall_usd', 0):,.0f}",
                                setup.get("tp_wall_strength", 0),
                                zct_mr, zct_momo)
                else:
                    logger.info("    walls: SL=$%s(%.1fx) TP=$%s(%.1fx) | imb=%.3f | MR:%s Momo:%s",
                                f"{setup.get('sl_wall_usd', 0):,.0f}",
                                setup.get("sl_wall_strength", 0),
                                f"{setup.get('tp_wall_usd', 0):,.0f}",
                                setup.get("tp_wall_strength", 0),
                                setup.get("imbalance_1pct", 0),
                                zct_mr, zct_momo)
            elif strat == "bouncy_ball":
                logger.info("  NEW %s %s %s: entry=%.6g tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) "
                            "RR=%.2f clean=%d [%s]",
                            strat_label, setup["side"].upper(), sym,
                            setup["entry"], setup["tp"], setup["tp_pct"],
                            setup["sl"], setup["sl_pct"],
                            setup["rr"], setup.get("clean_score", 0),
                            setup.get("dps_confidence", "?"))
                logger.info("    range: %.6g-%.6g (%.2f%%) up_t=%d lo_t=%d "
                            "inside=%.0f%% pre=%s(%.1f%%) dur=%dmin",
                            setup.get("range_lower", 0), setup.get("range_upper", 0),
                            setup.get("range_pct", 0),
                            setup.get("upper_touches", 0), setup.get("lower_touches", 0),
                            setup.get("inside_pct", 0),
                            setup.get("pre_trend", "?"), setup.get("pre_trend_pct", 0),
                            setup.get("range_duration_bars", 0))
            elif strat == "mr_chop":
                variant = setup.get("strategy_variant", "?")
                logger.info("  NEW %s(%s) %s %s: entry=%.6g tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) "
                            "RR=%.2f DPS=%d [%s]",
                            strat_label, variant, setup["side"].upper(), sym,
                            setup["entry"], setup["tp"], setup["tp_pct"],
                            setup["sl"], setup["sl_pct"],
                            setup["rr"], setup.get("dps_total", 0),
                            setup.get("dps_confidence", "?"))
                logger.info("    swings=%d last=%dm vol=%s chop=%.1fh pre=%s shift=%.2f%% touches=%d",
                            setup.get("n_swings", 0), setup.get("last_swing_mins", 0),
                            setup.get("vol_type", "?"), setup.get("chop_hrs", 0),
                            setup.get("pre_trend", "?"), setup.get("shift_pct", 0),
                            setup.get("level_touches", 0))
            else:
                tp_src = setup.get("tp_source", "strategy")
                tp_tag = f" tp_src={tp_src}" if tp_src != "strategy" else ""
                logger.info("  NEW %s %s %s: entry=%.6g tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) "
                            "RR=%.2f DPS=%d [%s]%s",
                            strat_label, setup["side"].upper(), sym,
                            setup["entry"], setup["tp"], setup["tp_pct"],
                            setup["sl"], setup["sl_pct"],
                            setup["rr"], setup["dps_total"],
                            setup["dps_confidence"], tp_tag)
            if depth_alt:
                logger.info("    DEPTH alt: tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) RR=%.2f wall=$%s",
                            depth_alt["depth_tp"], depth_alt["depth_tp_pct"],
                            depth_alt["depth_sl"], depth_alt["depth_sl_pct"],
                            depth_alt["depth_rr"],
                            f"{depth_alt['depth_sl_wall_usd']:,.0f}")

        time.sleep(0.12)  # Rate limit between symbols

    # --- Manage pending MR_CHOP limit orders ---
    import time as _time
    expired_keys = []
    for limit_key, lim in list(pending_limits.items()):
        sym_lim = lim["symbol"]
        elapsed = _time.time() - lim["placed_at"]
        limit_price = lim["limit_price"]
        side_lim = lim["side"]

        # Check if filled: did price reach limit level?
        if sym_lim in candle_cache:
            df_lim = candle_cache[sym_lim]
            if len(df_lim) > 0:
                last_bar = df_lim.iloc[-1]
                if side_lim == "short" and float(last_bar["high"]) >= limit_price:
                    # Filled! Open position
                    pos_mgr.open_position(lim)
                    log_trade_entry(lim)
                    logger.info("  LIMIT FILLED MR_CHOP %s %s: fill=%.6g waited=%.0fs",
                                side_lim.upper(), sym_lim, limit_price, elapsed)
                    expired_keys.append(limit_key)
                    continue
                elif side_lim == "long" and float(last_bar["low"]) <= limit_price:
                    pos_mgr.open_position(lim)
                    log_trade_entry(lim)
                    logger.info("  LIMIT FILLED MR_CHOP %s %s: fill=%.6g waited=%.0fs",
                                side_lim.upper(), sym_lim, limit_price, elapsed)
                    expired_keys.append(limit_key)
                    continue

        # Expired? (3 minutes)
        if elapsed >= lim.get("limit_expiry_mins", 3) * 60:
            logger.info("  MISSED MR_CHOP %s %s: price=%.6g expired after %.0fs",
                        side_lim.upper(), sym_lim, limit_price, elapsed)
            expired_keys.append(limit_key)

    for k in expired_keys:
        pending_limits.pop(k, None)

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

    # Load approved symbols whitelist
    global _approved_symbols
    _approved_symbols = load_approved_symbols()

    logger.info("Starting Live Dummy Trader")
    logger.info("  Interval: %ds | Min vol: $%s | Top N: %d",
                args.interval, f"{min_vol:,.0f}" if (min_vol := args.min_vol) else "0",
                args.top_n)
    logger.info("  Strategies: MR=%s  StrictMR=%s  Momo=%s  Depth=%s  DepthBounce=%s  BB=%s  (min DPS=%d)",
                "ON" if _strategy_cfg.enable_mean_reversion else "OFF",
                "ON" if _strategy_cfg.enable_strict_mr else "OFF",
                "ON" if _strategy_cfg.enable_momentum else "OFF",
                "ON" if _strategy_cfg.enable_depth else "OFF",
                "ON" if _strategy_cfg.enable_depth_bounce else "OFF",
                "ON" if _strategy_cfg.enable_bouncy_ball else "OFF",
                _strategy_cfg.min_dps_live)
    logger.info("  Approved symbols: %d (from %s)", len(_approved_symbols),
                APPROVED_SYMBOLS_FILE if _approved_symbols else "NONE - all allowed")
    logger.info("  Positions file: %s", POSITIONS_FILE)
    logger.info("  Open positions: %d", len(pos_mgr.positions))

    candle_cache = {}  # {symbol: DataFrame} - avoids re-fetching full history
    depth_cooldown = {}  # {symbol: cycle_num} - cooldown for depth re-entry
    pending_limits = {}  # {key: setup} - pending limit orders for mr_chop
    market_cond = MarketCondition()

    if args.once:
        trading_cycle(pos_mgr, args.min_vol, args.top_n, 1, candle_cache, depth_cooldown, market_cond, pending_limits)
        return

    cycle = 0
    while True:
        cycle += 1
        try:
            trading_cycle(pos_mgr, args.min_vol, args.top_n, cycle, candle_cache, depth_cooldown, market_cond, pending_limits)
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
