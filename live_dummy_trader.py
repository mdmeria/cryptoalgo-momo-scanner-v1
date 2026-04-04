#!/usr/bin/env python3
"""
WebSocket-Based Live Dummy Trader

Mirrors live_trader_ws.py architecture exactly, but uses local position
tracking instead of exchange execution. This ensures dummy results are
representative of actual live performance.

Architecture (same as live):
  1. Startup: REST backfill 800 bars for all active Orion coins
  2. WS kline: on candle close → append to cache → queue for batch check
  3. Periodic loop (2s): process queued symbols → depth fetch → strategies
  4. Position management: local TP/SL/trail check on each candle close

Usage:
  python live_dummy_trader.py [--min-vol 200000] [--top-n 60] [--once] [--status]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Logging — both console and file
_log_fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s",
                              datefmt="%H:%M:%S")

_dummy_log_dir = Path("datasets/live/dummy_trader")
_dummy_log_dir.mkdir(parents=True, exist_ok=True)

_console_handler = logging.StreamHandler(stream=sys.stderr)
_console_handler.setFormatter(_log_fmt)
_file_handler = logging.FileHandler(
    _dummy_log_dir / "trader_log.txt", encoding="utf-8")
_file_handler.setFormatter(_log_fmt)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_console_handler)
logger.addHandler(_file_handler)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from ws_client import BitunixWS
from live_data_collector import (
    fetch_orion_active_coins,
    fetch_bitunix_depth,
    fetch_bitunix_klines,
    LIVE_DIR,
    BITUNIX_BASE,
)
from strategies import (
    StrategyConfig,
    MRSettings,
    MomoGateSettings,
    detect_setups,
    get_risk_pct,
    MarketCondition,
)
from depth_tp_sl_analyzer import compute_depth_tp_sl
from depth_snapshot import compute_depth_snapshot, compute_close_snapshot

import requests

# Load configs
_strategy_cfg = StrategyConfig.from_json()
_mr_cfg = MRSettings()
_momo_gate_cfg = MomoGateSettings.from_json("momo_gate_settings.json")

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
TRADER_DIR = LIVE_DIR / "dummy_trader"
TRADES_LOG = TRADER_DIR / "trades.csv"
POSITIONS_FILE = TRADER_DIR / "open_positions.json"
DEPTH_COMPARISON_LOG = TRADER_DIR / "depth_comparison.csv"
CLOSED_TRADES_LOG = TRADER_DIR / "closed_trades.csv"
NEW_COINS_LOG = TRADER_DIR / "new_coins.csv"

APPROVED_SYMBOLS_FILE = Path("approved_symbols.txt")

MR_WARMUP_BARS = 800
MOMO_WARMUP_BARS = 150


# ---------------------------------------------------------------------------
# Approved Symbols
# ---------------------------------------------------------------------------

def load_approved_symbols() -> set:
    if not APPROVED_SYMBOLS_FILE.exists():
        return set()
    with open(APPROVED_SYMBOLS_FILE) as f:
        return {line.strip() for line in f if line.strip()}


_approved_symbols: set = set()
_alerted_new_coins: set = set()


def log_new_coin(symbol: str, volume_5m: float = 0, trades_5m: int = 0):
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
    logger.info("  NEW COIN not in whitelist: %s (vol=$%.0f) — logged for review",
                symbol, volume_5m)


# ---------------------------------------------------------------------------
# Bitunix paginated kline fetch (REST — startup backfill only)
# ---------------------------------------------------------------------------

def fetch_klines_paginated(symbol: str, n_bars: int = 800,
                            interval: str = "1m") -> Optional[pd.DataFrame]:
    """Fetch up to n_bars of 1m candles from Bitunix by paginating."""
    all_rows = []
    now_ms = int(time.time() * 1000)
    end_ms = now_ms
    bars_remaining = n_bars
    max_requests = 12

    for _ in range(max_requests):
        if bars_remaining <= 0:
            break
        batch_size = min(bars_remaining, 100)
        start_ms = end_ms - batch_size * 60 * 1000
        try:
            resp = requests.get(
                f"{BITUNIX_BASE}/api/v1/futures/market/kline",
                params={
                    "symbol": symbol, "interval": interval,
                    "startTime": str(start_ms), "endTime": str(end_ms),
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
        end_ms = start_ms - 1
        time.sleep(0.12)

    if not all_rows:
        return None
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Position Manager (local — no exchange)
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
        return any(
            p["symbol"] == symbol and p["strategy"] == strategy
            for p in self.positions
        )

    def has_any_position(self, symbol: str) -> bool:
        return any(p["symbol"] == symbol for p in self.positions)

    def open_position(self, trade: dict):
        self.positions.append(trade)
        self._save()

    def check_tp_sl(self, symbol: str, current_high: float, current_low: float,
                     current_close: float, current_ts: str) -> list[dict]:
        """Check if any positions hit TP or SL. Returns closed trades."""
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

            if "sl_original" not in pos:
                pos["sl_original"] = pos["sl"]

            # --- Trail rules: move SL to 0.1R when conditions met ---
            if not pos.get("sl_trailed", False):
                if pos["side"] == "long":
                    r_dist = entry - sl_orig
                    if r_dist > 0:
                        target_09r = entry + r_dist * 0.9
                        new_sl = entry + r_dist * 0.1
                        if current_high >= target_09r:
                            pos["sl"] = round(new_sl, 8)
                            pos["sl_trailed"] = True
                        elif bars >= 60 and current_close > entry:
                            pos["sl"] = round(new_sl, 8)
                            pos["sl_trailed"] = True
                else:
                    r_dist = sl_orig - entry
                    if r_dist > 0:
                        target_09r = entry - r_dist * 0.9
                        new_sl = entry - r_dist * 0.1
                        if current_low <= target_09r:
                            pos["sl"] = round(new_sl, 8)
                            pos["sl_trailed"] = True
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
            else:
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
                remaining.append(pos)

        self.positions = remaining
        self._save()
        return closed

    def get_all_symbols(self) -> set:
        return {p["symbol"] for p in self.positions}


# ---------------------------------------------------------------------------
# Trade Logging
# ---------------------------------------------------------------------------

def log_trade_entry(trade: dict, depth_alt: Optional[dict] = None):
    TRADER_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not TRADES_LOG.exists()
    with open(TRADES_LOG, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(trade.keys()) + "\n")
        vals = []
        for v in trade.values():
            s = str(v) if v is not None else ""
            if "," in s:
                s = f'"{s}"'
            vals.append(s)
        f.write(",".join(vals) + "\n")

    if depth_alt:
        comparison = {
            "timestamp": trade["timestamp"],
            "symbol": trade["symbol"],
            "strategy": trade["strategy"],
            "side": trade["side"],
            "entry": trade["entry"],
            "strat_tp": trade["tp"], "strat_tp_pct": trade["tp_pct"],
            "strat_sl": trade["sl"], "strat_sl_pct": trade["sl_pct"],
            "strat_rr": trade["rr"],
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
    TRADER_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CLOSED_TRADES_LOG.exists()
    with open(CLOSED_TRADES_LOG, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(trade.keys()) + "\n")
        vals = []
        for v in trade.values():
            s = str(v) if v is not None else ""
            if "," in s:
                s = f'"{s}"'
            vals.append(s)
        f.write(",".join(vals) + "\n")


def get_depth_alternative(symbol: str, depth_data: dict, current_price: float,
                           side: str, strategy: str) -> Optional[dict]:
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


# ═══════════════════════════════════════════════════════════════════════════
# WS Dummy Trader
# ═══════════════════════════════════════════════════════════════════════════

class WSDummyTrader:
    """WebSocket-based dummy trader — mirrors live_trader_ws.py architecture."""

    def __init__(self, min_vol: float = 200_000, top_n: int = 60):
        self.min_vol = min_vol
        self.top_n = top_n

        # WS client (public only — no API keys needed for kline)
        self.ws = BitunixWS("", "")
        self.ws.on_kline = self._handle_kline

        # Position tracking (local)
        self.pos_mgr = PositionManager()
        self.market_cond = MarketCondition()

        # Data
        self.candle_cache: dict[str, pd.DataFrame] = {}
        self.approved_symbols = load_approved_symbols()
        self.active_symbols: set[str] = set()
        self.depth_cooldown: dict[str, float] = {}
        self.pending_limits: dict[str, dict] = {}

        # Batch candle processing (same as live)
        self._pending_candle_symbols: set[str] = set()
        self._batch_lock = threading.Lock()

        # Track last processed candle timestamp per symbol
        self._last_candle_ts: dict[str, int] = {}

        # Pending WS subscriptions
        self._pending_subscriptions: set[str] = set()

        # Refresh intervals
        self._last_coin_refresh = 0.0
        self._coin_refresh_interval = 120  # 2 minutes
        self._last_mkt_refresh = 0.0
        self._mkt_refresh_interval = 7200  # 2 hours

        # Lock
        self._lock = threading.Lock()

    # --- Startup ---

    async def start(self):
        """Start the WS dummy trader."""
        TRADER_DIR.mkdir(parents=True, exist_ok=True)

        # Step 1: Load active coins from Orion
        self._refresh_active_coins()

        # Step 2: REST backfill candle history for all active symbols
        self._load_initial_candles()

        # Step 3: Initial market condition
        self._refresh_market_condition()

        logger.info("WS Dummy Trader started")
        logger.info("  Strategies: Momo=%s  MR_Chop=%s  MR=%s  (min DPS=%d)",
                     "ON" if _strategy_cfg.enable_momentum else "OFF",
                     "ON" if _strategy_cfg.enable_mr_chop else "OFF",
                     "ON" if _strategy_cfg.enable_mean_reversion else "OFF",
                     _strategy_cfg.min_dps_live)
        logger.info("  Approved: %d | Active: %d | Open positions: %d",
                     len(self.approved_symbols), len(self.active_symbols),
                     len(self.pos_mgr.positions))
        logger.info("  Balance: $%s | %s",
                     f"{_strategy_cfg.dummy_balance:,.0f}",
                     self.market_cond.summary())

        # Step 4: Start WS
        ws_task = asyncio.create_task(self.ws.start())
        await asyncio.sleep(3)

        # Step 5: Subscribe to kline for all active symbols
        if self.active_symbols:
            await self.ws.subscribe_kline(list(self.active_symbols))
            logger.info("Subscribed to %d symbols via WS", len(self.active_symbols))

        # Step 6: Run periodic loop
        periodic_task = asyncio.create_task(self._periodic_loop())

        try:
            await asyncio.gather(ws_task, periodic_task)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            self.ws.stop()

    # --- WS Kline Handler (called from WS thread) ---

    def _handle_kline(self, symbol: str, candle: dict):
        """On kline update — only process on new minute (candle close)."""
        ts = candle.get("timestamp", 0)
        ts_minute = (ts // 60000) * 60000

        last_minute = self._last_candle_ts.get(symbol, 0)
        if ts_minute <= last_minute:
            return  # same minute, skip

        self._last_candle_ts[symbol] = ts_minute

        # Append closed candle to cache
        if symbol in self.candle_cache:
            new_row = {
                "timestamp": pd.Timestamp(ts, unit="ms", tz="UTC"),
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle["volume"],
            }
            df = self.candle_cache[symbol]
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df]).drop_duplicates(
                subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            if len(df) > MR_WARMUP_BARS + 50:
                df = df.tail(MR_WARMUP_BARS + 50).reset_index(drop=True)
            self.candle_cache[symbol] = df

            # Queue for batch processing
            with self._batch_lock:
                self._pending_candle_symbols.add(symbol)

    # --- Periodic Loop (2s interval, same as live) ---

    async def _periodic_loop(self):
        while True:
            try:
                # Process batched candle symbols
                with self._batch_lock:
                    symbols_to_check = list(self._pending_candle_symbols)
                    self._pending_candle_symbols.clear()

                if symbols_to_check:
                    await self._batch_check_setups(symbols_to_check)

                # Periodic tasks
                self._periodic_tasks()

                # Process pending WS subscriptions
                if self._pending_subscriptions and self.ws._public_connected:
                    syms = list(self._pending_subscriptions)
                    self._pending_subscriptions.clear()
                    await self.ws.subscribe_kline(syms)
                    logger.info("Subscribed %d new symbols via WS", len(syms))

                # Manage pending limit orders
                self._manage_pending_limits()

            except Exception as e:
                logger.error("Periodic loop error: %s", e, exc_info=True)

            await asyncio.sleep(2)

    # --- Batch Setup Check (parallel depth fetch, same as live) ---

    async def _batch_check_setups(self, symbols: list[str]):
        """Check TP/SL for open positions, then scan for new setups."""

        # First: check TP/SL on all symbols with candle updates
        for sym in symbols:
            if sym not in self.candle_cache:
                continue
            full_df = self.candle_cache[sym]
            if len(full_df) < 2:
                continue
            # Use last CLOSED bar (exclude current incomplete)
            last_bar = full_df.iloc[-2]
            current_high = float(last_bar["high"])
            current_low = float(last_bar["low"])
            current_close = float(last_bar["close"])
            current_ts = str(last_bar["timestamp"])

            closed_trades = self.pos_mgr.check_tp_sl(
                sym, current_high, current_low, current_close, current_ts)
            for ct in closed_trades:
                # Fetch depth at close for snapshot comparison
                entry_snap = ct.get("entry_depth_snapshot")
                if entry_snap and ct.get("strategy") in ("zct_momo", "momentum"):
                    try:
                        close_depth = fetch_bitunix_depth(sym, limit="max")
                        if close_depth:
                            close_snap = compute_close_snapshot(
                                close_depth, ct.get("exit_price", 0),
                                entry_snap, ct["side"])
                            if close_snap:
                                ct.update(close_snap)
                    except Exception:
                        pass

                log_closed_trade(ct)
                outcome = ct["outcome"]
                pnl = ct.get("pnl_pct", 0)
                trailed = " [TRAILED]" if ct.get("sl_trailed") else ""
                logger.info("  CLOSED %s %s %s: %s @ %.6g -> %.6g  PnL=%.2f%%%s",
                            ct["strategy"].upper(), ct["side"].upper(), sym,
                            outcome, ct["entry"], ct["exit_price"], pnl, trailed)
                # Log depth snapshot comparison if available
                if entry_snap:
                    wall_status = ""
                    if ct.get("dc_wall_absorbed"):
                        wall_status = " WALL_ABSORBED(spoofed?)"
                    elif ct.get("dc_wall_still_exists"):
                        wall_status = " WALL_HELD"
                    logger.info("    DEPTH: imb=%.3f→%.3f block=%d walls $%s "
                                "protect=%d walls $%s%s",
                                entry_snap.get("d_imb_1", 0),
                                ct.get("dc_imb_1", 0),
                                entry_snap.get("d_block_n", 0),
                                f"{entry_snap.get('d_block_total_usd', 0):,.0f}",
                                entry_snap.get("d_protect_n", 0),
                                f"{entry_snap.get('d_protect_total_usd', 0):,.0f}",
                                wall_status)

        # Filter eligible symbols for new setups
        pending_syms = {lim["symbol"] for lim in self.pending_limits.values()}
        eligible = []
        for sym in symbols:
            if self.pos_mgr.has_any_position(sym):
                continue
            if sym in pending_syms:
                continue
            if sym not in self.candle_cache:
                continue
            if len(self.candle_cache[sym]) < 500:
                continue
            eligible.append(sym)

        if not eligible:
            return

        # Run strategies without depth — depth fetched only when setup triggers
        for sym in eligible:
            self._check_setup(sym, depth_data=None)

    # --- Strategy Check (mirrors live) ---

    def _check_setup(self, symbol: str, depth_data: dict):
        """Run strategy on a symbol — enter dummy positions."""
        if symbol not in self.candle_cache:
            return

        with self._lock:
            full_df = self.candle_cache[symbol]
            # Exclude last bar (current incomplete candle)
            df = full_df.iloc[:-1].reset_index(drop=True) if len(full_df) > 1 else full_df

            if len(df) < 500:
                return

            if self.pos_mgr.has_any_position(symbol):
                return

            try:
                found_setups = detect_setups(
                    df, symbol, _strategy_cfg, _mr_cfg, _momo_gate_cfg,
                    depth_data=depth_data)
            except Exception as e:
                logger.debug("Strategy error %s: %s", symbol, e)
                return

            for setup in found_setups:
                strat = setup["strategy"]
                side = setup["side"]

                # Skip disabled strategies
                if strat == "depth_bounce":
                    continue

                # Cooldown for depth/mr_chop
                if strat in ("depth", "depth_bounce", "mr_chop"):
                    cd_key = f"{symbol}_{strat}"
                    if time.time() - self.depth_cooldown.get(cd_key, 0) < 1800:
                        continue

                # DPS filter for momo
                if strat == "momentum" and setup.get("dps_total", 0) < 4:
                    continue

                # Market condition filter (disabled for zct_momo — collecting data)
                if strat != "zct_momo" and not self.market_cond.is_allowed(strat, side):
                    logger.debug("  SKIP %s %s %s: market=%s",
                                 strat, side, symbol, self.market_cond.summary())
                    continue

                # Fetch depth on-demand (only when setup passes all gates)
                depth_data = None
                if strat in ("zct_momo", "momentum"):
                    try:
                        depth_data = fetch_bitunix_depth(symbol, limit="50")
                    except Exception:
                        pass

                risk_pct = get_risk_pct(setup, _strategy_cfg)
                position_usd = _strategy_cfg.dummy_balance * risk_pct / 100

                setup["position_usd"] = round(position_usd, 2)
                setup["risk_pct"] = risk_pct
                setup["market_score"] = self.market_cond.score
                setup["bars_held"] = 0
                setup["mode"] = "dummy"

                # --- ZCT Momo: limit order at breakout level ---
                if strat == "zct_momo":
                    limit_key = f"{symbol}_zct_momo"
                    if limit_key not in self.pending_limits:
                        # Cancel at 75% of way to TP (matches backtest)
                        if side == "long":
                            cancel_price = setup["entry"] + (setup["tp"] - setup["entry"]) * 0.75
                        else:
                            cancel_price = setup["entry"] - (setup["entry"] - setup["tp"]) * 0.75

                        depth_alt = None
                        if depth_data:
                            try:
                                depth_alt = get_depth_alternative(
                                    symbol, depth_data, float(df.iloc[-1]["close"]),
                                    side, strat)
                            except Exception:
                                pass

                        # Depth snapshot at signal time
                        entry_depth_snap = None
                        if depth_data:
                            try:
                                entry_depth_snap = compute_depth_snapshot(
                                    depth_data, setup["entry"], setup["tp"],
                                    setup["sl"], side)
                            except Exception:
                                pass

                        self.pending_limits[limit_key] = {
                            **setup,
                            "placed_at": time.time(),
                            "limit_price": setup["entry"],
                            "limit_expiry_mins": 10,
                            "cancel_075r_price": round(cancel_price, 8),
                            "confirm_count": 0,
                            "confirmed": False,
                            "depth_alt": depth_alt,
                            "entry_depth_snapshot": entry_depth_snap,
                        }
                        wall_info = ""
                        dw = setup.get("depth_wall")
                        if dw and dw.get("walls", 0) > 0:
                            wall_info = f" wall=${dw['max_wall_usd']:,.0f}({dw['max_wall_strength']:.1f}x)"
                        logger.info("  PENDING ZCT_MOMO %s %s: level=%.6g tp=%.6g(%.2f%%) "
                                    "sl=%.6g(%.2f%%) RR=%.2f DPS=%d(%d%d%d) "
                                    "R²sm5=%.3f steps=%d eff15=%.2f%s "
                                    "(awaiting 1-bar confirm)",
                                    side.upper(), symbol,
                                    setup["entry"], setup["tp"], setup["tp_pct"],
                                    setup["sl"], setup["sl_pct"],
                                    setup["rr"], setup.get("dps_total", 0),
                                    setup.get("dps_dur", 0), setup.get("dps_app", 0),
                                    setup.get("dps_vol", 0),
                                    setup.get("r2_sm5", 0), setup.get("steps", 0),
                                    setup.get("eff_15m", 0), wall_info)
                    continue

                # --- MR Chop: limit order ---
                if strat == "mr_chop":
                    limit_key = f"{symbol}_mr_chop"
                    if limit_key not in self.pending_limits:
                        self.pending_limits[limit_key] = {
                            **setup,
                            "placed_at": time.time(),
                            "limit_price": setup["entry"],
                            "limit_expiry_mins": 3,
                        }
                        logger.info("  LIMIT MR_CHOP %s %s: price=%.6g tp=%.6g(%.2f%%) "
                                    "sl=%.6g(%.2f%%) RR=%.2f DPS=%d (3min expiry)",
                                    side.upper(), symbol,
                                    setup["entry"], setup["tp"], setup["tp_pct"],
                                    setup["sl"], setup["sl_pct"],
                                    setup["rr"], setup.get("dps_total", 0))
                    continue

                # --- Momentum: limit order with 2-bar confirm + 0.75R cancel ---
                if strat == "momentum":
                    limit_key = f"{symbol}_momentum"
                    if limit_key not in self.pending_limits:
                        r_dist = abs(setup["entry"] - setup["sl"])
                        if side == "long":
                            cancel_price = setup["entry"] + r_dist * 0.75
                        else:
                            cancel_price = setup["entry"] - r_dist * 0.75

                        depth_alt = None
                        if depth_data:
                            try:
                                depth_alt = get_depth_alternative(
                                    symbol, depth_data, float(df.iloc[-1]["close"]),
                                    side, strat)
                            except Exception:
                                pass

                        # Depth snapshot at signal time
                        entry_depth_snap = None
                        if depth_data:
                            try:
                                entry_depth_snap = compute_depth_snapshot(
                                    depth_data, setup["entry"], setup["tp"],
                                    setup["sl"], side)
                            except Exception:
                                pass

                        self.pending_limits[limit_key] = {
                            **setup,
                            "placed_at": time.time(),
                            "limit_price": setup["entry"],
                            "limit_expiry_mins": 10,
                            "cancel_075r_price": round(cancel_price, 8),
                            "confirm_count": 0,
                            "confirmed": False,
                            "depth_alt": depth_alt,
                            "entry_depth_snapshot": entry_depth_snap,
                        }
                        logger.info("  PENDING MOMO %s %s: entry=%.6g tp=%.6g(%.2f%%) "
                                    "sl=%.6g(%.2f%%) RR=%.2f DPS=%d(%d%d%d) "
                                    "(awaiting 2-bar confirm)",
                                    side.upper(), symbol,
                                    setup["entry"], setup["tp"], setup["tp_pct"],
                                    setup["sl"], setup["sl_pct"],
                                    setup["rr"], setup.get("dps_total", 0),
                                    setup.get("dps_dur", 0), setup.get("dps_app", 0),
                                    setup.get("dps_vol", 0))
                        if depth_alt:
                            logger.info("    DEPTH alt: tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) "
                                        "RR=%.2f wall=$%s",
                                        depth_alt["depth_tp"], depth_alt["depth_tp_pct"],
                                        depth_alt["depth_sl"], depth_alt["depth_sl_pct"],
                                        depth_alt["depth_rr"],
                                        f"{depth_alt['depth_sl_wall_usd']:,.0f}")
                    continue

                # --- Direct entry (depth, bouncy_ball, etc.) ---
                depth_alt = None
                if depth_data and strat != "depth":
                    try:
                        depth_alt = get_depth_alternative(
                            symbol, depth_data, float(df.iloc[-1]["close"]),
                            side, strat)
                    except Exception:
                        pass

                self.pos_mgr.open_position(setup)
                log_trade_entry(setup, depth_alt)

                if strat in ("depth", "depth_bounce", "mr_chop"):
                    self.depth_cooldown[f"{symbol}_{strat}"] = time.time()

                strat_labels = {"mean_reversion": "MR", "momentum": "MOMO",
                               "depth": "DEPTH", "mr_chop": "MR_CHOP"}
                strat_label = strat_labels.get(strat, strat.upper())
                logger.info("  NEW %s %s %s: entry=%.6g tp=%.6g(%.2f%%) "
                            "sl=%.6g(%.2f%%) RR=%.2f DPS=%d",
                            strat_label, side.upper(), symbol,
                            setup["entry"], setup["tp"], setup["tp_pct"],
                            setup["sl"], setup["sl_pct"],
                            setup["rr"], setup.get("dps_total", 0))
                if depth_alt:
                    logger.info("    DEPTH alt: tp=%.6g(%.2f%%) sl=%.6g(%.2f%%) "
                                "RR=%.2f wall=$%s",
                                depth_alt["depth_tp"], depth_alt["depth_tp_pct"],
                                depth_alt["depth_sl"], depth_alt["depth_sl_pct"],
                                depth_alt["depth_rr"],
                                f"{depth_alt['depth_sl_wall_usd']:,.0f}")
                break  # one setup per symbol

    # --- Pending Limit Order Management ---

    def _manage_pending_limits(self):
        """Manage pending limit orders: confirmation, fill, expiry, cancel."""
        if not self.pending_limits:
            return

        expired_keys = []
        for limit_key, lim in list(self.pending_limits.items()):
            sym = lim["symbol"]
            elapsed = time.time() - lim["placed_at"]
            limit_price = lim["limit_price"]
            side = lim["side"]
            strat = lim.get("strategy", "mr_chop")

            if sym not in self.candle_cache or len(self.candle_cache[sym]) < 2:
                continue
            df = self.candle_cache[sym]
            last_bar = df.iloc[-2]  # closed bar
            last_high = float(last_bar["high"])
            last_low = float(last_bar["low"])
            last_close = float(last_bar["close"])

            # --- Momentum / ZCT Momo: confirm → 0.75R cancel → limit fill ---
            if strat in ("momentum", "zct_momo"):
                # ZCT momo uses 1-bar confirm, momentum uses 2-bar
                confirm_needed = 1 if strat == "zct_momo" else 2
                if not lim.get("confirmed", False):
                    if side == "long":
                        onside = last_close > limit_price
                    else:
                        onside = last_close < limit_price
                    if onside:
                        lim["confirm_count"] = lim.get("confirm_count", 0) + 1
                    else:
                        logger.info("  MISSED %s %s %s: confirm failed bar %d",
                                    strat.upper(), side.upper(), sym,
                                    lim.get("confirm_count", 0) + 1)
                        expired_keys.append(limit_key)
                        continue

                    if lim["confirm_count"] >= confirm_needed:
                        lim["confirmed"] = True
                        lim["confirmed_at"] = time.time()
                        logger.info("  CONFIRMED %s %s %s: %d-bar passed, limit=%.6g",
                                    strat.upper(), side.upper(), sym,
                                    confirm_needed, limit_price)
                    continue

                # 0.75R cancel
                cancel_price = lim.get("cancel_075r_price", 0)
                if side == "long" and last_high >= cancel_price:
                    logger.info("  MISSED %s %s %s: 0.75R onside (cancel=%.6g)",
                                strat.upper(), side.upper(), sym, cancel_price)
                    expired_keys.append(limit_key)
                    continue
                if side == "short" and last_low <= cancel_price:
                    logger.info("  MISSED %s %s %s: 0.75R onside (cancel=%.6g)",
                                strat.upper(), side.upper(), sym, cancel_price)
                    expired_keys.append(limit_key)
                    continue

                # Check fill
                filled = False
                if side == "long" and last_low <= limit_price:
                    filled = True
                elif side == "short" and last_high >= limit_price:
                    filled = True

                if filled:
                    depth_alt = lim.pop("depth_alt", None)
                    self.pos_mgr.open_position(lim)
                    log_trade_entry(lim, depth_alt)
                    logger.info("  FILLED %s %s %s: fill=%.6g waited=%.0fs DPS=%d",
                                strat.upper(), side.upper(), sym, limit_price,
                                elapsed, lim.get("dps_total", 0))
                    expired_keys.append(limit_key)
                    continue

                # Expiry
                if elapsed >= lim.get("limit_expiry_mins", 10) * 60:
                    logger.info("  MISSED %s %s %s: expired after %.0fs",
                                strat.upper(), side.upper(), sym, elapsed)
                    expired_keys.append(limit_key)
                continue

            # --- MR Chop: simple fill check ---
            filled = False
            if side == "short" and last_high >= limit_price:
                filled = True
            elif side == "long" and last_low <= limit_price:
                filled = True

            if filled:
                self.pos_mgr.open_position(lim)
                log_trade_entry(lim)
                logger.info("  FILLED MR_CHOP %s %s: fill=%.6g waited=%.0fs",
                            side.upper(), sym, limit_price, elapsed)
                expired_keys.append(limit_key)
                continue

            if elapsed >= lim.get("limit_expiry_mins", 3) * 60:
                logger.info("  MISSED MR_CHOP %s %s: expired after %.0fs",
                            side.upper(), sym, elapsed)
                expired_keys.append(limit_key)

        for k in expired_keys:
            self.pending_limits.pop(k, None)

    # --- Periodic Tasks ---

    def _periodic_tasks(self):
        now = time.time()

        # Refresh active coins every 2 min
        if now - self._last_coin_refresh > self._coin_refresh_interval:
            self._refresh_active_coins()
            self._last_coin_refresh = now

        # Refresh market condition every 2 hours
        if now - self._last_mkt_refresh > self._mkt_refresh_interval:
            self._refresh_market_condition()
            self._last_mkt_refresh = now

    def _refresh_active_coins(self):
        coins = fetch_orion_active_coins(min_vol_5m=self.min_vol, top_n=self.top_n)
        if not coins:
            return

        coin_symbols = {c["symbol"] for c in coins}

        if self.approved_symbols:
            for c in coins:
                sym = c["symbol"]
                if sym not in self.approved_symbols:
                    log_new_coin(sym, c.get("volume_5m", 0), c.get("trades_5m", 0))
            coin_symbols = {s for s in coin_symbols if s in self.approved_symbols}

        open_syms = self.pos_mgr.get_all_symbols()
        new_symbols = (coin_symbols | open_syms) - self.active_symbols

        if new_symbols:
            self.active_symbols.update(new_symbols)
            # Backfill candles for new symbols (REST)
            for sym in new_symbols:
                if sym not in self.candle_cache:
                    df = fetch_klines_paginated(sym, n_bars=MR_WARMUP_BARS)
                    if df is not None and len(df) >= 200:
                        self.candle_cache[sym] = df
            # Queue WS subscriptions
            self._pending_subscriptions.update(new_symbols)
            logger.info("Added %d symbols (total active: %d)",
                        len(new_symbols), len(self.active_symbols))

    def _load_initial_candles(self):
        to_load = [s for s in sorted(self.active_symbols) if s not in self.candle_cache]
        logger.info("Loading initial candles for %d symbols...", len(to_load))
        loaded = 0
        failed = 0
        for sym in to_load:
            try:
                df = fetch_klines_paginated(sym, n_bars=MR_WARMUP_BARS)
                if df is not None and len(df) >= 200:
                    self.candle_cache[sym] = df
                    loaded += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
            time.sleep(0.1)
            if (loaded + failed) % 20 == 0:
                logger.info("  Progress: %d/%d loaded, %d failed",
                             loaded, len(to_load), failed)
        logger.info("Initial candles: %d loaded, %d failed, %d total cached",
                     loaded, failed, len(self.candle_cache))

    def _refresh_market_condition(self):
        btc_sym = "BTCUSDT"
        if btc_sym in self.candle_cache:
            self.market_cond.update_btc(self.candle_cache[btc_sym])
        if len(self.candle_cache) > 10:
            self.market_cond.update_breadth(self.candle_cache)
        logger.info("  %s (updated)", self.market_cond.summary())


# ═══════════════════════════════════════════════════════════════════════════
# Status Display
# ═══════════════════════════════════════════════════════════════════════════

def show_status():
    pos_mgr = PositionManager()
    print(f"\n{'=' * 60}")
    print(f"DUMMY TRADER STATUS")
    print(f"{'=' * 60}")
    print(f"Open positions: {len(pos_mgr.positions)}")
    for p in pos_mgr.positions:
        print(f"  {p['strategy']:15s} {p['side']:5s} {p['symbol']:15s} "
              f"entry={p['entry']:.6g}  tp={p['tp']:.6g}  sl={p['sl']:.6g}  "
              f"bars={p.get('bars_held', 0)}")

    if CLOSED_TRADES_LOG.exists():
        ct = pd.read_csv(str(CLOSED_TRADES_LOG), on_bad_lines="skip")
        total_trades = len(ct)
        tp_count = (ct["outcome"].isin(["TP"])).sum()
        sl_count = (ct["outcome"].isin(["SL"])).sum()
        trail_count = (ct["outcome"].isin(["TRAIL_SL"])).sum()
        total_pnl = ct["pnl_pct"].sum()
        wr = tp_count / (tp_count + sl_count + trail_count) * 100 if (tp_count + sl_count + trail_count) > 0 else 0

        print(f"\nClosed trades: {total_trades}")
        print(f"  TP: {tp_count}  SL: {sl_count}  TRAIL_SL: {trail_count}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  Total PnL: {total_pnl:+.2f}%")
        if total_trades > 0:
            print(f"  Avg PnL/trade: {total_pnl / total_trades:+.3f}%")

        for strat in ct["strategy"].unique():
            sub = ct[ct["strategy"] == strat]
            if len(sub) == 0:
                continue
            tp_s = sub["outcome"].isin(["TP"]).sum()
            sl_s = sub["outcome"].isin(["SL", "TRAIL_SL"]).sum()
            wr_s = tp_s / (tp_s + sl_s) * 100 if (tp_s + sl_s) > 0 else 0
            pnl_s = sub["pnl_pct"].sum()
            print(f"\n  {strat.upper()}: {len(sub)} trades | "
                  f"TP={tp_s} SL={sl_s} WR={wr_s:.1f}% PnL={pnl_s:+.2f}%")

    if DEPTH_COMPARISON_LOG.exists():
        dc = pd.read_csv(str(DEPTH_COMPARISON_LOG))
        if len(dc) > 0:
            print(f"\nDepth Comparison ({len(dc)} trades):")
            print(f"  Avg Strategy RR: {dc['strat_rr'].mean():.2f}")
            print(f"  Avg Depth RR:    {dc['depth_rr'].mean():.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="WS-Based Live Dummy Trader")
    parser.add_argument("--min-vol", type=float, default=200_000,
                        help="Min 5-minute volume (default: 200000)")
    parser.add_argument("--top-n", type=int, default=60,
                        help="Max coins to scan (default: 60)")
    parser.add_argument("--status", action="store_true",
                        help="Show current positions and stats, then exit")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    global _approved_symbols
    _approved_symbols = load_approved_symbols()

    trader = WSDummyTrader(min_vol=args.min_vol, top_n=args.top_n)

    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        logger.info("Stopped by user")


if __name__ == "__main__":
    main()
