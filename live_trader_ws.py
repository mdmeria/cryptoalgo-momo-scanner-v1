#!/usr/bin/env python3
"""
WebSocket-Based Live Trader — Near-zero latency execution.

Architecture:
  - WebSocket receives real-time 1m candle + depth updates
  - On each new candle close, immediately runs strategy gates
  - If setup triggers, places order within milliseconds
  - Private WS receives instant fill/TP/SL notifications
  - REST API used only for order placement and initial data load

Usage:
  python live_trader_ws.py [--dry-run] [--once]
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
_log_dir = Path("datasets/live/live_trader")
_log_dir.mkdir(parents=True, exist_ok=True)

_console_handler = logging.StreamHandler(stream=sys.stderr)
_console_handler.setFormatter(_log_fmt)
_file_handler = logging.FileHandler(
    _log_dir / "trader_log.txt", encoding="utf-8")
_file_handler.setFormatter(_log_fmt)

logger = logging.getLogger("live_ws")
logger.setLevel(logging.INFO)
logger.addHandler(_console_handler)
logger.addHandler(_file_handler)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from ws_client import BitunixWS
from live_trader import (
    BitunixClient,
    LivePositionTracker,
    DailyPnLTracker,
    load_config,
    log_trade,
    check_kill_switch,
    calculate_qty,
    TRADER_DIR,
)
from live_dummy_trader import (
    fetch_klines_paginated,
    load_approved_symbols,
    MR_WARMUP_BARS,
)
from live_data_collector import (
    fetch_orion_active_coins,
    LIVE_DIR,
)
from strategies import (
    StrategyConfig,
    MRSettings,
    MomoGateSettings,
    detect_setups,
    get_risk_pct,
    MarketCondition,
)

_strategy_cfg = StrategyConfig.from_json()
_mr_cfg = MRSettings()
_momo_gate_cfg = MomoGateSettings.from_json("momo_gate_settings.json")


class WSTrader:
    """WebSocket-based live trader with near-zero latency."""

    def __init__(self, config: dict, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        trading_cfg = config["trading"]

        # REST client for orders
        self.client = BitunixClient(
            config["api_key"], config["api_secret"], dry_run=dry_run)

        # WebSocket client for real-time data
        self.ws = BitunixWS(config["api_key"], config["api_secret"])
        self.ws.on_kline = self._handle_kline
        self.ws.on_depth = self._handle_depth
        self.ws.on_position = self._handle_position
        self.ws.on_order = self._handle_order

        # Position tracking
        self.pos_tracker = LivePositionTracker(self.client)
        self.pnl_tracker = DailyPnLTracker(trading_cfg["max_daily_loss_pct"])
        self.market_cond = MarketCondition()

        # Data
        self.candle_cache: dict[str, pd.DataFrame] = {}
        self.approved_symbols = load_approved_symbols()
        self.pairs_info = {}
        self.depth_cooldown: dict[str, float] = {}
        self.strat_modes = config.get("strategies", {})

        # Load persistent cooldown
        cooldown_file = TRADER_DIR / "depth_cooldown.json"
        if cooldown_file.exists():
            with open(cooldown_file) as f:
                self.depth_cooldown = json.load(f)

        # Active symbols being tracked
        self.active_symbols: set[str] = set()

        # Symbols with margin/leverage already configured
        self._margin_configured: set[str] = set()

        # Lock for thread safety
        self._lock = threading.Lock()

        # Track last processed candle timestamp per symbol
        self._last_candle_ts: dict[str, int] = {}

        # Batch candle processing — collect symbols, process together
        self._pending_candle_symbols: set[str] = set()
        self._batch_lock = threading.Lock()

        # Pending limit orders: {orderId: {symbol, side, tp, sl, qty, setup, placed_at}}
        self._pending_limit_orders: dict[str, dict] = {}

        # Depth watchlist: {symbol: {setup details, created_at}}
        self._depth_watchlist: dict[str, dict] = {}

        # Pending WS subscriptions (sync -> async bridge)
        self._pending_subscriptions: set[str] = set()

        # Refresh intervals
        self._last_coin_refresh = 0
        self._coin_refresh_interval = 120  # 2 minutes
        self._last_mkt_refresh = 0
        self._mkt_refresh_interval = 7200  # 2 hours

    async def start(self):
        """Start the WebSocket trader (async)."""
        TRADER_DIR.mkdir(parents=True, exist_ok=True)

        # Load trading pairs info (REST — sync)
        self.pairs_info = self.client.get_trading_pairs()
        logger.info("Trading pairs loaded: %d", len(self.pairs_info))

        # Load active coins + candle history (REST — sync)
        self._refresh_active_coins()
        self._load_initial_candles()

        # Pre-configure margin (REST — sync)
        self._preconfigure_margin(self.active_symbols)

        # Initial market condition
        self._refresh_market_condition()

        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info("WebSocket Trader started [%s]", mode)
        logger.info("  Strategies: MR=%s StrictMR=%s Momo=%s Depth=%s DepthBounce=%s",
                     self.strat_modes.get("mean_reversion", "off").upper(),
                     self.strat_modes.get("strict_mr", "off").upper(),
                     self.strat_modes.get("momentum", "off").upper(),
                     self.strat_modes.get("depth", "off").upper(),
                     self.strat_modes.get("depth_bounce", "off").upper())
        logger.info("  Approved: %d | Active: %d | Positions: %d",
                     len(self.approved_symbols), len(self.active_symbols),
                     len(self.pos_tracker.local_positions))
        logger.info("  %s", self.market_cond.summary())

        # Set up WS callbacks
        self.ws.on_kline = self._handle_kline
        self.ws.on_depth = self._handle_depth
        self.ws.on_position = self._handle_position
        self.ws.on_order = self._handle_order

        # Start WS + periodic tasks concurrently
        ws_task = asyncio.create_task(self.ws.start())

        # Wait for WS to connect
        await asyncio.sleep(3)

        # Subscribe to kline only (depth fetched via REST when needed)
        await self.ws.subscribe_kline(list(self.active_symbols))

        # Run periodic tasks alongside WS
        periodic_task = asyncio.create_task(self._periodic_loop())

        try:
            await asyncio.gather(ws_task, periodic_task)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            self.ws.stop()

    async def _periodic_loop(self):
        """Run periodic tasks + batch candle processing."""
        while True:
            try:
                # Process batched candle symbols (parallel depth fetch)
                with self._batch_lock:
                    symbols_to_check = list(self._pending_candle_symbols)
                    self._pending_candle_symbols.clear()

                if symbols_to_check:
                    await self._batch_check_setups(symbols_to_check)

                # Periodic tasks (sync — runs every loop but cheap)
                self._periodic_tasks()

                # Process pending WS subscriptions (kline only)
                if self._pending_subscriptions and self.ws.is_connected:
                    syms = list(self._pending_subscriptions)
                    self._pending_subscriptions.clear()
                    await self.ws.subscribe_kline(syms)
                    logger.info("Subscribed %d new symbols via WS", len(syms))

                # Manage pending limit orders (3 min expiry)
                self._manage_pending_orders()
            except Exception as e:
                logger.error("Periodic task error: %s", e)
            await asyncio.sleep(2)  # check every 2 seconds for batched candles

    def _manage_pending_orders(self):
        """Check pending limit orders: cancel after 3 min, detect fills."""
        if not self._pending_limit_orders:
            return

        now = time.time()
        to_remove = []

        # Log pending status
        for oid, inf in self._pending_limit_orders.items():
            wait = now - inf["placed_at"]
            logger.info("    LIMIT PENDING: %s %s waiting %.0fs (price=%s)",
                        inf["symbol"], inf["side"], wait, inf["entry_price"])

        for order_id, info in self._pending_limit_orders.items():
            symbol = info["symbol"]
            elapsed = now - info["placed_at"]

            # Check if filled — look for position on exchange
            positions = self.client.get_positions(symbol)
            has_position = len(positions) > 0

            if has_position:
                # Order filled! TP/SL already attached to order
                pos = positions[0]
                pos_id = pos.get("positionId")
                fill_price = float(pos.get("avgOpenPrice") or 0)
                filled_qty = float(pos.get("qty") or 0)

                setup = info["setup"]
                setup["order_id"] = order_id
                setup["exchange_position_id"] = pos_id
                setup["qty"] = filled_qty
                setup["actual_fill"] = fill_price
                setup["risk_pct"] = info["risk_pct"]
                setup["market_score"] = self.market_cond.score
                setup["bars_held"] = 0
                setup["mode"] = "live"
                setup["slippage_pct"] = 0.0  # limit order = zero slippage

                setup["fill_time_sec"] = round(elapsed, 1)
                self.pos_tracker.add_position(setup)
                log_trade(setup, "ENTRY_LIVE_LIMIT")

                logger.info("    LIMIT FILLED: %s fill=%.6g posId=%s filled_in=%.0fs (TP/SL on order)",
                            symbol, fill_price, pos_id, elapsed)

                to_remove.append(order_id)

            elif elapsed >= 180:  # 3 minutes
                # Cancel the order
                cancel_result = self.client._post(
                    "/api/v1/futures/trade/cancel_orders",
                    {"symbol": symbol, "orderList": [{"orderId": order_id}]})
                cancel_code = cancel_result.get("code", -1)
                if cancel_code == 0:
                    logger.info("    LIMIT CANCELLED (%.0fs): %s id=%s — price didn't reach our level",
                                elapsed, symbol, order_id)
                else:
                    logger.info("    LIMIT CANCEL response: %s %s (may have filled)",
                                symbol, cancel_result.get("msg"))

                # Log as missed trade
                setup = info["setup"]
                setup["order_id"] = order_id
                setup["risk_pct"] = info["risk_pct"]
                setup["market_score"] = self.market_cond.score
                setup["mode"] = "live"
                setup["wait_time_sec"] = round(elapsed, 1)
                log_trade(setup, "MISSED_LIMIT")
                logger.info("    MISSED: %s %s entry=%s tp=%s sl=%s RR=%.2f waited=%.0fs",
                            symbol, info["side"], info["entry_price"],
                            info["tp_price"], info["sl_price"],
                            setup.get("rr", 0), elapsed)

                to_remove.append(order_id)

        for oid in to_remove:
            self._pending_limit_orders.pop(oid, None)

    async def _batch_check_setups(self, symbols: list[str]):
        """Fetch depth for multiple symbols in parallel, then run strategies."""
        from live_data_collector import fetch_bitunix_depth

        # Filter: skip symbols with existing positions or at max capacity
        trading_cfg = self.config["trading"]
        live_positions = [p for p in self.pos_tracker.local_positions if p.get("mode") != "dummy"]
        # Symbols with pending limit orders
        pending_syms = {info["symbol"] for info in self._pending_limit_orders.values()}

        eligible = []
        for sym in symbols:
            if sym in self.pos_tracker.get_all_symbols():
                continue
            if sym in pending_syms:
                continue
            if len(live_positions) >= trading_cfg["max_positions"]:
                break
            if sym not in self.candle_cache:
                continue
            eligible.append(sym)

        if not eligible:
            return

        # Throttled parallel depth fetch (max 8 concurrent to stay under rate limit)
        semaphore = asyncio.Semaphore(8)
        loop = asyncio.get_event_loop()
        depth_results = {}

        async def fetch_one(sym):
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, fetch_bitunix_depth, sym, "max"),
                        timeout=5)
                    depth_results[sym] = result
                except Exception:
                    depth_results[sym] = None

        await asyncio.gather(*[fetch_one(sym) for sym in eligible])

        logger.info("Batch: %d candles, %d eligible, %d depth fetched in parallel",
                     len(symbols), len(eligible),
                     sum(1 for d in depth_results.values() if d))

        # Run strategies for each symbol
        for sym in eligible:
            depth_data = depth_results.get(sym)
            self._check_setup_with_depth(sym, depth_data)

        # Also check watchlist for wall touches on this candle batch
        self._check_watchlist(symbols)

    def _check_setup_with_depth(self, symbol: str, depth_data: dict):
        """Run strategy on a symbol with pre-fetched depth.
        Uses closed bars only (excludes current incomplete candle)."""
        if symbol not in self.candle_cache:
            return

        with self._lock:
            if check_kill_switch(self.config):
                return
            if self.pnl_tracker.is_limit_hit():
                return

            trading_cfg = self.config["trading"]
            # Exclude last bar (current incomplete candle) — strategy needs closed bars
            full_df = self.candle_cache[symbol]
            df = full_df.iloc[:-1].reset_index(drop=True) if len(full_df) > 1 else full_df

            if len(df) < 500:
                return

            if symbol in self.pos_tracker.get_all_symbols():
                return

            live_count = sum(1 for p in self.pos_tracker.local_positions if p.get("mode") != "dummy")
            if live_count >= trading_cfg["max_positions"]:
                return

            # Run strategies
            try:
                found_setups = detect_setups(
                    df, symbol, _strategy_cfg, _mr_cfg, _momo_gate_cfg,
                    depth_data=depth_data)
            except Exception as e:
                logger.debug("Strategy error %s: %s", symbol, e)
                return

            # --- Depth: add to watchlist instead of entering ---
            if depth_data and symbol not in self._depth_watchlist:
                from strategy_depth import find_depth_watchlist_setup, DepthWatchlistSettings
                wl_cfg = DepthWatchlistSettings()
                wl_result = find_depth_watchlist_setup(depth_data, float(df.iloc[-1]["close"]), wl_cfg)
                if wl_result["passed"]:
                    strat_mode = self.strat_modes.get("depth", "off")
                    if strat_mode != "off":
                        side = wl_result["side"]
                        if self.market_cond.is_allowed("depth", side):
                            cd_key = f"{symbol}_depth"
                            if time.time() - self.depth_cooldown.get(cd_key, 0) >= 1800:
                                self._depth_watchlist[symbol] = {
                                    **wl_result,
                                    "created_at": time.time(),
                                    "strat_mode": strat_mode,
                                }
                                logger.info("  WATCHLIST ADD %s %s: wall=%.6g dist=%.2f%% tp=%.6g sl=%.6g RR=%.2f (wait %dmin)",
                                            side.upper(), symbol,
                                            wl_result["entry_wall_price"],
                                            wl_result["distance_pct"],
                                            wl_result["tp"], wl_result["sl"],
                                            wl_result["rr"],
                                            wl_cfg.max_watch_minutes)

            # --- MR / Momo: direct entry (unchanged) ---
            for setup in found_setups:
                strat = setup["strategy"]

                # Skip depth — handled by watchlist above
                if strat in ("depth", "depth_bounce"):
                    continue

                strat_mode = self.strat_modes.get(strat, "off")
                if strat_mode == "off":
                    continue

                side = setup["side"]
                if strat == "momentum" and setup.get("dps_total", 0) < 4:
                    continue
                if not self.market_cond.is_allowed(strat, side):
                    continue

                balance_data = self.client.get_balance(trading_cfg["margin_coin"])
                available = float(balance_data.get("available", 0))
                if available <= 0:
                    continue

                risk_pct = get_risk_pct(setup, _strategy_cfg)
                entry_price = setup["entry"]
                sl_pct = setup["sl_pct"]

                qty = calculate_qty(
                    symbol, entry_price, risk_pct, sl_pct,
                    available, trading_cfg["leverage"],
                    self.pairs_info)

                if float(qty) <= 0:
                    continue

                order_side = "BUY" if side == "long" else "SELL"
                tp_price = str(round(setup["tp"], 8))
                sl_price = str(round(setup["sl"], 8))

                mode_tag = f"[{strat_mode.upper()}]"
                logger.info("  %s %s %s %s: entry=~%.6g tp=%s sl=%s RR=%.2f DPS=%s",
                            mode_tag, strat.upper(), side.upper(), symbol,
                            entry_price, tp_price, sl_price,
                            setup["rr"], setup.get("dps_total", "?"))

                if strat_mode == "dummy":
                    setup["order_id"] = f"dummy_{int(time.time())}"
                    setup["qty"] = qty
                    setup["risk_pct"] = risk_pct
                    setup["market_score"] = self.market_cond.score
                    setup["bars_held"] = 0
                    setup["mode"] = "dummy"
                    self.pos_tracker.add_position(setup)
                    log_trade(setup, "ENTRY_DUMMY")

                elif strat_mode == "live":
                    already_configured = symbol in self._margin_configured
                    if not already_configured:
                        self.client.ensure_margin_and_leverage(
                            symbol, trading_cfg["leverage"], trading_cfg["margin_mode"])
                        self._margin_configured.add(symbol)

                    if strat in ("depth", "depth_bounce"):
                        # Depth handled by watchlist — should not reach here
                        continue

                    else:
                        # MARKET order for Momo — immediate fill with TP/SL
                        result = self.client.place_order_and_verify(
                            symbol=symbol, side=order_side, qty=qty,
                            tp_price=tp_price, sl_price=sl_price,
                            leverage=trading_cfg["leverage"],
                            margin_mode=trading_cfg["margin_mode"],
                            tp_sl_type=trading_cfg["tp_sl_type"],
                            skip_margin_setup=True)

                        if result.get("success"):
                            setup["order_id"] = result["order_id"]
                            setup["exchange_position_id"] = result["position_id"]
                            setup["qty"] = result.get("filled_qty", qty)
                            setup["actual_fill"] = result["actual_fill"]
                            setup["risk_pct"] = risk_pct
                            setup["market_score"] = self.market_cond.score
                            setup["bars_held"] = 0
                            setup["mode"] = "live"
                            self.pos_tracker.add_position(setup)
                            log_trade(setup, "ENTRY_LIVE")
                            logger.info("    FILLED: %s fill=%.6g posId=%s",
                                        symbol, result["actual_fill"],
                                        result["position_id"] or "?")
                        else:
                            logger.error("    FAILED: %s — %s",
                                         symbol, result.get("error", "Unknown"))

                if strat in ("depth", "depth_bounce"):
                    self.depth_cooldown[f"{symbol}_{strat}"] = time.time()
                    with open(TRADER_DIR / "depth_cooldown.json", "w") as f:
                        json.dump(self.depth_cooldown, f)

                break

    def _check_watchlist(self, candle_symbols: list[str]):
        """Check watchlist for wall touches, expirations, and invalidations."""
        from strategy_depth import check_wall_touch
        from live_data_collector import fetch_bitunix_depth

        if not self._depth_watchlist:
            return

        now = time.time()
        to_remove = []
        trading_cfg = self.config["trading"]

        for symbol, wl in self._depth_watchlist.items():
            elapsed_min = (now - wl["created_at"]) / 60

            # Log watchlist status with real-time distance
            live_dist = wl["distance_pct"]  # default to original
            if symbol in self.candle_cache and len(self.candle_cache[symbol]) >= 2:
                cur_price = float(self.candle_cache[symbol].iloc[-2]["close"])
                if cur_price > 0:
                    if wl["side"] == "long":
                        live_dist = (cur_price - wl["entry_wall_price"]) / cur_price * 100
                    else:
                        live_dist = (wl["entry_wall_price"] - cur_price) / cur_price * 100
            logger.info("  WATCH %s %s: wall=%.6g dist=%.2f%% waiting=%.0fmin",
                        wl["side"].upper(), symbol,
                        wl["entry_wall_price"], live_dist, elapsed_min)

            # Expired?
            max_watch = wl.get("max_watch_minutes", 30)
            if elapsed_min >= max_watch:
                logger.info("  WATCH EXPIRED: %s (%.0fmin) — wall=%.6g never reached",
                            symbol, elapsed_min, wl["entry_wall_price"])
                log_trade({"symbol": symbol, "strategy": "depth", "side": wl["side"],
                           "entry": wl["entry_price"], "tp": wl["tp"], "sl": wl["sl"],
                           "rr": wl["rr"], "entry_wall_price": wl["entry_wall_price"],
                           "wait_time_sec": round(elapsed_min * 60)},
                          "WATCHLIST_EXPIRED")
                to_remove.append(symbol)
                continue

            # Has candle data updated for this symbol?
            if symbol not in candle_symbols:
                continue
            if symbol not in self.candle_cache:
                continue

            full_df = self.candle_cache[symbol]
            if len(full_df) < 2:
                continue

            # Use last CLOSED bar
            last_bar = full_df.iloc[-2]
            candle = {
                "high": float(last_bar["high"]),
                "low": float(last_bar["low"]),
                "close": float(last_bar["close"]),
            }

            # Check wall touch
            status = check_wall_touch(candle, wl["entry_wall_price"],
                                       wl["side"], wl["invalidation_price"])

            if status == "broken":
                logger.info("  WATCH BROKEN: %s — candle closed beyond wall %.6g (close=%.6g)",
                            symbol, wl["entry_wall_price"], candle["close"])
                log_trade({"symbol": symbol, "strategy": "depth", "side": wl["side"],
                           "entry": wl["entry_price"], "tp": wl["tp"], "sl": wl["sl"],
                           "rr": wl["rr"], "entry_wall_price": wl["entry_wall_price"],
                           "close_price": candle["close"],
                           "wait_time_sec": round(elapsed_min * 60)},
                          "WATCHLIST_BROKEN")
                to_remove.append(symbol)

            elif status == "confirmed":
                # Wall touched and held! Re-check wall strength before entering
                logger.info("  WATCH CONFIRMED: %s — wick touched wall %.6g, close=%.6g (wall held!)",
                            symbol, wl["entry_wall_price"], candle["close"])

                # Re-fetch depth to verify wall still exists
                fresh_depth = fetch_bitunix_depth(symbol, limit="max")
                wall_still_valid = True
                if fresh_depth:
                    from live_data_collector import analyze_depth
                    analysis = analyze_depth(fresh_depth, candle["close"])
                    # Check if wall is still there
                    if wl["side"] == "long":
                        walls = analysis.get("tp_sl_walls_support", [])
                    else:
                        walls = analysis.get("tp_sl_walls_resistance", [])
                    # Find our entry wall
                    found = False
                    for w in walls:
                        if abs(w["price"] - wl["entry_wall_price"]) / wl["entry_wall_price"] < 0.005:
                            found = True
                            logger.info("    Wall verified: $%.0f (%.1fx) — still there",
                                        w["price"] * w["qty"], w.get("strength", 0))
                            break
                    if not found:
                        wall_still_valid = False
                        logger.info("    Wall GONE at %.6g — skipping entry", wl["entry_wall_price"])

                if not wall_still_valid:
                    log_trade({"symbol": symbol, "strategy": "depth", "side": wl["side"],
                               "entry": wl["entry_price"], "tp": wl["tp"], "sl": wl["sl"],
                               "rr": wl["rr"], "entry_wall_price": wl["entry_wall_price"]},
                              "WATCHLIST_WALL_GONE")
                    to_remove.append(symbol)
                    continue

                # Wall confirmed + still exists → PLACE ORDER
                with self._lock:
                    live_count = sum(1 for p in self.pos_tracker.local_positions if p.get("mode") != "dummy")
                    if live_count >= trading_cfg["max_positions"]:
                        logger.info("    Max positions reached — skipping %s", symbol)
                        to_remove.append(symbol)
                        continue

                    if symbol in self.pos_tracker.get_all_symbols():
                        to_remove.append(symbol)
                        continue

                    # Calculate qty
                    balance_data = self.client.get_balance(trading_cfg["margin_coin"])
                    available = float(balance_data.get("available", 0))
                    if available <= 0:
                        continue

                    risk_pct = get_risk_pct({"strategy": "depth"}, _strategy_cfg)
                    entry_price = wl["entry_price"]
                    sl_pct = wl["sl_pct"]

                    qty = calculate_qty(symbol, entry_price, risk_pct, sl_pct,
                                        available, trading_cfg["leverage"], self.pairs_info)
                    if float(qty) <= 0:
                        continue

                    order_side = "BUY" if wl["side"] == "long" else "SELL"
                    tp_price = str(round(wl["tp"], 8))
                    sl_price = str(round(wl["sl"], 8))
                    limit_price = str(round(entry_price, 8))

                    strat_mode = wl.get("strat_mode", "live")

                    if strat_mode == "dummy":
                        setup = {"symbol": symbol, "strategy": "depth_wall",
                                 "side": wl["side"], "entry": entry_price,
                                 "tp": wl["tp"], "sl": wl["sl"],
                                 "sl_pct": sl_pct, "tp_pct": wl["tp_pct"], "rr": wl["rr"],
                                 "entry_wall_price": wl["entry_wall_price"],
                                 "entry_wall_usd": wl["entry_wall_usd"],
                                 "entry_wall_strength": wl["entry_wall_strength"],
                                 "imbalance_1pct": wl["imbalance_1pct"],
                                 "order_id": f"dummy_{int(time.time())}",
                                 "qty": qty, "risk_pct": risk_pct,
                                 "market_score": self.market_cond.score,
                                 "bars_held": 0, "mode": "dummy",
                                 "timestamp": str(pd.Timestamp.now(tz="UTC")),
                                 "wait_time_sec": round(elapsed_min * 60)}
                        self.pos_tracker.add_position(setup)
                        log_trade(setup, "ENTRY_WALL_DUMMY")
                        logger.info("    WALL ENTRY (dummy): %s %s at %.6g RR=%.2f",
                                    wl["side"], symbol, entry_price, wl["rr"])

                    elif strat_mode == "live":
                        # Ensure margin
                        if symbol not in self._margin_configured:
                            self.client.ensure_margin_and_leverage(
                                symbol, trading_cfg["leverage"], trading_cfg["margin_mode"])
                            self._margin_configured.add(symbol)

                        # Place limit at wall price with TP/SL
                        result = self.client.place_order(
                            symbol=symbol, side=order_side, qty=qty,
                            order_type="LIMIT", price=limit_price,
                            tp_price=tp_price, sl_price=sl_price)

                        if result.get("code") == 0:
                            order_id = result["data"]["orderId"]
                            self._pending_limit_orders[order_id] = {
                                "symbol": symbol, "side": wl["side"],
                                "order_side": order_side,
                                "entry_price": entry_price,
                                "tp_price": tp_price, "sl_price": sl_price,
                                "qty": qty,
                                "setup": {"symbol": symbol, "strategy": "depth_wall",
                                          "side": wl["side"], "entry": entry_price,
                                          "tp": wl["tp"], "sl": wl["sl"],
                                          "sl_pct": sl_pct, "tp_pct": wl["tp_pct"],
                                          "rr": wl["rr"],
                                          "entry_wall_price": wl["entry_wall_price"],
                                          "entry_wall_usd": wl["entry_wall_usd"],
                                          "entry_wall_strength": wl["entry_wall_strength"],
                                          "imbalance_1pct": wl["imbalance_1pct"],
                                          "timestamp": str(pd.Timestamp.now(tz="UTC")),
                                          "dps_total": 0, "dps_confidence": "wall_confirmed"},
                                "risk_pct": risk_pct,
                                "placed_at": time.time(),
                                "strat": "depth_wall",
                            }
                            logger.info("    WALL ENTRY: %s %s LIMIT id=%s price=%s tp=%s sl=%s RR=%.2f (3min expiry)",
                                        wl["side"].upper(), symbol, order_id,
                                        limit_price, tp_price, sl_price, wl["rr"])
                        else:
                            logger.error("    WALL ENTRY FAILED: %s — %s",
                                         symbol, result.get("msg"))

                    # Set cooldown
                    self.depth_cooldown[f"{symbol}_depth"] = time.time()
                    with open(TRADER_DIR / "depth_cooldown.json", "w") as f:
                        json.dump(self.depth_cooldown, f)

                    to_remove.append(symbol)

        for sym in to_remove:
            self._depth_watchlist.pop(sym, None)

    # --- Periodic Tasks (background, not time-critical) ---

    def _periodic_tasks(self):
        now = time.time()

        if check_kill_switch(self.config):
            logger.warning("KILL SWITCH active")
            return

        self.pnl_tracker.reset_if_new_day()
        if self.pnl_tracker.is_limit_hit():
            logger.warning("DAILY LOSS LIMIT: %s", self.pnl_tracker.summary())
            return

        # Refresh coins every 2 min
        if now - self._last_coin_refresh > self._coin_refresh_interval:
            self._refresh_active_coins()
            self._last_coin_refresh = now

        # Refresh market condition every 2 hours
        if now - self._last_mkt_refresh > self._mkt_refresh_interval:
            self._refresh_market_condition()
            self._last_mkt_refresh = now

        # Sync positions (catch missed closes)
        with self._lock:
            closed = self.pos_tracker.sync_with_exchange()
            for ct in closed:
                outcome = ct.get("outcome", "CLOSED")
                close_price = ct.get("close_price", 0)
                pnl_pct = ct.get("pnl_pct", 0)
                realized = ct.get("realized_pnl", 0)
                fee = ct.get("fee", 0)
                logger.info("  %s %s %s %s: close=%.6g PnL=%.2f%% realized=$%.2f fee=$%.2f",
                             outcome, ct["strategy"].upper(), ct["side"].upper(),
                             ct["symbol"], close_price, pnl_pct, realized, fee)
                log_trade(ct, "CLOSED")
                # Track daily PnL using account-relative risk
                risk_pct = ct.get("risk_pct", 0.1)
                if pnl_pct > 0:
                    account_pnl = risk_pct * (pnl_pct / ct.get("sl_pct", 1.0))
                else:
                    account_pnl = -risk_pct
                self.pnl_tracker.add_trade(account_pnl)

        # Trail SL check
        self._check_trail_sl()

    def _refresh_active_coins(self):
        scan_cfg = self.config["scanning"]
        coins = fetch_orion_active_coins(
            min_vol_5m=scan_cfg["min_vol_5m"], top_n=scan_cfg["top_n"])
        if not coins:
            return

        coin_symbols = {c["symbol"] for c in coins}

        # Whitelist filter
        if self.approved_symbols:
            for sym in coin_symbols - self.approved_symbols:
                pair = self.pairs_info.get(sym, {})
                lev = pair.get("maxLeverage", 999)
                if isinstance(lev, int) and lev <= 10:
                    continue
                logger.info("  NEW COIN: %s (not in whitelist)", sym)
            coin_symbols = {s for s in coin_symbols if s in self.approved_symbols}

        open_syms = self.pos_tracker.get_all_symbols()
        new_symbols = (coin_symbols | open_syms) - self.active_symbols

        if new_symbols:
            self.active_symbols.update(new_symbols)
            for sym in new_symbols:
                if sym not in self.candle_cache:
                    df = fetch_klines_paginated(sym, n_bars=MR_WARMUP_BARS)
                    if df is not None and len(df) >= 200:
                        self.candle_cache[sym] = df
            self._preconfigure_margin(new_symbols)
            # Queue subscriptions for async processing
            self._pending_subscriptions.update(new_symbols)
            logger.info("Added %d symbols (total active: %d)",
                        len(new_symbols), len(self.active_symbols))

    def _preconfigure_margin(self, symbols: set[str]):
        trading_cfg = self.config["trading"]
        to_configure = symbols - self._margin_configured
        if not to_configure or self.dry_run:
            return
        logger.info("Pre-configuring margin for %d symbols...", len(to_configure))
        for sym in sorted(to_configure):
            self.client.ensure_margin_and_leverage(
                sym, trading_cfg["leverage"], trading_cfg["margin_mode"])
            self._margin_configured.add(sym)
            time.sleep(0.05)

    def _load_initial_candles(self):
        to_load = [s for s in sorted(self.active_symbols) if s not in self.candle_cache]
        logger.info("Loading initial candles for %d symbols (%d already cached)...",
                     len(to_load), len(self.candle_cache))
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
                    logger.debug("Candle fetch %s: got %d bars",
                                 sym, len(df) if df is not None else 0)
            except Exception as e:
                failed += 1
                logger.debug("Candle fetch %s error: %s", sym, e)
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

    # --- WebSocket Handlers (called from WS threads) ---

    def _handle_kline(self, symbol: str, candle: dict):
        """Called on every kline update — only process when MINUTE changes."""
        ts = candle.get("timestamp", 0)

        # Round to minute — only process when a new minute starts
        ts_minute = (ts // 60000) * 60000  # floor to minute boundary (ms)
        last_minute = self._last_candle_ts.get(symbol, 0)
        if ts_minute <= last_minute:
            # Same minute — mid-candle tick, skip
            return

        # New minute — previous candle is CLOSED
        self._last_candle_ts[symbol] = ts_minute
        logger.debug("Candle closed: %s minute=%d", symbol, ts_minute)

        # Append the new candle to cache
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

            # Queue for batch processing — only on candle close
            with self._batch_lock:
                self._pending_candle_symbols.add(symbol)

    def _handle_depth(self, symbol: str, depth_data: dict):
        pass  # Cached in ws_client._depth_cache

    def _handle_position(self, data: dict):
        event = data.get("event", "")
        symbol = data.get("symbol", "")
        side = data.get("side", "")

        if event == "CLOSE":
            pnl = float(data.get("realizedPNL", 0))
            logger.info("  WS POSITION CLOSED: %s %s realized=$%.4f", symbol, side, pnl)

        elif event == "OPEN":
            pos_id = data.get("positionId", "")
            logger.info("  WS POSITION OPENED: %s %s posId=%s", symbol, side, pos_id)
            with self._lock:
                for lp in self.pos_tracker.local_positions:
                    if lp["symbol"] == symbol and not lp.get("exchange_position_id"):
                        lp["exchange_position_id"] = pos_id
                        self.pos_tracker._save_local()
                        break

    def _handle_order(self, data: dict):
        symbol = data.get("symbol", "")
        status = data.get("orderStatus", "")
        avg_price = data.get("averagePrice") or "0"

        if status == "FILLED":
            logger.info("  WS ORDER FILLED: %s avg=%.6g", symbol, float(avg_price))
            with self._lock:
                for lp in self.pos_tracker.local_positions:
                    if lp["symbol"] == symbol and not lp.get("actual_fill"):
                        lp["actual_fill"] = float(avg_price)
                        self.pos_tracker._save_local()
                        break

    # --- Strategy Execution (called from WS kline handler) ---

    def _check_setup(self, symbol: str):
        """Run strategy on a symbol — called on each new candle."""
        if symbol not in self.candle_cache:
            return

        with self._lock:
            if check_kill_switch(self.config):
                return
            if self.pnl_tracker.is_limit_hit():
                return

            trading_cfg = self.config["trading"]
            df = self.candle_cache[symbol]

            if len(df) < 500:
                return

            # One position per coin
            if symbol in self.pos_tracker.get_all_symbols():
                return

            # Max positions
            live_count = sum(1 for p in self.pos_tracker.local_positions if p.get("mode") != "dummy")
            if live_count >= trading_cfg["max_positions"]:
                return

            # Fetch depth via REST (WS depth only gives 1 level, not enough for walls)
            from live_data_collector import fetch_bitunix_depth
            depth_data = fetch_bitunix_depth(symbol, limit="max")

            # Run strategies
            try:
                found_setups = detect_setups(
                    df, symbol, _strategy_cfg, _mr_cfg, _momo_gate_cfg,
                    depth_data=depth_data)
            except Exception as e:
                logger.debug("Strategy error %s: %s", symbol, e)
                return

            for setup in found_setups:
                strat = setup["strategy"]

                strat_mode = self.strat_modes.get(strat, "off")
                if strat_mode == "off":
                    continue

                # Cooldown for depth (timestamp-based, 30 min)
                if strat in ("depth", "depth_bounce"):
                    cd_key = f"{symbol}_{strat}"
                    if time.time() - self.depth_cooldown.get(cd_key, 0) < 1800:
                        continue

                # Market condition filter
                side = setup["side"]
                if strat == "momentum" and setup.get("dps_total", 0) < 4:
                    continue
                if not self.market_cond.is_allowed(strat, side):
                    continue

                # Position size
                balance_data = self.client.get_balance(trading_cfg["margin_coin"])
                available = float(balance_data.get("available", 0))
                if available <= 0:
                    continue

                risk_pct = get_risk_pct(setup, _strategy_cfg)
                entry_price = setup["entry"]
                sl_pct = setup["sl_pct"]

                qty = calculate_qty(
                    symbol, entry_price, risk_pct, sl_pct,
                    available, trading_cfg["leverage"],
                    self.pairs_info)

                if float(qty) <= 0:
                    continue

                order_side = "BUY" if side == "long" else "SELL"
                tp_price = str(round(setup["tp"], 8))
                sl_price = str(round(setup["sl"], 8))

                mode_tag = f"[{strat_mode.upper()}]"
                logger.info("  %s %s %s %s: entry=~%.6g tp=%s sl=%s RR=%.2f DPS=%s",
                            mode_tag, strat.upper(), side.upper(), symbol,
                            entry_price, tp_price, sl_price,
                            setup["rr"], setup.get("dps_total", "?"))

                if strat_mode == "dummy":
                    setup["order_id"] = f"dummy_{int(time.time())}"
                    setup["qty"] = qty
                    setup["risk_pct"] = risk_pct
                    setup["market_score"] = self.market_cond.score
                    setup["bars_held"] = 0
                    setup["mode"] = "dummy"
                    self.pos_tracker.add_position(setup)
                    log_trade(setup, "ENTRY_DUMMY")

                elif strat_mode == "live":
                    already_configured = symbol in self._margin_configured
                    result = self.client.place_order_and_verify(
                        symbol=symbol, side=order_side, qty=qty,
                        tp_price=tp_price, sl_price=sl_price,
                        leverage=trading_cfg["leverage"],
                        margin_mode=trading_cfg["margin_mode"],
                        tp_sl_type=trading_cfg["tp_sl_type"],
                        skip_margin_setup=already_configured)

                    if result.get("success"):
                        setup["order_id"] = result["order_id"]
                        setup["exchange_position_id"] = result["position_id"]
                        setup["qty"] = result.get("filled_qty", qty)
                        setup["actual_fill"] = result["actual_fill"]
                        setup["risk_pct"] = risk_pct
                        setup["market_score"] = self.market_cond.score
                        setup["bars_held"] = 0
                        setup["mode"] = "live"

                        if result["actual_fill"] > 0:
                            slippage = abs(result["actual_fill"] - entry_price) / entry_price * 100
                            setup["slippage_pct"] = round(slippage, 4)
                            if slippage > 0.05:
                                logger.info("    SLIPPAGE: expected=%.6g actual=%.6g (%.3f%%)",
                                            entry_price, result["actual_fill"], slippage)

                        self.pos_tracker.add_position(setup)
                        log_trade(setup, "ENTRY_LIVE")

                        logger.info("    FILLED: %s fill=%.6g posId=%s qty=%s",
                                    symbol, result["actual_fill"],
                                    result["position_id"] or "?",
                                    result.get("filled_qty", "?"))
                    else:
                        logger.error("    FAILED: %s — %s",
                                     symbol, result.get("error", "Unknown"))

                # Set cooldown
                if strat in ("depth", "depth_bounce"):
                    self.depth_cooldown[f"{symbol}_{strat}"] = time.time()
                    with open(TRADER_DIR / "depth_cooldown.json", "w") as f:
                        json.dump(self.depth_cooldown, f)

                break  # One setup per symbol per candle

    def _check_trail_sl(self):
        """Trail SL disabled — rely on exchange TP/SL only.
        Will revisit with better logic later."""
        return
        # --- DISABLED ---
        with self._lock:
            for pos in self.pos_tracker.local_positions:
                sym = pos["symbol"]
                if sym not in self.candle_cache:
                    continue
                if pos.get("sl_trailed"):
                    continue

                full_df = self.candle_cache[sym]
                if len(full_df) < 2:
                    continue

                # Use CLOSED bars only (exclude current incomplete candle)
                df = full_df.iloc[:-1]
                current_high = float(df.iloc[-1]["high"])
                current_low = float(df.iloc[-1]["low"])
                current_close = float(df.iloc[-1]["close"])

                # Use actual fill price, not setup entry
                entry = pos.get("actual_fill") or pos["entry"]
                if entry <= 0:
                    entry = pos["entry"]
                sl_orig = pos.get("sl_original", pos["sl"])

                # bars_held: use time-based counting (minutes since entry)
                entry_ts = pos.get("timestamp", "")
                if entry_ts:
                    try:
                        import pandas as _pd
                        entry_time = _pd.Timestamp(entry_ts)
                        now_time = _pd.Timestamp.now(tz="UTC")
                        bars_elapsed = int((now_time - entry_time).total_seconds() / 60)
                    except Exception:
                        bars_elapsed = pos.get("bars_held", 0)
                else:
                    bars_elapsed = pos.get("bars_held", 0)
                pos["bars_held"] = bars_elapsed

                should_trail = False
                if pos["side"] == "long":
                    r_dist = entry - sl_orig
                    if r_dist <= 0:
                        continue
                    target_09r = entry + r_dist * 0.9
                    new_sl = entry + r_dist * 0.1
                    if current_high >= target_09r:
                        should_trail = True
                    elif bars_elapsed >= 60 and current_close > entry:
                        should_trail = True
                else:
                    r_dist = sl_orig - entry
                    if r_dist <= 0:
                        continue
                    target_09r = entry - r_dist * 0.9
                    new_sl = entry - r_dist * 0.1
                    if current_low <= target_09r:
                        should_trail = True
                    elif bars_elapsed >= 60 and current_close < entry:
                        should_trail = True

                if should_trail:
                    pos_id = pos.get("exchange_position_id")
                    pos_qty = pos.get("qty") or pos.get("exchange_qty")
                    if pos_id and pos_qty and pos.get("mode") == "live":
                        # Validate SL is on correct side of current price
                        valid_sl = True
                        if pos["side"] == "long" and new_sl >= current_close:
                            valid_sl = False
                        elif pos["side"] == "short" and new_sl <= current_close:
                            valid_sl = False

                        if valid_sl:
                            logger.info("  TRAIL SL %s %s %s: SL -> %.6g (entry=%.6g, 0.9R=%.6g, bars=%d)",
                                        pos["strategy"], pos["side"], sym,
                                        new_sl, entry, target_09r, bars_elapsed)
                            self.client.modify_tp_sl(
                                sym, pos_id,
                                sl_price=str(round(new_sl, 8)),
                                qty=str(pos_qty))
                        else:
                            logger.info("  TRAIL SL skip %s: price=%.6g past new_sl=%.6g (bars=%d)",
                                        sym, current_close, new_sl, bars_elapsed)
                    if "sl_original" not in pos:
                        pos["sl_original"] = pos["sl"]
                    pos["sl"] = round(new_sl, 8)
                    pos["sl_trailed"] = True
                    self.pos_tracker._save_local()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WebSocket Live Trader")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config()
    dry_run = args.dry_run or config.get("safety", {}).get("dry_run", True)

    if not dry_run and config.get("safety", {}).get("require_confirmation", True):
        print("\n" + "=" * 60)
        print("  WEBSOCKET LIVE TRADING — REAL MONEY")
        print("=" * 60)
        print(f"  Risk: {config['trading']['risk_pct']}%")
        print(f"  Max positions: {config['trading']['max_positions']}")
        print(f"  Leverage: {config['trading']['leverage']}x")
        print(f"  Near-zero latency mode")
        confirm = input("\n  Type 'CONFIRM' to start: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return

    trader = WSTrader(config, dry_run=dry_run)
    asyncio.run(trader.start())


if __name__ == "__main__":
    main()
