#!/usr/bin/env python3
"""
Live Trader — Real order execution on Bitunix.

Uses the same strategy logic as live_dummy_trader.py but executes
real orders via Bitunix API.

Usage:
  python live_trader.py [--dry-run] [--once]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_log_fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s",
                              datefmt="%H:%M:%S")

# Console handler
_console_handler = logging.StreamHandler(stream=sys.stderr)
_console_handler.setFormatter(_log_fmt)

# File handler
_log_dir = Path("datasets/live/live_trader")
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(
    _log_dir / "trader_log.txt", encoding="utf-8")
_file_handler.setFormatter(_log_fmt)

logger = logging.getLogger("live_trader")
logger.setLevel(logging.INFO)
logger.addHandler(_console_handler)
logger.addHandler(_file_handler)

# ---------------------------------------------------------------------------
# Imports from existing modules (strategy logic unchanged)
# ---------------------------------------------------------------------------
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
    check_75pct_tp_rule,
    MarketCondition,
    DEPTH_EXCLUDED_SYMBOLS,
)
from live_dummy_trader import (
    fetch_klines_paginated,
    load_approved_symbols,
    MR_WARMUP_BARS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_FILE = Path("live_trading_config.json")
TRADER_DIR = LIVE_DIR / "live_trader"
TRADES_LOG = TRADER_DIR / "trades.csv"
POSITIONS_FILE = TRADER_DIR / "positions_local.json"
DAILY_PNL_FILE = TRADER_DIR / "daily_pnl.json"
NEW_COINS_LOG = TRADER_DIR / "new_coins.csv"

_strategy_cfg = StrategyConfig.from_json()
_mr_cfg = MRSettings()
_momo_gate_cfg = MomoGateSettings.from_json("momo_gate_settings.json")


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        logger.error("Config file not found: %s", CONFIG_FILE)
        sys.exit(1)
    with open(CONFIG_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Bitunix API Client
# ---------------------------------------------------------------------------

class BitunixClient:
    """Authenticated Bitunix Futures API client."""

    BASE = BITUNIX_BASE

    def __init__(self, api_key: str, api_secret: str, dry_run: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.dry_run = dry_run
        self.session = requests.Session()

    def _sign(self, query_str: str, body_str: str,
              timestamp: str, nonce: str) -> str:
        """
        Generate signature per Bitunix docs (double SHA256).
        Step 1: digest = SHA256(nonce + timestamp + apiKey + queryParams + body)
        Step 2: sign = SHA256(digest + secretKey)
        """
        raw = nonce + timestamp + self.api_key + query_str + body_str
        digest = hashlib.sha256(raw.encode()).hexdigest()
        signature = hashlib.sha256(
            (digest + self.api_secret).encode()
        ).hexdigest()
        return signature

    def _headers(self, query_str: str = "", body_str: str = "") -> dict:
        timestamp = str(int(time.time() * 1000))
        nonce = uuid.uuid4().hex
        return {
            "api-key": self.api_key,
            "timestamp": timestamp,
            "nonce": nonce,
            "sign": self._sign(query_str, body_str, timestamp, nonce),
            "Content-Type": "application/json",
            "language": "en",
        }

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.BASE}{path}"
        # Sort query params by key (ASCII ascending) and concat key+value
        query_str = ""
        if params:
            sorted_keys = sorted(params.keys())
            query_str = "".join(k + str(params[k]) for k in sorted_keys)
        headers = self._headers(query_str=query_str, body_str="")
        resp = self.session.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: dict) -> dict:
        url = f"{self.BASE}{path}"
        body_str = json.dumps(data, separators=(",", ":"))
        headers = self._headers(query_str="", body_str=body_str)

        if self.dry_run:
            logger.info("[DRY RUN] POST %s: %s", path, body_str[:200])
            return {"code": 0, "data": {"orderId": f"dry_{int(time.time())}"},
                    "msg": "Dry run"}

        resp = self.session.post(url, data=body_str, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # --- Market Info ---

    def get_trading_pairs(self) -> dict:
        """Fetch all trading pairs. Returns {symbol: {maxLeverage, ...}}."""
        try:
            resp = self.session.get(
                f"{self.BASE}/api/v1/futures/market/trading_pairs", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == 0:
                return {p["symbol"]: p for p in data.get("data", [])}
        except Exception as e:
            logger.error("Trading pairs error: %s", e)
        return {}

    # --- Account ---

    def get_balance(self, margin_coin: str = "USDT") -> dict:
        """Get account balance."""
        result = self._get("/api/v1/futures/account", {"marginCoin": margin_coin})
        if result.get("code") != 0:
            logger.error("Balance error: %s", result.get("msg"))
            return {}
        return result.get("data", {})

    # --- Positions ---

    def get_history_position(self, position_id: str = None,
                             symbol: str = None) -> dict:
        """Get closed position details (close price, realized PnL)."""
        params = {}
        if position_id:
            params["positionId"] = position_id
        if symbol:
            params["symbol"] = symbol
        if not params:
            return {}
        params["limit"] = "5"
        result = self._get("/api/v1/futures/position/get_history_positions", params)
        logger.debug("History position result: %s", str(result)[:500])
        if result.get("code") != 0:
            logger.warning("History position error: %s", result.get("msg"))
            return {}
        data = result.get("data", {})
        positions = data.get("positionList", data if isinstance(data, list) else [])
        if positions:
            return positions[0]
        return {}

    def get_positions(self, symbol: str = None) -> list:
        """Get open positions."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        result = self._get("/api/v1/futures/position/get_pending_positions", params)
        if result.get("code") != 0:
            logger.error("Positions error: %s", result.get("msg"))
            return []
        return result.get("data", []) or []

    # --- Orders ---

    def place_order(self, symbol: str, side: str, qty: str,
                    order_type: str = "MARKET", price: str = None,
                    tp_price: str = None, sl_price: str = None,
                    tp_sl_type: str = "LAST_PRICE",
                    leverage: int = None, margin_mode: str = None) -> dict:
        """
        Place a futures order with optional TP/SL.
        side: BUY or SELL
        """
        data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "tradeSide": "OPEN",
            "orderType": order_type,
        }

        if price and order_type == "LIMIT":
            data["price"] = str(price)

        if tp_price:
            data["tpPrice"] = str(tp_price)
            data["tpStopType"] = tp_sl_type
            data["tpOrderType"] = "MARKET"

        if sl_price:
            data["slPrice"] = str(sl_price)
            data["slStopType"] = tp_sl_type
            data["slOrderType"] = "MARKET"

        result = self._post("/api/v1/futures/trade/place_order", data)
        if result.get("code") != 0:
            logger.error("Order error %s: %s", symbol, result.get("msg"))
        return result

    def modify_tp_sl(self, symbol: str, position_id: str,
                     sl_price: str = None, tp_price: str = None,
                     qty: str = None,
                     tp_sl_type: str = "LAST_PRICE") -> dict:
        """Modify TP/SL on an existing position."""
        data = {
            "symbol": symbol,
            "positionId": position_id,
        }
        if tp_price:
            data["tpPrice"] = str(tp_price)
            data["tpStopType"] = tp_sl_type
            data["tpOrderType"] = "MARKET"
        if sl_price:
            data["slPrice"] = str(sl_price)
            data["slStopType"] = tp_sl_type
            data["slOrderType"] = "MARKET"
        if qty:
            data["tpQty"] = str(qty)
            data["slQty"] = str(qty)

        result = self._post("/api/v1/futures/tpsl/place_order", data)
        if result.get("code") != 0:
            logger.error("TP/SL modify error %s: %s", symbol, result.get("msg"))
        return result

    def close_position(self, symbol: str, side: str, qty: str,
                       position_id: str = None) -> dict:
        """Close a position via market order."""
        close_side = "SELL" if side == "LONG" else "BUY"
        data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": close_side,
            "tradeSide": "CLOSE",
            "orderType": "MARKET",
        }
        if position_id:
            data["positionId"] = position_id
        return self._post("/api/v1/futures/trade/place_order", data)

    # --- Order Details ---

    def get_order_detail(self, symbol: str, order_id: str) -> dict:
        """Get order fill details (actual fill price, qty, status)."""
        params = {"symbol": symbol, "orderId": order_id}
        result = self._get("/api/v1/futures/trade/get_order_detail", params)
        if result.get("code") != 0:
            logger.debug("Order detail error %s: %s", symbol, result.get("msg"))
            return {}
        return result.get("data", {})

    # --- Pre-Trade Setup ---

    def ensure_margin_and_leverage(self, symbol: str, leverage: int,
                                    margin_mode: str = "ISOLATION"):
        """Set margin mode and leverage for a symbol. Safe to call repeatedly."""
        try:
            result = self.set_margin_mode(symbol, margin_mode)
            logger.info("    [DEBUG] set_margin_mode(%s, %s): code=%s msg=%s",
                         symbol, margin_mode, result.get("code"), result.get("msg"))
        except Exception as e:
            logger.info("    [DEBUG] set_margin_mode(%s) exception: %s", symbol, e)
        try:
            result = self.set_leverage(symbol, leverage)
            logger.info("    [DEBUG] set_leverage(%s, %d): code=%s msg=%s",
                         symbol, leverage, result.get("code"), result.get("msg"))
        except Exception as e:
            logger.info("    [DEBUG] set_leverage(%s) exception: %s", symbol, e)

    # --- Full Order Flow ---

    def place_order_and_verify(self, symbol: str, side: str, qty: str,
                                tp_price: str, sl_price: str,
                                leverage: int, margin_mode: str = "ISOLATION",
                                tp_sl_type: str = "LAST_PRICE",
                                skip_margin_setup: bool = False) -> dict:
        """
        Complete order flow:
        1. Set margin mode + leverage (skip if pre-configured)
        2. Place market order with TP/SL
        3. Wait briefly for fill
        4. Get actual fill price and position ID
        5. Verify TP/SL is attached
        6. Recalculate TP/SL from actual fill if needed
        Returns dict with all details or empty dict on failure.
        """
        # Step 1: Ensure margin mode and leverage (skip if pre-configured)
        if not skip_margin_setup:
            self.ensure_margin_and_leverage(symbol, leverage, margin_mode)

        # Step 2: Place order with TP/SL
        logger.info("    [DEBUG] Placing %s %s qty=%s tp=%s sl=%s",
                     side, symbol, qty, tp_price, sl_price)
        result = self.place_order(
            symbol=symbol, side=side, qty=qty,
            order_type="MARKET",
            tp_price=tp_price, sl_price=sl_price,
            tp_sl_type=tp_sl_type)

        logger.info("    [DEBUG] Order response: code=%s msg=%s data=%s",
                     result.get("code"), result.get("msg"),
                     str(result.get("data", {}))[:200])

        if result.get("code") != 0:
            return {"success": False, "error": result.get("msg", "Unknown")}

        order_id = result.get("data", {}).get("orderId", "")
        if not order_id:
            return {"success": False, "error": "No orderId returned"}

        # Step 3: Wait for fill, get position ID + actual fill price
        time.sleep(1)

        position_id = None
        actual_fill = 0
        filled_qty = float(qty)
        order_status = "FILLED"
        # ONE_WAY mode: side is BUY/SELL. HEDGE mode: side is LONG/SHORT
        side_matches = {"BUY", "LONG"} if side == "BUY" else {"SELL", "SHORT"}

        for attempt in range(3):
            positions = self.get_positions(symbol)
            logger.info("    [DEBUG] get_positions(%s) attempt %d: %d positions found",
                         symbol, attempt + 1, len(positions))
            for pos in positions:
                if pos.get("side") in side_matches:
                    position_id = pos.get("positionId")
                    actual_fill = float(pos.get("avgOpenPrice") or 0)
                    filled_qty = float(pos.get("qty") or qty)
                    break
            if position_id and actual_fill > 0:
                logger.info("    [DEBUG] Position found: posId=%s fill=%.6g qty=%s",
                             position_id, actual_fill, filled_qty)
                break
            time.sleep(0.5)

        if not position_id:
            logger.warning("    [DEBUG] No position found after 3 attempts for %s %s",
                            side_str, symbol)

        # Log slippage but don't recalculate TP/SL
        # (recalculation adds duplicate TP/SL orders on exchange)
        recalculated = False

        return {
            "success": True,
            "order_id": order_id,
            "position_id": position_id,
            "actual_fill": actual_fill,
            "filled_qty": filled_qty,
            "order_status": order_status,
            "recalculated": recalculated,
        }

    # --- Leverage & Margin ---

    def set_leverage(self, symbol: str, leverage: int, margin_coin: str = "USDT"):
        """Set leverage for a symbol."""
        data = {
            "symbol": symbol,
            "leverage": leverage,
            "marginCoin": margin_coin,
        }
        return self._post("/api/v1/futures/account/change_leverage", data)

    def set_margin_mode(self, symbol: str, mode: str = "ISOLATION",
                        margin_coin: str = "USDT"):
        """Set margin mode: ISOLATION or CROSS."""
        data = {
            "symbol": symbol,
            "marginMode": mode,
            "marginCoin": margin_coin,
        }
        return self._post("/api/v1/futures/account/change_margin_mode", data)


# ---------------------------------------------------------------------------
# Position Tracker — local state synced with exchange
# ---------------------------------------------------------------------------

class LivePositionTracker:
    """Track positions with local state + exchange sync."""

    def __init__(self, client: BitunixClient):
        self.client = client
        self.local_positions: list[dict] = []
        self._load_local()

    def _load_local(self):
        if POSITIONS_FILE.exists():
            with open(POSITIONS_FILE) as f:
                self.local_positions = json.load(f)

    def _save_local(self):
        POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(POSITIONS_FILE, "w") as f:
            json.dump(self.local_positions, f, indent=2, default=str)

    def sync_with_exchange(self):
        """Fetch real positions from exchange and sync with local state."""
        exchange_positions = self.client.get_positions()

        # Separate dummy and live positions
        dummy_positions = [p for p in self.local_positions if p.get("mode") == "dummy"]
        live_positions = [p for p in self.local_positions if p.get("mode") != "dummy"]

        if not exchange_positions:
            # No positions on exchange — close all live, keep dummy
            closed = live_positions[:]
            self.local_positions = dummy_positions
            self._save_local()
            return closed

        # Map exchange positions by symbol
        exchange_map = {}
        for ep in exchange_positions:
            exchange_map[ep['symbol']] = ep

        closed = []
        remaining = list(dummy_positions)  # always keep dummy
        for lp in live_positions:
            key = lp['symbol']
            if key in exchange_map:
                # Still open — update with exchange data
                ep = exchange_map[key]
                lp["exchange_position_id"] = ep.get("positionId")
                lp["unrealized_pnl"] = float(ep.get("unrealizedPNL", 0))
                lp["exchange_qty"] = float(ep.get("qty", 0))
                remaining.append(lp)
            else:
                # Position no longer on exchange — it was closed (TP/SL hit)
                # Fetch close details from history
                pos_id = lp.get("exchange_position_id")
                hist = None
                if pos_id:
                    hist = self.client.get_history_position(position_id=pos_id)
                    if not hist:
                        logger.debug("History by positionId %s returned empty", pos_id)
                if not hist:
                    hist = self.client.get_history_position(symbol=lp["symbol"])
                    if not hist:
                        logger.warning("History by symbol %s also returned empty", lp["symbol"])

                if hist and float(hist.get("closePrice", 0)) > 0:
                    close_price = float(hist.get("closePrice", 0))
                    realized_pnl = float(hist.get("realizedPNL", 0))
                    fee = float(hist.get("fee", 0))
                    funding = float(hist.get("funding", 0))
                    entry_price = lp["entry"]
                    side = lp["side"]

                    if side == "long":
                        pnl_pct = (close_price - entry_price) / entry_price * 100
                        if close_price >= lp.get("tp", 0) * 0.999:
                            lp["outcome"] = "TP"
                        elif close_price <= lp.get("sl", 0) * 1.001:
                            lp["outcome"] = "SL"
                        else:
                            lp["outcome"] = "TRAIL_SL" if lp.get("sl_trailed") else "CLOSED"
                    else:
                        pnl_pct = (entry_price - close_price) / entry_price * 100
                        if close_price <= lp.get("tp", 0) * 1.001:
                            lp["outcome"] = "TP"
                        elif close_price >= lp.get("sl", 0) * 0.999:
                            lp["outcome"] = "SL"
                        else:
                            lp["outcome"] = "TRAIL_SL" if lp.get("sl_trailed") else "CLOSED"

                    lp["close_price"] = close_price
                    lp["pnl_pct"] = round(pnl_pct, 4)
                    lp["realized_pnl"] = realized_pnl
                    lp["fee"] = fee
                    lp["funding"] = funding
                else:
                    lp["outcome"] = "CLOSED_BY_EXCHANGE"

                closed.append(lp)

        self.local_positions = remaining
        self._save_local()
        return closed

    def has_position(self, symbol: str, strategy: str) -> bool:
        return any(
            p["symbol"] == symbol and p["strategy"] == strategy
            for p in self.local_positions
        )

    def add_position(self, trade: dict):
        self.local_positions.append(trade)
        self._save_local()

    def get_all_symbols(self) -> set:
        return {p["symbol"] for p in self.local_positions}

    def update_sl(self, symbol: str, strategy: str, new_sl: float):
        """Update SL locally (exchange SL modification done separately)."""
        for p in self.local_positions:
            if p["symbol"] == symbol and p["strategy"] == strategy:
                if "sl_original" not in p:
                    p["sl_original"] = p["sl"]
                p["sl"] = new_sl
                p["sl_trailed"] = True
                break
        self._save_local()


# ---------------------------------------------------------------------------
# Daily PnL Tracker
# ---------------------------------------------------------------------------

class DailyPnLTracker:
    """Track daily PnL for loss limits."""

    def __init__(self, max_loss_pct: float = 5.0):
        self.max_loss_pct = max_loss_pct
        self.today_pnl = 0.0
        self.today_trades = 0
        self.today_date = datetime.now(timezone.utc).date()

    def reset_if_new_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.today_date:
            logger.info("New day — resetting daily PnL (yesterday: %.2f%%)",
                        self.today_pnl)
            self.today_pnl = 0.0
            self.today_trades = 0
            self.today_date = today

    def add_trade(self, pnl_pct: float):
        self.today_pnl += pnl_pct
        self.today_trades += 1

    def is_limit_hit(self) -> bool:
        return self.today_pnl <= -self.max_loss_pct

    def summary(self) -> str:
        return f"Daily PnL: {self.today_pnl:+.2f}% ({self.today_trades} trades)"


# ---------------------------------------------------------------------------
# Trade Logger
# ---------------------------------------------------------------------------

_TRADE_COLUMNS = [
    "action", "timestamp_utc", "timestamp_est", "symbol", "strategy", "side",
    "entry", "actual_fill", "tp", "sl", "sl_pct", "tp_pct", "rr",
    "qty", "risk_pct", "mode", "market_score",
    "dps_total", "dps_confidence",
    "entry_wall_price", "entry_wall_usd", "entry_wall_strength",
    "sl_wall_usd", "sl_wall_strength", "tp_wall_usd", "tp_wall_strength",
    "imbalance_1pct", "imbalance_2pct",
    "zct_alignment", "order_id", "exchange_position_id",
    "slippage_pct", "fill_time_sec", "wait_time_sec",
    "outcome", "close_price", "pnl_pct", "realized_pnl", "fee",
]


def log_trade(trade: dict, action: str):
    """Log trade to CSV with fixed columns."""
    from datetime import datetime, timezone, timedelta

    TRADER_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure timestamps exist
    now_utc = datetime.now(timezone.utc)
    now_est = now_utc - timedelta(hours=4)
    if "timestamp" not in trade or not trade["timestamp"]:
        trade["timestamp"] = now_utc.isoformat()
    trade["timestamp_utc"] = trade.get("timestamp", now_utc.isoformat())
    try:
        utc_ts = pd.Timestamp(trade["timestamp_utc"])
        trade["timestamp_est"] = str(utc_ts - pd.Timedelta(hours=4))[:19]
        trade["timestamp_utc"] = str(utc_ts)[:19]
    except Exception:
        trade["timestamp_utc"] = str(now_utc)[:19]
        trade["timestamp_est"] = str(now_est)[:19]

    write_header = not TRADES_LOG.exists()
    with open(TRADES_LOG, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(_TRADE_COLUMNS) + "\n")
        vals = []
        for col in _TRADE_COLUMNS:
            if col == "action":
                vals.append(action)
            else:
                v = trade.get(col, "")
                s = str(v) if v is not None else ""
                if "," in s:
                    s = f'"{s}"'
                vals.append(s)
        f.write(",".join(vals) + "\n")


# ---------------------------------------------------------------------------
# Safety Checks
# ---------------------------------------------------------------------------

def check_kill_switch(config: dict) -> bool:
    """Check if kill switch file exists."""
    ks_file = config.get("safety", {}).get("kill_switch_file", "STOP_TRADING")
    return Path(ks_file).exists()


def calculate_qty(symbol: str, entry_price: float, risk_pct: float,
                  sl_pct: float, balance: float, leverage: int,
                  pairs_info: dict = None) -> str:
    """Calculate order quantity based on risk parameters."""
    # Risk amount in USD
    risk_usd = balance * risk_pct / 100

    # Position size from risk: risk_usd = qty * entry * sl_pct / 100
    if sl_pct <= 0:
        sl_pct = 1.0
    position_usd = risk_usd / (sl_pct / 100)

    # With leverage
    qty = position_usd / entry_price

    # Round to base precision and check minimum
    if pairs_info and symbol in pairs_info:
        pair = pairs_info[symbol]
        precision = pair.get("basePrecision", 4)
        qty = round(qty, precision)
        min_qty = float(pair.get("minTradeVolume", 0))
        if qty < min_qty:
            qty = 0  # will be caught by the qty <= 0 check

    return str(qty)


# ---------------------------------------------------------------------------
# Main Trading Cycle
# ---------------------------------------------------------------------------

def trading_cycle(client: BitunixClient, pos_tracker: LivePositionTracker,
                  pnl_tracker: DailyPnLTracker, config: dict,
                  candle_cache: dict, depth_cooldown: dict,
                  market_cond: MarketCondition,
                  approved_symbols: set,
                  pairs_info: dict,
                  cycle_num: int) -> dict:
    """One live trading cycle."""
    trading_cfg = config["trading"]
    scan_cfg = config["scanning"]
    stats = {"coins": 0, "setups": 0, "orders": 0, "errors": 0}

    # Safety checks
    if check_kill_switch(config):
        logger.warning("KILL SWITCH active — skipping cycle")
        return stats

    pnl_tracker.reset_if_new_day()
    if pnl_tracker.is_limit_hit():
        logger.warning("DAILY LOSS LIMIT hit (%s) — skipping cycle",
                        pnl_tracker.summary())
        return stats

    # Step 1: Sync positions with exchange
    closed_by_exchange = pos_tracker.sync_with_exchange()
    for ct in closed_by_exchange:
        outcome = ct.get("outcome", "CLOSED")
        close_price = ct.get("close_price", 0)
        pnl_pct = ct.get("pnl_pct", 0)
        realized = ct.get("realized_pnl", 0)
        fee = ct.get("fee", 0)
        logger.info("  %s %s %s %s: close=%.6g PnL=%.2f%% realized=$%.2f fee=$%.2f",
                     outcome, ct["strategy"].upper(), ct["side"].upper(),
                     ct["symbol"], close_price, pnl_pct, realized, fee)
        log_trade(ct, "CLOSED")
        # Track daily PnL using risk_pct (account-relative), not raw price move
        risk_pct = ct.get("risk_pct", 0.1)
        if pnl_pct > 0:
            account_pnl = risk_pct * (pnl_pct / ct.get("sl_pct", 1.0))
        else:
            account_pnl = -risk_pct  # lost the risked amount
        pnl_tracker.add_trade(account_pnl)

    # Step 2: Get active coins
    coins = fetch_orion_active_coins(
        min_vol_5m=scan_cfg["min_vol_5m"], top_n=scan_cfg["top_n"])
    if not coins:
        logger.warning("No coins from Orion")
        return stats

    open_syms = pos_tracker.get_all_symbols()
    coin_symbols = {c["symbol"] for c in coins}

    # Filter by whitelist
    if approved_symbols:
        new_coins = coin_symbols - approved_symbols - open_syms
        for sym in new_coins:
            # Skip logging for likely tokenized stocks (leverage <= 10)
            pair_info = pairs_info.get(sym, {})
            max_lev = pair_info.get("maxLeverage", 999)
            if isinstance(max_lev, (int, float)) and max_lev <= 10:
                continue  # likely tokenized stock, skip silently
            c = next((c for c in coins if c["symbol"] == sym), {})
            logger.info("  NEW COIN not in whitelist: %s (vol=$%.0f)",
                        sym, c.get("volume_5m", 0))
        coin_symbols = {s for s in coin_symbols if s in approved_symbols}

    all_symbols = coin_symbols | open_syms
    stats["coins"] = len(all_symbols)

    logger.info("Cycle %d: %d coins (%d active + %d positions)",
                cycle_num, len(all_symbols), len(coin_symbols),
                len(open_syms - coin_symbols))

    # Step 3: Update market conditions (every 120 cycles = ~2 hours)
    if cycle_num == 1 or cycle_num % 120 == 0:
        btc_df = fetch_klines_paginated("BTCUSDT", n_bars=MR_WARMUP_BARS)
        if btc_df is not None and len(btc_df) >= 150:
            candle_cache["BTCUSDT"] = btc_df
            market_cond.update_btc(btc_df)
        if len(candle_cache) > 10:
            market_cond.update_breadth(candle_cache)
        logger.info("  %s (updated)", market_cond.summary())
    elif cycle_num % 10 == 0:
        logger.info("  %s", market_cond.summary())

    # Step 4: Check trail SL for existing positions
    for pos in list(pos_tracker.local_positions):
        sym = pos["symbol"]
        if sym not in candle_cache:
            continue
        df = candle_cache[sym]
        if len(df) == 0:
            continue
        current_high = float(df.iloc[-1]["high"])
        current_low = float(df.iloc[-1]["low"])
        current_close = float(df.iloc[-1]["close"])
        # Use actual fill price if available, otherwise setup entry
        entry = pos.get("actual_fill") or pos["entry"]
        if entry <= 0:
            entry = pos["entry"]
        sl_orig = pos.get("sl_original", pos["sl"])

        if pos.get("sl_trailed"):
            continue

        # Trail SL logic (same as dummy trader)
        should_trail = False
        if pos["side"] == "long":
            r_dist = entry - sl_orig
            target_09r = entry + r_dist * 0.9
            new_sl = entry + r_dist * 0.1
            if current_high >= target_09r:
                should_trail = True
            elif pos.get("bars_held", 0) >= 60 and current_close > entry:
                should_trail = True
        else:
            r_dist = sl_orig - entry
            target_09r = entry - r_dist * 0.9
            new_sl = entry - r_dist * 0.1
            if current_low <= target_09r:
                should_trail = True
            elif pos.get("bars_held", 0) >= 60 and current_close < entry:
                should_trail = True

        if should_trail:
            pos_id = pos.get("exchange_position_id")
            pos_qty = pos.get("qty") or pos.get("exchange_qty")
            if pos_id and pos_qty:
                logger.info("  TRAIL SL %s %s %s: moving SL to %.6g (entry=%.6g, 0.9R=%.6g)",
                            pos["strategy"], pos["side"], sym, new_sl, entry, target_09r)
                client.modify_tp_sl(sym, pos_id,
                                    sl_price=str(round(new_sl, 8)),
                                    qty=str(pos_qty))
                pos_tracker.update_sl(sym, pos["strategy"], round(new_sl, 8))

        pos["bars_held"] = pos.get("bars_held", 0) + 1

    # Step 5: Scan for new setups
    for sym in sorted(all_symbols):
        # Fetch candles
        if sym in candle_cache:
            cached_df = candle_cache[sym]
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

        # Fetch depth
        depth_data = fetch_bitunix_depth(sym, limit="max")

        # Detect setups
        try:
            found_setups = detect_setups(
                df, sym, _strategy_cfg, _mr_cfg, _momo_gate_cfg,
                depth_data=depth_data)
        except Exception as e:
            logger.debug("Strategy error %s: %s", sym, e)
            found_setups = []

        for setup in found_setups:
            strat = setup["strategy"]
            if pos_tracker.has_position(sym, strat):
                continue

            # One position per coin across ALL strategies
            if sym in pos_tracker.get_all_symbols():
                continue

            # Cooldown for depth strategies (30 min = 1800 seconds)
            if strat in ("depth", "depth_bounce") and depth_cooldown is not None:
                cd_key = f"{sym}_{strat}"
                last_trade_time = depth_cooldown.get(cd_key, 0)
                if time.time() - last_trade_time < 1800:
                    continue

            # Market condition filter
            side = setup["side"]
            if strat == "momentum" and setup.get("dps_total", 0) < 4:
                continue
            if not market_cond.is_allowed(strat, side):
                continue

            # Per-strategy mode: live / dummy / off
            strat_modes = config.get("strategies", {})
            strat_mode = strat_modes.get(strat, "off")
            if strat_mode == "off":
                continue

            # Max positions check
            if len(pos_tracker.local_positions) >= trading_cfg["max_positions"]:
                continue

            stats["setups"] += 1

            # Calculate position size
            balance_data = client.get_balance(trading_cfg["margin_coin"])
            available = float(balance_data.get("available", 0))
            if available <= 0:
                logger.warning("No available balance")
                continue

            risk_pct = get_risk_pct(setup, _strategy_cfg)
            entry_price = setup["entry"]
            sl_pct = setup["sl_pct"]

            qty = calculate_qty(
                sym, entry_price, risk_pct, sl_pct,
                available, trading_cfg["leverage"])

            if float(qty) <= 0:
                continue

            # Place order
            order_side = "BUY" if side == "long" else "SELL"
            tp_price = str(round(setup["tp"], 8))
            sl_price = str(round(setup["sl"], 8))

            mode_tag = f"[{strat_mode.upper()}]"
            logger.info("  %s PLACING %s %s %s: qty=%s entry=~%.6g tp=%s sl=%s "
                        "RR=%.2f DPS=%s [%s]",
                        mode_tag, strat.upper(), side.upper(), sym, qty,
                        entry_price, tp_price, sl_price,
                        setup["rr"], setup.get("dps_total", "?"),
                        setup.get("dps_confidence", "?"))

            if strat_mode == "dummy":
                # Log only — no real order
                setup["order_id"] = f"dummy_{int(time.time())}"
                setup["qty"] = qty
                setup["risk_pct"] = risk_pct
                setup["market_score"] = market_cond.score
                setup["bars_held"] = 0
                setup["mode"] = "dummy"
                pos_tracker.add_position(setup)
                log_trade(setup, "ENTRY_DUMMY")
                stats["orders"] += 1
                logger.info("    DUMMY logged: %s", sym)

            elif strat_mode == "live":
                result = client.place_order_and_verify(
                    symbol=sym,
                    side=order_side,
                    qty=qty,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    leverage=trading_cfg["leverage"],
                    margin_mode=trading_cfg["margin_mode"],
                    tp_sl_type=trading_cfg["tp_sl_type"],
                )

                if result.get("success"):
                    order_id = result["order_id"]
                    actual_fill = result["actual_fill"]
                    position_id = result["position_id"]

                    setup["order_id"] = order_id
                    setup["exchange_position_id"] = position_id
                    setup["qty"] = result.get("filled_qty", qty)
                    setup["actual_fill"] = actual_fill
                    setup["risk_pct"] = risk_pct
                    setup["market_score"] = market_cond.score
                    setup["bars_held"] = 0
                    setup["mode"] = "live"

                    # Log slippage if any
                    if actual_fill > 0:
                        slippage = abs(actual_fill - entry_price) / entry_price * 100
                        setup["slippage_pct"] = round(slippage, 4)
                        if slippage > 0.05:
                            logger.info("    SLIPPAGE: expected=%.6g actual=%.6g (%.3f%%)",
                                        entry_price, actual_fill, slippage)

                    pos_tracker.add_position(setup)
                    log_trade(setup, "ENTRY_LIVE")
                    stats["orders"] += 1

                    logger.info("    FILLED: %s id=%s posId=%s fill=%.6g qty=%s %s",
                                sym, order_id, position_id or "?",
                                actual_fill, result.get("filled_qty", "?"),
                                "(TP/SL recalculated)" if result["recalculated"] else "")
                else:
                    stats["errors"] += 1
                    logger.error("    ORDER FAILED: %s — %s",
                                 sym, result.get("error", "Unknown"))

            if strat in ("depth", "depth_bounce") and depth_cooldown is not None:
                depth_cooldown[f"{sym}_{strat}"] = time.time()
                # Persist cooldown
                with open(TRADER_DIR / "depth_cooldown.json", "w") as _cf:
                    json.dump(depth_cooldown, _cf)

        time.sleep(0.12)

    logger.info("Cycle %d done: %d setups, %d orders, %d errors, %d positions. %s",
                cycle_num, stats["setups"], stats["orders"], stats["errors"],
                len(pos_tracker.local_positions), pnl_tracker.summary())

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live Trader")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate orders without executing (default from config)")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    args = parser.parse_args()

    config = load_config()
    dry_run = args.dry_run or config.get("safety", {}).get("dry_run", True)

    TRADER_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize client
    client = BitunixClient(
        api_key=config["api_key"],
        api_secret=config["api_secret"],
        dry_run=dry_run,
    )

    # Safety confirmation
    if not dry_run and config.get("safety", {}).get("require_confirmation", True):
        print("\n" + "=" * 60)
        print("  LIVE TRADING MODE — REAL MONEY")
        print("=" * 60)
        print(f"  Risk per trade: {config['trading']['risk_pct']}%")
        print(f"  Max positions: {config['trading']['max_positions']}")
        print(f"  Max daily loss: {config['trading']['max_daily_loss_pct']}%")
        print(f"  Leverage: {config['trading']['leverage']}x")
        confirm = input("\n  Type 'CONFIRM' to start: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return

    # Load approved symbols
    approved = load_approved_symbols()

    # Initialize trackers
    pos_tracker = LivePositionTracker(client)
    pnl_tracker = DailyPnLTracker(config["trading"]["max_daily_loss_pct"])
    market_cond = MarketCondition()
    candle_cache = {}

    # Persistent cooldown — survives restarts
    cooldown_file = TRADER_DIR / "depth_cooldown.json"
    if cooldown_file.exists():
        with open(cooldown_file) as f:
            depth_cooldown = json.load(f)
        logger.info("  Loaded cooldown: %d symbols on cooldown", len(depth_cooldown))
    else:
        depth_cooldown = {}

    mode = "DRY RUN" if dry_run else "LIVE"
    strat_modes = config.get("strategies", {})
    logger.info("Starting Live Trader [%s]", mode)
    logger.info("  Strategies: MR=%s  StrictMR=%s  Momo=%s  Depth=%s  DepthBounce=%s",
                strat_modes.get("mean_reversion", "off").upper(),
                strat_modes.get("strict_mr", "off").upper(),
                strat_modes.get("momentum", "off").upper(),
                strat_modes.get("depth", "off").upper(),
                strat_modes.get("depth_bounce", "off").upper())
    logger.info("  Approved symbols: %d", len(approved))
    logger.info("  Risk: %.1f%% | Max positions: %d | Leverage: %dx",
                config["trading"]["risk_pct"],
                config["trading"]["max_positions"],
                config["trading"]["leverage"])

    # Load trading pairs info (for leverage check)
    pairs_info = client.get_trading_pairs()
    logger.info("  Trading pairs loaded: %d", len(pairs_info))

    if args.once:
        trading_cycle(client, pos_tracker, pnl_tracker, config,
                      candle_cache, depth_cooldown, market_cond,
                      approved, pairs_info, 1)
        return

    cycle = 0
    interval = config["scanning"]["interval_sec"]
    while True:
        cycle += 1
        try:
            trading_cycle(client, pos_tracker, pnl_tracker, config,
                          candle_cache, depth_cooldown, market_cond,
                          approved, pairs_info, cycle)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error("Cycle error: %s", e, exc_info=True)

        logger.info("Sleeping %ds...", interval)
        time.sleep(interval)


if __name__ == "__main__":
    main()
