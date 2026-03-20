#!/usr/bin/env python3
"""
Bitunix WebSocket Client — Real-time market data + private channels.

Public channels:
  - market_kline_1min: 1m candle updates for each subscribed symbol
  - depth_books: order book updates

Private channels (authenticated):
  - position: position open/update/close events
  - order: order fill/cancel events

Usage:
    client = BitunixWS(api_key, api_secret)
    client.on_kline = my_kline_handler
    client.on_depth = my_depth_handler
    client.on_position = my_position_handler
    client.on_order = my_order_handler
    client.start()
    client.subscribe_kline(["BTCUSDT", "ETHUSDT"])
    client.subscribe_depth(["BTCUSDT"])
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from typing import Callable, Optional

import websocket

logger = logging.getLogger("ws_client")

PUBLIC_WS_URL = "wss://fapi.bitunix.com/public/"
PRIVATE_WS_URL = "wss://fapi.bitunix.com/private/"
PING_INTERVAL = 20  # seconds


class BitunixWS:
    """WebSocket client for Bitunix public + private channels."""

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret

        # WebSocket connections
        self._public_ws: Optional[websocket.WebSocketApp] = None
        self._private_ws: Optional[websocket.WebSocketApp] = None
        self._public_thread: Optional[threading.Thread] = None
        self._private_thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._running = False

        # Subscriptions tracking
        self._kline_symbols: set[str] = set()
        self._depth_symbols: set[str] = set()

        # Callbacks
        self.on_kline: Optional[Callable] = None      # (symbol, data) -> None
        self.on_depth: Optional[Callable] = None       # (symbol, data) -> None
        self.on_position: Optional[Callable] = None    # (data) -> None
        self.on_order: Optional[Callable] = None       # (data) -> None
        self.on_connect: Optional[Callable] = None     # () -> None

        # Candle buffer — builds full candle history from WS updates
        self._candle_buffer: dict[str, list] = {}  # symbol -> list of candle dicts
        self._depth_cache: dict[str, dict] = {}    # symbol -> latest depth data

    # --- Signing ---

    def _sign_ws(self) -> dict:
        """Generate authentication payload for private WS."""
        timestamp = str(int(time.time() * 1000))
        nonce = uuid.uuid4().hex
        raw = nonce + timestamp + self.api_key
        digest = hashlib.sha256(raw.encode()).hexdigest()
        sign = hashlib.sha256((digest + self.api_secret).encode()).hexdigest()
        return {
            "apiKey": self.api_key,
            "timestamp": int(timestamp),
            "nonce": nonce,
            "sign": sign,
        }

    # --- Public WS ---

    def _on_public_open(self, ws):
        logger.info("Public WS connected")
        # Re-subscribe to any tracked symbols
        if self._kline_symbols:
            self._send_subscribe(ws, "market_kline_1min", self._kline_symbols)
        if self._depth_symbols:
            self._send_subscribe(ws, "depth_books", self._depth_symbols)
        if self.on_connect:
            self.on_connect()

    def _on_public_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        # Handle ping/pong
        if data.get("op") == "ping":
            pong_ts = data.get("ping", int(time.time()))
            ws.send(json.dumps({"op": "ping", "ping": pong_ts}))
            return

        ch = data.get("ch", "")
        symbol = data.get("symbol", "")

        if "kline" in ch and self.on_kline:
            kline_data = data.get("data", {})
            ts = data.get("ts", 0)
            candle = {
                "timestamp": ts,
                "open": float(kline_data.get("o", 0)),
                "high": float(kline_data.get("h", 0)),
                "low": float(kline_data.get("l", 0)),
                "close": float(kline_data.get("c", 0)),
                "volume": float(kline_data.get("q", 0)),  # quote volume
            }
            # Update candle buffer
            if symbol not in self._candle_buffer:
                self._candle_buffer[symbol] = []
            self._candle_buffer[symbol].append(candle)
            # Keep last 1000 candles
            if len(self._candle_buffer[symbol]) > 1000:
                self._candle_buffer[symbol] = self._candle_buffer[symbol][-1000:]

            self.on_kline(symbol, candle)

        elif "depth" in ch and self.on_depth:
            depth_data = data.get("data", data)
            self._depth_cache[symbol] = {
                "asks": depth_data.get("asks", []),
                "bids": depth_data.get("bids", []),
                "ts": data.get("ts", 0),
            }
            self.on_depth(symbol, self._depth_cache[symbol])

    def _on_public_error(self, ws, error):
        logger.error("Public WS error: %s", error)

    def _on_public_close(self, ws, close_status, close_msg):
        logger.warning("Public WS closed: %s %s", close_status, close_msg)
        if self._running:
            logger.info("Reconnecting public WS in 5s...")
            time.sleep(5)
            self._start_public()

    # --- Private WS ---

    def _on_private_open(self, ws):
        logger.info("Private WS connected, authenticating...")
        login = {
            "op": "login",
            "args": [self._sign_ws()],
        }
        ws.send(json.dumps(login))

    def _on_private_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        # Handle ping/pong
        if data.get("op") == "ping":
            pong_ts = data.get("ping", int(time.time()))
            ws.send(json.dumps({"op": "ping", "ping": pong_ts}))
            return

        # Handle login response
        if data.get("op") == "login":
            if data.get("code") == 0:
                logger.info("Private WS authenticated")
                # Subscribe to private channels
                self._send_private_subscribe(ws)
            else:
                logger.error("Private WS auth failed: %s", data.get("msg"))
            return

        ch = data.get("ch", "")

        if ch == "position" and self.on_position:
            self.on_position(data.get("data", {}))

        elif ch == "order" and self.on_order:
            self.on_order(data.get("data", {}))

    def _on_private_error(self, ws, error):
        logger.error("Private WS error: %s", error)

    def _on_private_close(self, ws, close_status, close_msg):
        logger.warning("Private WS closed: %s %s", close_status, close_msg)
        if self._running:
            logger.info("Reconnecting private WS in 5s...")
            time.sleep(5)
            self._start_private()

    def _send_private_subscribe(self, ws):
        """Subscribe to all private channels."""
        sub = {
            "op": "subscribe",
            "args": [
                {"ch": "position"},
                {"ch": "order"},
            ],
        }
        ws.send(json.dumps(sub))
        logger.info("Subscribed to private channels: position, order")

    # --- Subscription management ---

    def _send_subscribe(self, ws, channel: str, symbols: set[str]):
        """Send subscription for a channel + symbols."""
        args = [{"symbol": sym, "ch": channel} for sym in symbols]
        msg = {"op": "subscribe", "args": args}
        ws.send(json.dumps(msg))
        logger.info("Subscribed %s: %d symbols", channel, len(symbols))

    def subscribe_kline(self, symbols: list[str]):
        """Subscribe to 1m kline for symbols."""
        new_syms = set(symbols) - self._kline_symbols
        if not new_syms:
            return
        self._kline_symbols.update(new_syms)
        if self._public_ws:
            self._send_subscribe(self._public_ws, "market_kline_1min", new_syms)

    def unsubscribe_kline(self, symbols: list[str]):
        """Unsubscribe from kline for symbols."""
        rm_syms = set(symbols) & self._kline_symbols
        if not rm_syms:
            return
        self._kline_symbols -= rm_syms
        if self._public_ws:
            args = [{"symbol": sym, "ch": "market_kline_1min"} for sym in rm_syms]
            self._public_ws.send(json.dumps({"op": "unsubscribe", "args": args}))

    def subscribe_depth(self, symbols: list[str]):
        """Subscribe to depth for symbols."""
        new_syms = set(symbols) - self._depth_symbols
        if not new_syms:
            return
        self._depth_symbols.update(new_syms)
        if self._public_ws:
            self._send_subscribe(self._public_ws, "depth_books", new_syms)

    # --- Accessors ---

    def get_latest_depth(self, symbol: str) -> dict:
        """Get latest cached depth data for a symbol."""
        return self._depth_cache.get(symbol, {})

    def get_candle_buffer(self, symbol: str) -> list:
        """Get buffered candles for a symbol."""
        return self._candle_buffer.get(symbol, [])

    # --- Ping thread ---

    def _ping_loop(self):
        """Send periodic pings to keep connections alive."""
        while self._running:
            try:
                ts = int(time.time())
                ping_msg = json.dumps({"op": "ping", "ping": ts})
                if self._public_ws:
                    self._public_ws.send(ping_msg)
                if self._private_ws:
                    self._private_ws.send(ping_msg)
            except Exception:
                pass
            time.sleep(PING_INTERVAL)

    # --- Start/Stop ---

    def _start_public(self):
        self._public_ws = websocket.WebSocketApp(
            PUBLIC_WS_URL,
            on_open=self._on_public_open,
            on_message=self._on_public_message,
            on_error=self._on_public_error,
            on_close=self._on_public_close,
        )
        self._public_thread = threading.Thread(
            target=self._public_ws.run_forever,
            kwargs={"ping_interval": 0},  # we handle pings manually
            daemon=True,
        )
        self._public_thread.start()

    def _start_private(self):
        if not self.api_key:
            return
        self._private_ws = websocket.WebSocketApp(
            PRIVATE_WS_URL,
            on_open=self._on_private_open,
            on_message=self._on_private_message,
            on_error=self._on_private_error,
            on_close=self._on_private_close,
        )
        self._private_thread = threading.Thread(
            target=self._private_ws.run_forever,
            kwargs={"ping_interval": 0},
            daemon=True,
        )
        self._private_thread.start()

    def start(self):
        """Start WebSocket connections."""
        self._running = True
        self._start_public()
        self._start_private()

        # Start ping thread
        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()

        logger.info("WebSocket client started (public + private)")

    def stop(self):
        """Stop all WebSocket connections."""
        self._running = False
        if self._public_ws:
            self._public_ws.close()
        if self._private_ws:
            self._private_ws.close()
        logger.info("WebSocket client stopped")

    def is_connected(self) -> bool:
        return (self._public_thread is not None and
                self._public_thread.is_alive())
