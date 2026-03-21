#!/usr/bin/env python3
"""
Bitunix WebSocket Client — Based on official SDK pattern.

Uses asyncio + websockets library (NOT websocket-client).
Handles public (kline, depth, ticker) and private (position, order) channels.

Public channels: depth_book1, trade, ticker, market_kline_1min
Private channels: position, order, balance

Usage:
    client = BitunixWS(api_key, api_secret)
    client.on_kline = my_handler
    await client.start()
    await client.subscribe_kline(["BTCUSDT"])
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import ssl
import string
import time
from typing import Callable, Optional

import websockets

logger = logging.getLogger("ws_client")

PUBLIC_WS_URL = "wss://fapi.bitunix.com/public/"
PRIVATE_WS_URL = "wss://fapi.bitunix.com/private/"
HEARTBEAT_INTERVAL = 3  # seconds — matches official SDK
RECONNECT_INTERVAL = 5  # seconds


# ---------------------------------------------------------------------------
# Authentication (matches official open_api_ws_sign.py)
# ---------------------------------------------------------------------------

def _generate_nonce() -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))


def _generate_sign(nonce: str, timestamp: str, api_key: str, secret_key: str) -> str:
    """SHA256(SHA256(nonce + timestamp + apiKey) + secretKey)"""
    digest_input = nonce + timestamp + api_key
    digest = hashlib.sha256(digest_input.encode()).hexdigest()
    sign = hashlib.sha256((digest + secret_key).encode()).hexdigest()
    return sign


def _get_ws_auth(api_key: str, secret_key: str) -> dict:
    nonce = _generate_nonce()
    timestamp = str(int(time.time()))  # seconds, NOT milliseconds
    sign = _generate_sign(nonce, timestamp, api_key, secret_key)
    return {
        "apiKey": api_key,
        "timestamp": int(timestamp),
        "nonce": nonce,
        "sign": sign,
    }


# ---------------------------------------------------------------------------
# SSL Context (matches official — no cert verification)
# ---------------------------------------------------------------------------

def _ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ---------------------------------------------------------------------------
# WebSocket Client
# ---------------------------------------------------------------------------

class BitunixWS:
    """Async WebSocket client for Bitunix public + private channels."""

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret

        # Connection state
        self._public_ws = None
        self._private_ws = None
        self._public_connected = False
        self._private_connected = False
        self._running = False

        # Subscription tracking
        self._kline_symbols: set[str] = set()
        self._depth_symbols: set[str] = set()

        # Callbacks (called from async context)
        self.on_kline: Optional[Callable] = None       # (symbol, candle_dict)
        self.on_depth: Optional[Callable] = None        # (symbol, depth_dict)
        self.on_position: Optional[Callable] = None     # (data_dict)
        self.on_order: Optional[Callable] = None        # (data_dict)
        self.on_ticker: Optional[Callable] = None       # (symbol, data_dict)

        # Data caches
        self._depth_cache: dict[str, dict] = {}
        self._candle_buffer: dict[str, list] = {}

        # Connection tracking
        self._last_disconnect_ts: float = 0
        self._total_disconnects: int = 0

        # Event loop reference
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # --- Public WS ---

    async def _public_connect(self):
        """Connect and maintain public WebSocket."""
        while self._running:
            try:
                async with websockets.connect(
                    PUBLIC_WS_URL,
                    ssl=_ssl_context(),
                    ping_interval=None,  # disable protocol-level ping
                    ping_timeout=5,
                    close_timeout=5,
                ) as ws:
                    self._public_ws = ws
                    self._public_connected = True
                    if self._last_disconnect_ts > 0:
                        downtime = time.time() - self._last_disconnect_ts
                        logger.info("Public WS RECONNECTED (downtime=%.1fs, total_disconnects=%d)",
                                    downtime, self._total_disconnects)
                    else:
                        logger.info("Public WS connected")

                    # Re-subscribe
                    await self._resubscribe_public()

                    # Start ping task
                    ping_task = asyncio.create_task(
                        self._ping_loop(ws, "public"))

                    try:
                        async for message in ws:
                            await self._handle_public_message(message)
                    except websockets.exceptions.ConnectionClosedError:
                        self._last_disconnect_ts = time.time()
                        self._total_disconnects += 1
                        logger.warning("Public WS DISCONNECTED (total=%d)", self._total_disconnects)
                    except Exception as e:
                        logger.error("Public WS error: %s", e)
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass

                self._public_connected = False
                if self._running:
                    logger.info("Public WS reconnecting in %ds...", RECONNECT_INTERVAL)
                    await asyncio.sleep(RECONNECT_INTERVAL)

            except Exception as e:
                logger.error("Public WS connect failed: %s", e)
                self._public_connected = False
                if self._running:
                    await asyncio.sleep(RECONNECT_INTERVAL)

    async def _handle_public_message(self, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        # Heartbeat response
        if data.get("op") == "ping":
            logger.debug("Public ping response: %s", data)
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
                "volume": float(kline_data.get("q", 0)),
            }
            # Update buffer
            if symbol not in self._candle_buffer:
                self._candle_buffer[symbol] = []
            self._candle_buffer[symbol].append(candle)
            if len(self._candle_buffer[symbol]) > 1000:
                self._candle_buffer[symbol] = self._candle_buffer[symbol][-1000:]

            self.on_kline(symbol, candle)

        elif "depth" in ch and self.on_depth:
            depth_data = data.get("data", data)
            # WS uses "a"/"b", REST uses "asks"/"bids" — handle both
            asks = depth_data.get("a") or depth_data.get("asks", [])
            bids = depth_data.get("b") or depth_data.get("bids", [])
            self._depth_cache[symbol] = {
                "asks": asks,
                "bids": bids,
                "ts": data.get("ts", 0),
            }
            self.on_depth(symbol, self._depth_cache[symbol])

        elif ch == "ticker" and self.on_ticker:
            self.on_ticker(symbol, data.get("data", {}))

    async def _resubscribe_public(self):
        """Re-subscribe to all tracked channels after reconnect."""
        if not self._public_ws:
            return
        args = []
        for sym in self._kline_symbols:
            args.append({"symbol": sym, "ch": "market_kline_1min"})
        for sym in self._depth_symbols:
            args.append({"symbol": sym, "ch": "depth_book15"})
        if args:
            await self._public_ws.send(json.dumps({"op": "subscribe", "args": args}))
            logger.info("Subscribed public: %d kline, %d depth",
                        len(self._kline_symbols), len(self._depth_symbols))

    # --- Private WS ---

    async def _private_connect(self):
        """Connect, authenticate, and maintain private WebSocket."""
        if not self.api_key:
            return

        while self._running:
            try:
                async with websockets.connect(
                    PRIVATE_WS_URL,
                    ssl=_ssl_context(),
                    ping_interval=None,
                    ping_timeout=5,
                    close_timeout=5,
                ) as ws:
                    self._private_ws = ws
                    self._private_connected = True
                    logger.info("Private WS connected")

                    # Authenticate
                    auth_data = _get_ws_auth(self.api_key, self.api_secret)
                    await ws.send(json.dumps({
                        "op": "login",
                        "args": [auth_data],
                    }))

                    # Start ping
                    ping_task = asyncio.create_task(
                        self._ping_loop(ws, "private"))

                    try:
                        async for message in ws:
                            await self._handle_private_message(message)
                    except websockets.exceptions.ConnectionClosedError:
                        logger.warning("Private WS closed by server")
                    except Exception as e:
                        logger.error("Private WS error: %s", e)
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass

                self._private_connected = False
                if self._running:
                    logger.info("Private WS reconnecting in %ds...", RECONNECT_INTERVAL)
                    await asyncio.sleep(RECONNECT_INTERVAL)

            except Exception as e:
                logger.error("Private WS connect failed: %s", e)
                self._private_connected = False
                if self._running:
                    await asyncio.sleep(RECONNECT_INTERVAL)

    async def _handle_private_message(self, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        # Log everything from private WS for debugging
        logger.info("Private WS msg: %s", str(message)[:300])

        # Heartbeat
        if data.get("op") == "ping":
            return

        # Login response — check multiple possible formats
        op = data.get("op", "")
        if op == "login" or "login" in str(message).lower():
            logger.info("Private WS login response: %s", json.dumps(data)[:300])
            code = data.get("code")
            if code == 0 or code == "0" or data.get("msg") == "Success":
                logger.info("Private WS authenticated")
                # Subscribe to private channels
                await self._private_ws.send(json.dumps({
                    "op": "subscribe",
                    "args": [
                        {"ch": "position"},
                        {"ch": "order"},
                    ],
                }))
                logger.info("Subscribed to private channels: position, order")
            else:
                logger.error("Private WS auth failed: %s", data.get("msg"))
            return

        ch = data.get("ch", "")

        if ch == "position" and self.on_position:
            self.on_position(data.get("data", {}))

        elif ch == "order" and self.on_order:
            self.on_order(data.get("data", {}))

    # --- Ping/Pong (matches official 3-second interval) ---

    async def _ping_loop(self, ws, label: str):
        """Send heartbeat every 3 seconds — matches official SDK."""
        while True:
            try:
                msg = json.dumps({"op": "ping", "ping": int(time.time())})
                await ws.send(msg)
                await asyncio.sleep(HEARTBEAT_INTERVAL)
            except websockets.exceptions.ConnectionClosedError:
                logger.debug("%s ping: connection closed", label)
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("%s ping error: %s", label, e)
                break

    # --- Subscription Management ---

    async def subscribe_kline(self, symbols: list[str]):
        new = set(symbols) - self._kline_symbols
        if not new:
            return
        self._kline_symbols.update(new)
        if self._public_ws and self._public_connected:
            args = [{"symbol": sym, "ch": "market_kline_1min"} for sym in new]
            await self._public_ws.send(json.dumps({"op": "subscribe", "args": args}))
            logger.info("Subscribed kline: %d new symbols", len(new))

    async def subscribe_depth(self, symbols: list[str]):
        new = set(symbols) - self._depth_symbols
        if not new:
            return
        self._depth_symbols.update(new)
        if self._public_ws and self._public_connected:
            args = [{"symbol": sym, "ch": "depth_book1"} for sym in new]
            await self._public_ws.send(json.dumps({"op": "subscribe", "args": args}))
            logger.info("Subscribed depth: %d new symbols", len(new))

    # --- Accessors ---

    def get_latest_depth(self, symbol: str) -> dict:
        return self._depth_cache.get(symbol, {})

    def get_candle_buffer(self, symbol: str) -> list:
        return self._candle_buffer.get(symbol, [])

    # --- Start/Stop ---

    async def start(self):
        """Start WebSocket connections. Private WS disabled for now."""
        self._running = True
        self._loop = asyncio.get_event_loop()

        tasks = [asyncio.create_task(self._public_connect())]
        # Private WS disabled — auth issue needs investigation
        # REST sync handles position/order updates every 10s instead
        # if self.api_key:
        #     tasks.append(asyncio.create_task(self._private_connect()))

        logger.info("WebSocket client starting (public only — REST sync for positions)")

        # Run until stopped
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self):
        """Signal stop."""
        self._running = False
        logger.info("WebSocket client stopping")

    @property
    def is_connected(self) -> bool:
        return self._public_connected
