"""Entry, SL, and TP calculation for momentum setups."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests


@dataclass
class OrderSetup:
    """Trade entry, SL, and TP specification."""

    symbol: str
    direction: str  # "long" or "short"
    timestamp_utc: datetime
    entry_price: float
    sl_price: float
    tp_price: float
    risk_amount: float  # Entry - SL for long
    reward_amount: float  # TP - Entry for long
    rr_ratio: float  # Reward / Risk
    entry_distance_pct: float  # TP distance from entry
    order_types: str  # "entry:LIMIT|sl:STOP_MARKET|tp:LIMIT"


def _fetch_klines(
    symbol: str, interval: str, limit: int, end_time_ms: int | None = None
) -> pd.DataFrame:
    """Fetch klines from Binance Vision."""
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)

    try:
        response = requests.get(url, params=params, timeout=8)
        if response.status_code != 200:
            return pd.DataFrame()
        raw = response.json()
        if not isinstance(raw, list) or len(raw) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            [
                {
                    "timestamp": pd.to_datetime(k[0], unit="ms", utc=True),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
                for k in raw
            ]
        )
        return df.set_index("timestamp")
    except Exception:
        return pd.DataFrame()


def _find_swing_low_2h(bars: pd.DataFrame, lookback: int = 4) -> float:
    """Find swing low in 2h window (4 x 30m bars). Return lowest low."""
    if len(bars) < lookback:
        return bars["low"].min()
    return bars["low"].iloc[-lookback:].min()


def _find_swing_high_2h(bars: pd.DataFrame, lookback: int = 4) -> float:
    """Find swing high in 2h window (4 x 30m bars). Return highest high."""
    if len(bars) < lookback:
        return bars["high"].max()
    return bars["high"].iloc[-lookback:].max()


def _find_second_swing_low(bars: pd.DataFrame) -> float:
    """Find second-most-recent swing low (more conservative SL placement)."""
    if len(bars) < 4:
        return bars["low"].min()
    # First swing low from last 4 bars
    first_swing = bars["low"].iloc[-4:].min()
    # Second swing from bars before that
    if len(bars) >= 8:
        second_swing = bars["low"].iloc[-8:-4].min()
        return min(first_swing, second_swing) if second_swing < first_swing else second_swing
    return first_swing


def _find_second_swing_high(bars: pd.DataFrame) -> float:
    """Find second-most-recent swing high (more conservative SL placement)."""
    if len(bars) < 4:
        return bars["high"].max()
    # First swing high from last 4 bars
    first_swing = bars["high"].iloc[-4:].max()
    # Second swing from bars before that
    if len(bars) >= 8:
        second_swing = bars["high"].iloc[-8:-4].max()
        return max(first_swing, second_swing) if second_swing > first_swing else second_swing
    return first_swing


def _find_resistance_level(bars_1m: pd.DataFrame, direction: str, entry_price: float) -> float:
    """
    Find nearby resistance (for longs) or support (for shorts) from recent 1m bars.
    Look at last 10-15 minutes of price action (10-15 x 1m bars).
    """
    if bars_1m.empty or len(bars_1m) < 5:
        if direction == "long":
            return entry_price * 1.03  # Default 3% above
        else:
            return entry_price * 0.97  # Default 3% below
    
    if direction == "long":
        # Find highest high in last 15 minutes above entry (recent swing high)
        highs = bars_1m["high"].iloc[-15:] if len(bars_1m) >= 15 else bars_1m["high"]
        resistances = highs[highs > entry_price * 1.001]  # Slightly above entry
        if len(resistances) > 0:
            return resistances.min()  # First resistance above entry
        return entry_price * 1.03  # Fallback to 3%
    else:
        # Find lowest low in last 15 minutes below entry (recent swing low)
        lows = bars_1m["low"].iloc[-15:] if len(bars_1m) >= 15 else bars_1m["low"]
        supports = lows[lows < entry_price * 0.999]  # Slightly below entry
        if len(supports) > 0:
            return supports.max()  # First support below entry
        return entry_price * 0.97  # Fallback to 3%


def _calculate_atr_1h(bars_1h: pd.DataFrame, length: int = 14) -> float:
    """Calculate ATR from 1h bars."""
    if bars_1h.empty or len(bars_1h) < length:
        return 0.0
    
    bars_1h = bars_1h.copy()
    bars_1h["tr"] = np.maximum(
        bars_1h["high"] - bars_1h["low"],
        np.maximum(
            abs(bars_1h["high"] - bars_1h["close"].shift(1)),
            abs(bars_1h["low"] - bars_1h["close"].shift(1)),
        ),
    )
    atr = bars_1h["tr"].rolling(length).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else 0.0


def calculate_order_setup(
    symbol: str,
    direction: str,
    timestamp_utc: str,  # ISO format
    entry_price: float,
) -> OrderSetup:
    """
    Calculate SL/TP prices with structural levels.
    
    Strategy:
    - SL: Second swing low/high + 0.5 ATR buffer (conservative)
    - TP: Before key resistance/support, constrained to 1-3.5%
    - Orders: Entry/TP as LIMIT, SL as STOP_MARKET
    """
    
    ts_utc = pd.to_datetime(timestamp_utc, utc=True)
    ts_ms = int(ts_utc.timestamp() * 1000)
    
    # Fetch 30m bars for swing structure (8 bars = 4h lookback)
    bars_30m = _fetch_klines(symbol, "30m", limit=8, end_time_ms=ts_ms)
    
    # Fetch 1h bars for ATR calculation
    bars_1h = _fetch_klines(symbol, "1h", limit=20, end_time_ms=ts_ms)
    
    # Fetch 1m bars for resistance identification (last 20 bars ≈ 20 minutes)
    bars_1m = _fetch_klines(symbol, "1m", limit=20, end_time_ms=ts_ms)
    
    # Calculate second swing levels (more conservative)
    if not bars_30m.empty and len(bars_30m) >= 4:
        if direction == "long":
            swing_level = _find_second_swing_low(bars_30m)
            resistance_level = _find_resistance_level(bars_1m, direction, entry_price)
        else:  # short
            swing_level = _find_second_swing_high(bars_30m)
            resistance_level = _find_resistance_level(bars_1m, direction, entry_price)
    else:
        # Fallback if we can't fetch bars
        if direction == "long":
            swing_level = entry_price * 0.97  # Assume 3% below
            resistance_level = entry_price * 1.03
        else:
            swing_level = entry_price * 1.03  # Assume 3% above
            resistance_level = entry_price * 0.97
    
    # Calculate ATR buffer for SL
    atr_1h = _calculate_atr_1h(bars_1h, length=14)
    atr_buffer = atr_1h * 0.5 if atr_1h > 0 else entry_price * 0.005
    
    # Ensure swing level is reasonable
    if direction == "long":
        # Swing low should be below entry
        if swing_level >= entry_price:
            swing_level = entry_price * 0.96
        if swing_level < entry_price * 0.90:  # More than 10% below seems wrong
            swing_level = entry_price * 0.94
    else:  # short
        # Swing high should be above entry
        if swing_level <= entry_price:
            swing_level = entry_price * 1.04
        if swing_level > entry_price * 1.10:  # More than 10% above seems wrong
            swing_level = entry_price * 1.06
    
    # Calculate SL with buffer
    if direction == "long":
        sl_price = swing_level - atr_buffer
        sl_price = min(sl_price, entry_price * 0.97)  # Cap at entry-3%
        risk_amount = entry_price - sl_price
        
        # TP: Before resistance, constrained to 1-3.5% range
        tp_to_resistance = resistance_level - entry_price
        tp_distance_pct = (tp_to_resistance / entry_price) * 100
        
        # Constrain to 1-3.5%
        min_tp_pct = 1.0
        max_tp_pct = 3.5
        
        if tp_distance_pct > max_tp_pct:
            # Resistance is far away, use max 3.5%
            tp_distance = entry_price * (max_tp_pct / 100)
        elif tp_distance_pct < min_tp_pct:
            # Resistance is too close, use min 1%
            tp_distance = entry_price * (min_tp_pct / 100)
        else:
            # Use actual distance to resistance
            tp_distance = tp_to_resistance
        
        tp_price = entry_price + tp_distance
        reward_amount = tp_price - entry_price
        
    else:  # short
        sl_price = swing_level + atr_buffer
        sl_price = max(sl_price, entry_price * 1.03)  # Cap at entry+3%
        risk_amount = sl_price - entry_price
        
        # TP: Before support, constrained to 1-3.5% range
        tp_to_support = entry_price - resistance_level
        tp_distance_pct = (tp_to_support / entry_price) * 100
        
        # Constrain to 1-3.5%
        min_tp_pct = 1.0
        max_tp_pct = 3.5
        
        if tp_distance_pct > max_tp_pct:
            # Support is far away, use max 3.5%
            tp_distance = entry_price * (max_tp_pct / 100)
        elif tp_distance_pct < min_tp_pct:
            # Support is too close, use min 1%
            tp_distance = entry_price * (min_tp_pct / 100)
        else:
            # Use actual distance to support
            tp_distance = tp_to_support
        
        tp_price = entry_price - tp_distance
        reward_amount = entry_price - tp_price
    
    rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0.0
    entry_distance_pct = (abs(tp_price - entry_price) / entry_price) * 100
    
    return OrderSetup(
        symbol=symbol,
        direction=direction,
        timestamp_utc=ts_utc,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        risk_amount=risk_amount,
        reward_amount=reward_amount,
        rr_ratio=rr_ratio,
        entry_distance_pct=entry_distance_pct,
        order_types="entry:LIMIT|sl:STOP_MARKET|tp:LIMIT",
    )


if __name__ == "__main__":
    # Test with one entry
    setup = calculate_order_setup(
        symbol="FLOWUSDT",
        direction="long",
        timestamp_utc="2026-03-06T05:00:00+00:00",
        entry_price=1.5331,
    )
    print(f"Symbol: {setup.symbol}")
    print(f"Direction: {setup.direction}")
    print(f"Entry: ${setup.entry_price:.8f}")
    print(f"SL: ${setup.sl_price:.8f} (risk ${setup.risk_amount:.8f})")
    print(f"TP: ${setup.tp_price:.8f} (reward ${setup.reward_amount:.8f})")
    print(f"R:R Ratio: {setup.rr_ratio:.2f}")
    print(f"TP distance: {setup.entry_distance_pct:.2f}%")
