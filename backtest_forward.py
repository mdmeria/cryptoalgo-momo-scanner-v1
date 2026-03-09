"""Forward backtest engine for momentum setups with 4h+ lookback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests


@dataclass
class BacktestResult:
    """Results from 4h forward backtest."""

    symbol: str
    timestamp_utc: datetime
    entry_price: float
    sl_price: float
    tp_price: float
    direction: str
    
    # 1h results
    sl_hit_1h: bool
    tp_hit_1h: bool
    mae_1h_pct: float  # Max adverse excursion %
    mfe_1h_pct: float  # Max favorable excursion %
    pnl_1h_pct: float
    time_to_exit_1h: int  # bars (0 if no exit)
    
    # 2h results
    sl_hit_2h: bool
    tp_hit_2h: bool
    mae_2h_pct: float
    mfe_2h_pct: float
    pnl_2h_pct: float
    time_to_exit_2h: int
    
    # 4h results
    sl_hit_4h: bool
    tp_hit_4h: bool
    mae_4h_pct: float
    mfe_4h_pct: float
    pnl_4h_pct: float
    time_to_exit_4h: int
    
    # Final outcome
    trade_outcome: str  # "TP_HIT" | "SL_HIT" | "OPEN_4H"


def _fetch_1h_bars(
    symbol: str, 
    start_time_ms: int, 
    limit: int = 5  # 5 hours forward
) -> pd.DataFrame:
    """Fetch 1h bars starting from start_time_ms."""
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": "1h",
        "limit": limit,
        "startTime": int(start_time_ms),
    }
    
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
        return df
    except Exception:
        return pd.DataFrame()


def _check_exit(
    bars: pd.DataFrame,
    direction: str,
    sl_price: float,
    tp_price: float,
) -> tuple[bool, bool, int]:
    """
    Check if SL or TP was hit. Returns (sl_hit, tp_hit, bars_to_exit).
    bars_to_exit is 0 if no exit by end of period.
    """
    if bars.empty:
        return False, False, 0
    
    for i, bar in bars.iterrows():
        if direction == "long":
            # SL hit if low <= sl_price
            if bar["low"] <= sl_price:
                return True, False, i + 1
            # TP hit if high >= tp_price
            if bar["high"] >= tp_price:
                return False, True, i + 1
        else:  # short
            # SL hit if high >= sl_price
            if bar["high"] >= sl_price:
                return True, False, i + 1
            # TP hit if low <= tp_price
            if bar["low"] <= tp_price:
                return False, True, i + 1
    
    return False, False, 0


def _calculate_mae_mfe(
    bars: pd.DataFrame,
    direction: str,
    entry_price: float,
) -> tuple[float, float]:
    """
    Calculate Max Adverse Excursion and Max Favorable Excursion as %.
    """
    if bars.empty:
        return 0.0, 0.0
    
    if direction == "long":
        # MAE: lowest low below entry
        lowest = bars["low"].min()
        mae_pct = ((entry_price - lowest) / entry_price) * 100
        
        # MFE: highest high above entry
        highest = bars["high"].max()
        mfe_pct = ((highest - entry_price) / entry_price) * 100
    else:  # short
        # MAE: highest high above entry
        highest = bars["high"].max()
        mae_pct = ((highest - entry_price) / entry_price) * 100
        
        # MFE: lowest low below entry
        lowest = bars["low"].min()
        mfe_pct = ((entry_price - lowest) / entry_price) * 100
    
    mae_pct = max(mae_pct, 0.0)  # MAE should be >= 0
    mfe_pct = max(mfe_pct, 0.0)  # MFE should be >= 0
    
    return mae_pct, mfe_pct


def _calculate_pnl(
    close_price: float,
    entry_price: float,
    direction: str,
) -> float:
    """Calculate P&L percentage from entry."""
    if direction == "long":
        pnl_pct = ((close_price - entry_price) / entry_price) * 100
    else:  # short
        pnl_pct = ((entry_price - close_price) / entry_price) * 100
    return pnl_pct


def backtest_forward(
    symbol: str,
    timestamp_utc: str,  # ISO format
    entry_price: float,
    sl_price: float,
    tp_price: float,
    direction: str,
) -> BacktestResult:
    """
    Backtest entry/SL/TP over 4h+ forward lookback.
    Fetch 5x 1h bars and analyze at 1h/2h/4h intervals.
    """
    
    ts_utc = pd.to_datetime(timestamp_utc, utc=True)
    ts_ms = int(ts_utc.timestamp() * 1000)
    
    # Fetch 5h forward (covers entry + 4h evaluation)
    bars_1h = _fetch_1h_bars(symbol, start_time_ms=ts_ms, limit=5)
    
    # Default results
    result = BacktestResult(
        symbol=symbol,
        timestamp_utc=ts_utc,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        direction=direction,
        sl_hit_1h=False, tp_hit_1h=False, mae_1h_pct=0.0, mfe_1h_pct=0.0,
        pnl_1h_pct=0.0, time_to_exit_1h=0,
        sl_hit_2h=False, tp_hit_2h=False, mae_2h_pct=0.0, mfe_2h_pct=0.0,
        pnl_2h_pct=0.0, time_to_exit_2h=0,
        sl_hit_4h=False, tp_hit_4h=False, mae_4h_pct=0.0, mfe_4h_pct=0.0,
        pnl_4h_pct=0.0, time_to_exit_4h=0,
        trade_outcome="OPEN_4H",
    )
    
    if bars_1h.empty or len(bars_1h) < 2:
        return result
    
    # Analyze at 1h, 2h, 4h intervals
    for interval_bars, attr_prefix in [
        (1, "1h"),
        (2, "2h"),
        (4, "4h"),
    ]:
        if len(bars_1h) < interval_bars:
            continue
        
        bars_interval = bars_1h.iloc[:interval_bars]
        
        # Check for SL/TP hits
        sl_hit, tp_hit, bars_to_exit = _check_exit(
            bars_interval, direction, sl_price, tp_price
        )
        
        # Calculate MAE/MFE
        mae_pct, mfe_pct = _calculate_mae_mfe(bars_interval, direction, entry_price)
        
        # Calculate PnL at close of interval
        closing_price = bars_interval["close"].iloc[-1]
        pnl_pct = _calculate_pnl(closing_price, entry_price, direction)
        
        # Set results
        setattr(result, f"sl_hit_{attr_prefix}", sl_hit)
        setattr(result, f"tp_hit_{attr_prefix}", tp_hit)
        setattr(result, f"mae_{attr_prefix}_pct", mae_pct)
        setattr(result, f"mfe_{attr_prefix}_pct", mfe_pct)
        setattr(result, f"pnl_{attr_prefix}_pct", pnl_pct)
        setattr(result, f"time_to_exit_{attr_prefix}", bars_to_exit)
        
        # Update trade outcome when first exit found
        if result.trade_outcome == "OPEN_4H":
            if sl_hit:
                result.trade_outcome = "SL_HIT"
            elif tp_hit:
                result.trade_outcome = "TP_HIT"
    
    return result


if __name__ == "__main__":
    # Test backtest
    res = backtest_forward(
        symbol="FLOWUSDT",
        timestamp_utc="2026-03-06T05:00:00+00:00",
        entry_price=1.5331,
        sl_price=1.5234,
        tp_price=1.5530,
        direction="long",
    )
    print(f"Symbol: {res.symbol}")
    print(f"Entry: ${res.entry_price:.8f}")
    print(f"1h: SL={res.sl_hit_1h} TP={res.tp_hit_1h} MAE={res.mae_1h_pct:.2f}% MFE={res.mfe_1h_pct:.2f}% PnL={res.pnl_1h_pct:.2f}%")
    print(f"2h: SL={res.sl_hit_2h} TP={res.tp_hit_2h} MAE={res.mae_2h_pct:.2f}% MFE={res.mfe_2h_pct:.2f}% PnL={res.pnl_2h_pct:.2f}%")
    print(f"4h: SL={res.sl_hit_4h} TP={res.tp_hit_4h} MAE={res.mae_4h_pct:.2f}% MFE={res.mfe_4h_pct:.2f}% PnL={res.pnl_4h_pct:.2f}%")
    print(f"Outcome: {res.trade_outcome}")
