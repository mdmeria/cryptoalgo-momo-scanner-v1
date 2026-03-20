#!/usr/bin/env python3
"""
Depth Wall Bounce Strategy

Same wall detection as depth strategy, but waits for price to touch a wall
and bounce before entering.

Entry logic:
  - Find significant support/resistance walls from order book
  - LONG: last candle's wick touches support wall AND closes above it
         → enter at wall price, SL below wall, TP at resistance wall
  - SHORT: last candle's wick touches resistance wall AND closes below it
         → enter at wall price, SL above wall, TP at support wall

Filters:
  - SL must be >= 1% from entry
  - TP:SL ratio must be > 1.0
  - Wall must meet min strength/USD thresholds
  - Excluded symbols (majors, commodities)

Position management:
  - 75% TP rule: if price reaches 75% of TP distance and reverses, close early
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from depth_tp_sl_analyzer import cluster_walls
from live_data_collector import analyze_depth
from strategy_depth import (
    DEPTH_EXCLUDED_SYMBOLS,
    DepthStrategySettings,
    evaluate_zct_alignment,
)


@dataclass
class DepthBounceSettings:
    """Settings for the depth wall bounce strategy."""
    min_sl_pct: float = 1.0           # Minimum SL distance %
    max_sl_pct: float = 3.0           # Maximum SL distance %
    min_rr: float = 1.0               # Minimum TP:SL ratio
    min_wall_strength: float = 3.0    # Min wall strength (x avg) for entry wall
    min_wall_usd: float = 5000.0      # Min wall USD value
    touch_tolerance_pct: float = 0.15 # How close wick must get to wall (% of price)
    tp_buffer_pct: float = 0.1        # Place TP this % before the TP wall
    sl_buffer_pct: float = 0.1        # Place SL this % beyond the entry wall
    tp_close_pct: float = 75.0        # Close if price reached this % of TP without hitting
    cooldown_bars: int = 30           # Min bars between trades on same symbol


def check_depth_bounce_setup(depth_data: dict, df: pd.DataFrame,
                              cfg: DepthBounceSettings = None) -> dict:
    """
    Check if the last candle touched a wall and bounced.

    Args:
        depth_data: Raw order book data
        df: 1m candle DataFrame (needs at least last bar with OHLC)
        cfg: Strategy settings

    Returns dict with 'passed', 'reason' (if failed), and trade details if passed.
    """
    if cfg is None:
        cfg = DepthBounceSettings()

    if not depth_data:
        return {"passed": False, "reason": "no_depth_data"}

    asks = depth_data.get("asks", [])
    bids = depth_data.get("bids", [])
    if not asks or not bids:
        return {"passed": False, "reason": "empty_order_book"}

    if len(df) < 2:
        return {"passed": False, "reason": "insufficient_bars"}

    # Get last bar
    last = df.iloc[-1]
    current_price = float(last["close"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_open = float(last["open"])

    analysis = analyze_depth(depth_data, current_price)

    # Get walls
    res_walls = analysis.get("tp_sl_walls_resistance", [])
    sup_walls = analysis.get("tp_sl_walls_support", [])

    # Cluster walls
    res_clusters = cluster_walls(res_walls)
    sup_clusters = cluster_walls(sup_walls)

    # --- Check for support wall bounce (LONG setup) ---
    long_setup = _check_wall_bounce(
        sup_clusters, res_clusters, current_price,
        last_high, last_low, last_open, "long", cfg)

    # --- Check for resistance wall bounce (SHORT setup) ---
    short_setup = _check_wall_bounce(
        res_clusters, sup_clusters, current_price,
        last_high, last_low, last_open, "short", cfg)

    # Pick the better setup if both triggered
    if long_setup and short_setup:
        # Prefer higher RR
        if long_setup["rr"] >= short_setup["rr"]:
            return long_setup
        return short_setup
    elif long_setup:
        return long_setup
    elif short_setup:
        return short_setup

    return {"passed": False, "reason": "no_wall_bounce"}


def _check_wall_bounce(entry_clusters: list[dict], tp_clusters: list[dict],
                       current_price: float, last_high: float, last_low: float,
                       last_open: float, side: str,
                       cfg: DepthBounceSettings) -> Optional[dict]:
    """
    Check if last candle bounced off an entry-side wall.

    For long: entry_clusters = support walls, tp_clusters = resistance walls
              wick must touch support wall, close above it
    For short: entry_clusters = resistance walls, tp_clusters = support walls
               wick must touch resistance wall, close below it
    """
    is_long = side == "long"

    for entry_wall in entry_clusters:
        # Wall must meet min strength
        if entry_wall["strength"] < cfg.min_wall_strength:
            continue
        if entry_wall["usd_value"] < cfg.min_wall_usd:
            continue

        wall_price = entry_wall["price"]
        wall_dist_pct = entry_wall["dist_pct"]

        # Wall must be within SL range
        if wall_dist_pct < 0.3 or wall_dist_pct > cfg.max_sl_pct:
            continue

        # Check if wick touched the wall
        tolerance = current_price * cfg.touch_tolerance_pct / 100

        if is_long:
            # Wick must reach down to wall level (within tolerance)
            touched = last_low <= wall_price + tolerance
            # But close must be ABOVE the wall
            closed_back = current_price > wall_price
        else:
            # Wick must reach up to wall level (within tolerance)
            touched = last_high >= wall_price - tolerance
            # But close must be BELOW the wall
            closed_back = current_price < wall_price

        if not touched or not closed_back:
            continue

        # Found a bounce! Now find TP wall on the opposite side
        tp_wall = _find_tp_wall(tp_clusters, cfg, wall_dist_pct)
        if tp_wall is None:
            continue

        # Calculate entry at wall price, SL beyond wall, TP at opposite wall
        entry_price = wall_price  # enter at the wall price

        if is_long:
            sl_price = entry_wall["price_lo"] * (1 - cfg.sl_buffer_pct / 100)
            tp_price = tp_wall["price_lo"] * (1 - cfg.tp_buffer_pct / 100)
            sl_pct = (entry_price - sl_price) / entry_price * 100
            tp_pct = (tp_price - entry_price) / entry_price * 100
        else:
            sl_price = entry_wall["price_hi"] * (1 + cfg.sl_buffer_pct / 100)
            tp_price = tp_wall["price_hi"] * (1 + cfg.tp_buffer_pct / 100)
            sl_pct = (sl_price - entry_price) / entry_price * 100
            tp_pct = (entry_price - tp_price) / entry_price * 100

        # Validate
        if sl_pct < cfg.min_sl_pct:
            continue
        if sl_pct > cfg.max_sl_pct:
            continue
        if tp_pct <= 0:
            continue

        rr = tp_pct / sl_pct if sl_pct > 0 else 0
        if rr < cfg.min_rr:
            continue

        return {
            "passed": True,
            "side": side,
            "entry": round(entry_price, 8),
            "sl": round(sl_price, 8),
            "tp": round(tp_price, 8),
            "sl_pct": round(sl_pct, 3),
            "tp_pct": round(tp_pct, 3),
            "rr": round(rr, 2),
            "entry_wall_usd": round(entry_wall["usd_value"], 2),
            "entry_wall_strength": entry_wall["strength"],
            "tp_wall_usd": round(tp_wall["usd_value"], 2),
            "tp_wall_strength": tp_wall["strength"],
            "bounce_type": "support" if is_long else "resistance",
            "wall_price": round(wall_price, 8),
            "wick_price": round(last_low if is_long else last_high, 8),
        }

    return None


def _find_tp_wall(tp_clusters: list[dict], cfg: DepthBounceSettings,
                  sl_dist_pct: float) -> Optional[dict]:
    """Find the best wall for TP placement."""
    for wall in tp_clusters:
        # TP must be far enough for RR
        if wall["dist_pct"] < sl_dist_pct * cfg.min_rr:
            continue
        if wall["usd_value"] < cfg.min_wall_usd * 0.5:
            continue
        return wall
    return None


def check_75pct_tp_rule(position: dict, current_high: float,
                        current_low: float) -> bool:
    """
    Check if the 75% TP rule should close this position.

    Returns True if position should be closed early.

    Logic: track if price ever reached 75%+ of the way to TP.
    If it did and then the current bar pulls back, close.
    """
    entry = position["entry"]
    tp = position["tp"]
    side = position["side"]
    tp_close_pct = 75.0  # hardcoded, matches DepthBounceSettings default

    if side == "long":
        tp_dist = tp - entry
        threshold_price = entry + tp_dist * (tp_close_pct / 100)
        # Check if high reached 75% of TP
        if current_high >= threshold_price:
            # Price reached 75% but didn't hit TP — and current bar pulled back
            if current_low < threshold_price:
                return True
    else:  # short
        tp_dist = entry - tp
        threshold_price = entry - tp_dist * (tp_close_pct / 100)
        if current_low <= threshold_price:
            if current_high > threshold_price:
                return True

    return False
