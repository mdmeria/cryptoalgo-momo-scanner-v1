#!/usr/bin/env python3
"""
Choppy Range Mean Reversion Scanner (ZCT Framework)

Detects MR setups by finding horizontal choppy ranges where:
  1. Price has been chopping in a bounded range for 2+ hours
  2. The entry-side level has been respected/rejected 3+ times
  3. Last touch to the entry-side level was within 30 minutes
  4. DPS scoring: approach quality, volume, choppy duration

Usage:
  python scan_mean_reversion.py \
    --dataset-dir datasets/momo_1m_mar2_mar14 \
    --out-prefix mr_choppy
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class MRSettings:
    # Range detection
    range_min_bars: int = 120           # min bars in range (2 hours)
    range_max_bars: int = 720           # max lookback for range (12 hours)
    range_max_width_pct: float = 4.0    # max range width as % of price
    range_min_width_pct: float = 1.0    # min range width — needs to be wide enough for TP > SL
    range_containment_pct: float = 90.0 # % of bars that must be inside range
    range_bound_touch_pct: float = 0.10 # % tolerance for "touching" a bound

    # Touch counting
    min_touches: int = 3                # min touches on entry-side level
    touch_cluster_bars: int = 15        # min bars between touches (debounce)
    last_touch_max_bars: int = 30       # last touch must be within 30 bars (30 min)

    # Entry trigger
    entry_spike_min_pct: float = 0.15   # min move into level on entry bar
    entry_close_back_bars: int = 3      # bars to wait for close back inside

    # 30 SMMA noise filter
    noise_lookback_bars: int = 480      # 8 hours for MA cross counting
    noise_min_crosses: int = 4          # min 30 SMMA crosses for choppy

    # Volume
    vol_lookback_bars: int = 60         # 1h of volume to assess trend

    # SL/TP
    min_sl_pct: float = 1.0
    max_sl_pct: float = 2.0
    rr: float = 1.0                    # 1:1 RR for MR

    # Cooldown
    cooldown_bars: int = 120            # 2 hours between setups per symbol

    @classmethod
    def from_json(cls, path: str) -> "MRSettings":
        with open(path) as f:
            data = json.load(f)
        obj = cls()
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smma(series: pd.Series, length: int = 30) -> pd.Series:
    """Smoothed Moving Average (SMMA) = EWM with alpha=1/length."""
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def count_smma_crosses(close: np.ndarray, smma: np.ndarray) -> int:
    """Count debounced crosses of price through SMMA30."""
    above = close > smma
    crosses = 0
    for i in range(1, len(above)):
        if above[i] != above[i - 1]:
            crosses += 1
    return crosses


def detect_volume_trend(volumes: np.ndarray) -> str:
    """Classify volume as 'increasing', 'flat', or 'decreasing'."""
    if len(volumes) < 10:
        return "flat"
    x = np.arange(len(volumes), dtype=float)
    y = volumes.astype(float)
    y_mean = np.mean(y)
    if y_mean < 1e-9:
        return "flat"
    slope = np.polyfit(x, y, 1)[0]
    norm_slope = slope / y_mean * len(volumes)
    if norm_slope > 0.3:
        return "increasing"
    elif norm_slope < -0.3:
        return "decreasing"
    else:
        return "flat"


def find_swing_high(highs: np.ndarray, end_idx: int, lookback: int = 30) -> Optional[float]:
    """Find the most recent swing high before end_idx."""
    start = max(0, end_idx - lookback)
    segment = highs[start:end_idx]
    if len(segment) < 3:
        return None
    for i in range(len(segment) - 2, 0, -1):
        if segment[i] >= segment[i - 1] and segment[i] >= segment[i + 1]:
            return float(segment[i])
    return float(np.max(segment))


def find_swing_low(lows: np.ndarray, end_idx: int, lookback: int = 30) -> Optional[float]:
    """Find the most recent swing low before end_idx."""
    start = max(0, end_idx - lookback)
    segment = lows[start:end_idx]
    if len(segment) < 3:
        return None
    for i in range(len(segment) - 2, 0, -1):
        if segment[i] <= segment[i - 1] and segment[i] <= segment[i + 1]:
            return float(segment[i])
    return float(np.min(segment))


# ---------------------------------------------------------------------------
# Choppy Range Detection
# ---------------------------------------------------------------------------

def detect_choppy_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                        end_idx: int, cfg: MRSettings) -> Optional[dict]:
    """
    Look backwards from end_idx to find a horizontal choppy range.

    A range is defined by upper/lower bounds where:
    - Price has been contained within the range for range_containment_pct% of bars
    - Range has lasted at least range_min_bars
    - Range width is between min and max width

    Returns dict with range info or None.
    """
    # Try different lookback windows, prefer longer ranges
    best_range = None

    for lookback in [480, 360, 240, 180, 120]:
        if lookback > end_idx:
            continue

        start = end_idx - lookback
        seg_highs = highs[start:end_idx + 1]
        seg_lows = lows[start:end_idx + 1]
        seg_closes = closes[start:end_idx + 1]
        n = len(seg_highs)

        if n < cfg.range_min_bars:
            continue

        # Find range bounds using percentiles to be robust to spikes
        # Upper bound: 90th percentile of highs (ignoring spike outliers)
        # Lower bound: 10th percentile of lows
        upper = float(np.percentile(seg_highs, 90))
        lower = float(np.percentile(seg_lows, 10))

        if lower <= 0:
            continue

        width_pct = (upper - lower) / lower * 100

        if width_pct > cfg.range_max_width_pct or width_pct < cfg.range_min_width_pct:
            continue

        # Check containment: what % of bars are inside the range?
        tolerance = (upper - lower) * 0.1  # 10% tolerance outside bounds
        inside = 0
        for j in range(n):
            if seg_lows[j] >= lower - tolerance and seg_highs[j] <= upper + tolerance:
                inside += 1

        containment = inside / n * 100
        if containment < cfg.range_containment_pct:
            continue

        # Calculate range midpoint and duration in hours
        mid = (upper + lower) / 2
        duration_hrs = lookback / 60.0

        # Check for directional bias (should be low for choppy)
        net_move_pct = abs(seg_closes[-1] - seg_closes[0]) / seg_closes[0] * 100
        efficiency = net_move_pct / width_pct if width_pct > 0 else 1.0

        if efficiency > 0.6:
            # Too directional, not choppy
            continue

        range_info = {
            "upper": upper,
            "lower": lower,
            "mid": mid,
            "width_pct": round(width_pct, 3),
            "containment_pct": round(containment, 1),
            "duration_bars": lookback,
            "duration_hrs": round(duration_hrs, 1),
            "start_idx": start,
            "end_idx": end_idx,
            "efficiency": round(efficiency, 3),
        }

        if best_range is None or lookback > best_range["duration_bars"]:
            best_range = range_info

    return best_range


def count_level_touches(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                        opens: np.ndarray, start_idx: int, end_idx: int,
                        level: float, side: str,
                        at_level_pct: float = 0.05,
                        debounce_bars: int = 5) -> dict:
    """
    Count how many times price touched/respected a level.

    For upper (resistance / short entry side):
      - Wick touches/crosses level (high >= level) AND body closes below it
      - OR close is right at the level (within at_level_pct%)

    For lower (support / long entry side):
      - Wick touches/crosses level (low <= level) AND body closes above it
      - OR close is right at the level (within at_level_pct%)

    Also checks level integrity: if price body (close) broke through the level
    between touches, the level is considered broken.

    Returns {count, last_touch_idx, touch_indices, broken, break_count}.
    """
    touches = []
    breaks = 0
    last_touch_bar = -999
    at_level_tol = level * at_level_pct / 100.0

    for i in range(start_idx, end_idx + 1):
        if side == "upper":
            # Check if body broke through (closed above level)
            if closes[i] > level + at_level_tol:
                breaks += 1

            if i - last_touch_bar < debounce_bars:
                continue

            wick_touch = highs[i] >= level
            body_below = closes[i] < level
            close_at_level = abs(closes[i] - level) <= at_level_tol

            if (wick_touch and body_below) or close_at_level:
                touches.append(i)
                last_touch_bar = i

        else:  # lower / support
            # Check if body broke through (closed below level)
            if closes[i] < level - at_level_tol:
                breaks += 1

            if i - last_touch_bar < debounce_bars:
                continue

            wick_touch = lows[i] <= level
            body_above = closes[i] > level
            close_at_level = abs(closes[i] - level) <= at_level_tol

            if (wick_touch and body_above) or close_at_level:
                touches.append(i)
                last_touch_bar = i

    n_bars = end_idx - start_idx + 1
    break_pct = breaks / n_bars * 100 if n_bars > 0 else 0

    return {
        "count": len(touches),
        "last_touch_idx": touches[-1] if touches else -1,
        "touch_indices": touches,
        "break_count": breaks,
        "break_pct": round(break_pct, 1),
    }


def check_opposite_bound_intact(highs: np.ndarray, lows: np.ndarray,
                                closes: np.ndarray,
                                last_touch_idx: int, end_idx: int,
                                range_info: dict, entry_side: str) -> dict:
    """
    After the last touch on the entry side, check if the OPPOSITE boundary
    was broken before the entry bar.

    E.g. for a long entry (entry at lower bound): check if upper bound was
    broken between last_touch_idx and end_idx. If so, range character has
    changed and this setup is suspect.

    Also checks for lower-highs/lower-lows (bearish) or higher-highs/
    higher-lows (bullish) pattern after last touch.
    """
    if last_touch_idx < 0 or last_touch_idx >= end_idx:
        return {"opposite_intact": True, "opposite_breaks": 0, "post_touch_trend": "none"}

    seg_highs = highs[last_touch_idx:end_idx + 1]
    seg_lows = lows[last_touch_idx:end_idx + 1]
    seg_closes = closes[last_touch_idx:end_idx + 1]

    opp_breaks = 0

    if entry_side == "lower":
        # Entry at lower bound (long) — check if upper bound was broken
        upper = range_info["upper"]
        for i in range(len(seg_closes)):
            if seg_closes[i] > upper:
                opp_breaks += 1
    else:
        # Entry at upper bound (short) — check if lower bound was broken
        lower = range_info["lower"]
        for i in range(len(seg_closes)):
            if seg_closes[i] < lower:
                opp_breaks += 1

    # Check for trending structure after last touch (lower-highs/lows or higher)
    trend_shift = "none"
    if len(seg_highs) >= 6:
        # Split into 2 halves and compare
        mid = len(seg_highs) // 2
        first_half_high = float(np.max(seg_highs[:mid]))
        second_half_high = float(np.max(seg_highs[mid:]))
        first_half_low = float(np.min(seg_lows[:mid]))
        second_half_low = float(np.min(seg_lows[mid:]))

        if second_half_high < first_half_high and second_half_low < first_half_low:
            trend_shift = "bearish"  # lower highs + lower lows
        elif second_half_high > first_half_high and second_half_low > first_half_low:
            trend_shift = "bullish"  # higher highs + higher lows

    return {
        "opposite_intact": opp_breaks == 0,
        "opposite_breaks": opp_breaks,
        "post_touch_trend": trend_shift,
    }


# ---------------------------------------------------------------------------
# Pre-chop trend detection
# ---------------------------------------------------------------------------

def detect_pre_chop_trend(closes: np.ndarray, range_start_idx: int,
                          lookback: int = 60) -> dict:
    """
    Determine what price was doing BEFORE the chop started.
    Look back `lookback` bars before range_start_idx.

    Returns {trend: 'up'/'down'/'unclear', move_pct, preferred_side}.
    """
    start = max(0, range_start_idx - lookback)
    if range_start_idx - start < 10:
        return {"trend": "unclear", "move_pct": 0, "preferred_side": None}

    seg = closes[start:range_start_idx]
    net_move = float(seg[-1]) - float(seg[0])
    move_pct = net_move / float(seg[0]) * 100

    # Use linear regression for cleaner signal
    x = np.arange(len(seg), dtype=float)
    slope = np.polyfit(x, seg.astype(float), 1)[0]
    norm_slope = slope / float(np.mean(seg)) * len(seg) * 100  # normalized % move

    if norm_slope > 0.3 and move_pct > 0.2:
        trend = "up"
        preferred_side = "long"
    elif norm_slope < -0.3 and move_pct < -0.2:
        trend = "down"
        preferred_side = "short"
    else:
        trend = "unclear"
        preferred_side = None

    return {
        "trend": trend,
        "move_pct": round(move_pct, 3),
        "preferred_side": preferred_side,
    }


# ---------------------------------------------------------------------------
# VWAP Bands
# ---------------------------------------------------------------------------

def compute_vwap_bands(df: pd.DataFrame, bar_idx: int,
                       band_mult: float = 1.0) -> dict:
    """
    Compute session VWAP and upper/lower bands at bar_idx.
    Session resets at 00:00 UTC each day.

    VWAP = cumsum(price * volume) / cumsum(volume)
    Bands = VWAP +/- band_mult * rolling std of (close - vwap)
    """
    ts = df.iloc[bar_idx]["timestamp"]
    day_start = ts.normalize()

    # Get all bars for current session
    session_mask = df["timestamp"] >= day_start
    session_end = df.index[session_mask].max()
    session = df.loc[session_mask]
    session = session[session.index <= bar_idx]

    if len(session) < 5:
        return {"vwap": None, "vwap_upper": None, "vwap_lower": None}

    typical_price = (session["high"] + session["low"] + session["close"]) / 3
    cum_tpv = (typical_price * session["volume"]).cumsum()
    cum_vol = session["volume"].cumsum()

    # Avoid division by zero
    cum_vol_safe = cum_vol.replace(0, np.nan)
    vwap_series = cum_tpv / cum_vol_safe

    vwap = float(vwap_series.iloc[-1]) if not np.isnan(vwap_series.iloc[-1]) else None
    if vwap is None:
        return {"vwap": None, "vwap_upper": None, "vwap_lower": None}

    # Band width: std of (close - vwap)
    deviations = session["close"].values - vwap_series.values
    std_dev = float(np.nanstd(deviations))

    return {
        "vwap": round(vwap, 8),
        "vwap_upper": round(vwap + band_mult * std_dev, 8),
        "vwap_lower": round(vwap - band_mult * std_dev, 8),
    }


def check_vwap_clear_path(entry_price: float, tp: float, side: str,
                           vwap_info: dict) -> dict:
    """
    Check that no VWAP band (upper, middle, lower) sits between entry and TP.
    Returns {clear: bool, blocking_band: str or None}.
    """
    if vwap_info["vwap"] is None:
        return {"vwap_clear": True, "blocking_band": None}

    bands = [
        ("vwap_lower", vwap_info["vwap_lower"]),
        ("vwap", vwap_info["vwap"]),
        ("vwap_upper", vwap_info["vwap_upper"]),
    ]

    for band_name, band_price in bands:
        if band_price is None:
            continue
        if side == "long":
            # TP is above entry for longs
            if entry_price < band_price < tp:
                return {"vwap_clear": False, "blocking_band": band_name}
        else:
            # TP is below entry for shorts
            if tp < band_price < entry_price:
                return {"vwap_clear": False, "blocking_band": band_name}

    return {"vwap_clear": True, "blocking_band": None}


# ---------------------------------------------------------------------------
# Entry detection at range boundary
# ---------------------------------------------------------------------------

def detect_range_entry(df: pd.DataFrame, bar_idx: int, range_info: dict,
                       cfg: MRSettings) -> Optional[dict]:
    """
    Check if current bar is a potential MR entry at a range boundary.

    Long entry: price spikes below/touches lower bound and closes back inside
    Short entry: price spikes above/touches upper bound and closes back inside
    """
    high = float(df.iloc[bar_idx]["high"])
    low = float(df.iloc[bar_idx]["low"])
    close = float(df.iloc[bar_idx]["close"])
    upper = range_info["upper"]
    lower = range_info["lower"]
    mid = range_info["mid"]
    # "at level" tolerance: close within 0.05% of the bound
    at_level_tol = 0.0005

    entries = []

    # Short entry: wick touches/crosses upper bound, close stays below or at level
    if high >= upper * (1 - at_level_tol):
        overshoot = (high - upper) / upper * 100 if high > upper else 0
        if overshoot <= 0.5 and close < upper:
            # Check it's not just sitting at the top — need some wick rejection
            wick_above = high - max(df.iloc[bar_idx]["open"], close)
            body = abs(close - df.iloc[bar_idx]["open"])
            if body < 1e-12:
                wick_ratio = 1.0
            else:
                wick_ratio = wick_above / body if body > 0 else 1.0

            entries.append({
                "side": "short",
                "entry_price": close,
                "level_price": upper,
                "level_side": "upper",
                "overshoot_pct": round(overshoot, 4),
            })

    # Long entry: low touches/dips below lower bound, close stays above
    # Long entry: wick touches/crosses lower bound, close stays above or at level
    if low <= lower * (1 + at_level_tol):
        overshoot = (lower - low) / lower * 100 if low < lower else 0
        if overshoot <= 0.5 and close > lower:
            entries.append({
                "side": "long",
                "entry_price": close,
                "level_price": lower,
                "level_side": "lower",
                "overshoot_pct": round(overshoot, 4),
            })

    # If both sides triggered (unlikely), pick the one closer to the bound
    if len(entries) == 2:
        dist_upper = abs(close - upper)
        dist_lower = abs(close - lower)
        return entries[0] if dist_upper < dist_lower else entries[1]
    elif len(entries) == 1:
        return entries[0]

    return None


# ---------------------------------------------------------------------------
# DPS Scoring (Mean Reversion)
# ---------------------------------------------------------------------------

def evaluate_dps(df: pd.DataFrame, bar_idx: int, range_info: dict,
                 entry_info: dict, cfg: MRSettings) -> dict:
    """
    DPS (Discretionary Point System) scoring for MR trades.

    3 variables, each scored 0-2, max score 6:
      1. Choppy Range Duration: <2hrs=0, >2hrs=1, >4hrs=2
      2. Approach to Level: Grind=0, Unclear=1, Spike=2
      3. Volume (differs by side):
         Long:  Increasing=0, Decreasing=1, Flat=2
         Short: Increasing=0, Flat=1, Decreasing=2

    Confidence mapping:
      6     = Max confidence (1% risk)
      4-5   = High confidence (1% risk)
      3     = Low confidence (0.1% dummy)
      0-2   = Avoid (stay away)
    """
    side = entry_info["side"]
    closes = df["close"].values
    volumes = df["volume"].values
    highs = df["high"].values
    lows = df["low"].values

    # --- Variable 1: Choppy Range Duration ---
    duration_hrs = range_info["duration_hrs"]
    if duration_hrs >= 4:
        v1_score = 2
        v1_label = ">4hrs"
    elif duration_hrs >= 2:
        v1_score = 1
        v1_label = ">2hrs"
    else:
        v1_score = 0
        v1_label = "<2hrs"

    # --- Variable 2: Approach to Level ---
    # How did price approach the entry-side level on THIS touch?
    # Spike into level = best for MR (2), Grind = worst (0)
    # Look at last 5-10 bars before entry to measure speed of approach
    for approach_lookback in [5, 8, 10]:
        approach_start = max(0, bar_idx - approach_lookback)
        approach_closes = closes[approach_start:bar_idx + 1]
        approach_highs = highs[approach_start:bar_idx + 1]
        approach_lows = lows[approach_start:bar_idx + 1]

        if len(approach_closes) < 3:
            continue

        # Net move toward the level as % of price
        approach_move_pct = abs(approach_closes[-1] - approach_closes[0]) / approach_closes[0] * 100

        # Efficiency: how direct was the approach (1.0 = straight line)
        approach_range = float(np.max(approach_highs) - np.min(approach_lows))
        approach_net = abs(float(approach_closes[-1]) - float(approach_closes[0]))
        approach_eff = approach_net / approach_range if approach_range > 0 else 0

        # Per-bar speed
        approach_speed = approach_move_pct / len(approach_closes)

        # Fast spike: high speed + high efficiency
        if approach_speed >= 0.06 and approach_eff >= 0.45:
            v2_score = 2
            v2_label = "spike"
            break
        elif approach_speed >= 0.03 and approach_eff >= 0.25:
            v2_score = 1
            v2_label = "unclear"
            break
        else:
            v2_score = 0
            v2_label = "grind"
    else:
        v2_score = 1
        v2_label = "unclear"

    # --- Variable 3: Volume ---
    vol_start = max(0, bar_idx - cfg.vol_lookback_bars)
    vol_window = volumes[vol_start:bar_idx]
    vol_trend = detect_volume_trend(vol_window)

    if side == "long":
        # Long: Increasing=0, Decreasing=1, Flat=2
        if vol_trend == "increasing":
            v3_score = 0
        elif vol_trend == "decreasing":
            v3_score = 1
        else:
            v3_score = 2
    else:
        # Short: Increasing=0, Flat=1, Decreasing=2
        if vol_trend == "increasing":
            v3_score = 0
        elif vol_trend == "flat":
            v3_score = 1
        else:
            v3_score = 2

    total_score = v1_score + v2_score + v3_score

    if total_score >= 6:
        confidence = "max"
        risk_pct = 1.0
    elif total_score >= 4:
        confidence = "high"
        risk_pct = 1.0
    elif total_score >= 3:
        confidence = "low"
        risk_pct = 0.1
    else:
        confidence = "avoid"
        risk_pct = 0.0

    return {
        "dps_total": total_score,
        "dps_confidence": confidence,
        "dps_risk_pct": risk_pct,
        "dps_v1_duration": v1_score,
        "dps_v1_label": v1_label,
        "dps_v2_approach": v2_score,
        "dps_v2_label": v2_label,
        "dps_v3_volume": v3_score,
        "dps_v3_vol_trend": vol_trend,
    }


# ---------------------------------------------------------------------------
# Noise classification (30 SMMA)
# ---------------------------------------------------------------------------

def classify_noise(closes: np.ndarray, bar_idx: int, cfg: MRSettings) -> dict:
    """Classify noise using 30 SMMA crosses + MA direction."""
    noise_start = max(0, bar_idx - cfg.noise_lookback_bars)
    seg = closes[noise_start:bar_idx + 1]
    smma30 = _smma(pd.Series(seg), 30).values
    n_crosses = count_smma_crosses(seg, smma30)

    # MA direction
    if len(smma30) > 30:
        ma_start = smma30[30]
        ma_end = smma30[-1]
        ma_change_pct = abs(ma_end - ma_start) / ma_start * 100 if ma_start > 0 else 0
        ma_direction = "sideways" if ma_change_pct < 2.0 else "trending"
    else:
        ma_direction = "unclear"
        ma_change_pct = 0

    if n_crosses >= 7 and ma_direction == "sideways":
        noise_level = "low"
    elif n_crosses >= 7 or (n_crosses >= 4 and ma_direction == "sideways"):
        noise_level = "medium"
    else:
        noise_level = "high"

    return {
        "noise_level": noise_level,
        "smma_crosses": n_crosses,
        "ma_direction": ma_direction,
        "ma_change_pct": round(ma_change_pct, 2),
    }


# ---------------------------------------------------------------------------
# SL/TP computation
# ---------------------------------------------------------------------------

def compute_sl_tp(df: pd.DataFrame, entry_idx: int, entry_price: float,
                  side: str, range_info: dict, vwap_info: dict,
                  cfg: MRSettings) -> Optional[dict]:
    """
    TP: target the opposite side of the choppy range (based on previous swings).
    SL: just outside the entry-side range bound, or at nearest VWAP band
        (whichever is tighter but still >= min_sl_pct).
    """
    highs = df["high"].values
    lows = df["low"].values
    upper = range_info["upper"]
    lower = range_info["lower"]

    if side == "short":
        # TP: target near the lower bound (opposite side of range)
        # Use 99.5% of range to give some buffer
        tp = lower * 1.002  # small buffer above range bottom
        tp_pct = (entry_price - tp) / entry_price * 100

        # SL: just above the upper bound
        swing_h = find_swing_high(highs, entry_idx, lookback=30)
        sl = upper * 1.003  # just above range top
        if swing_h is not None and swing_h > sl:
            sl = swing_h * 1.001

        # Consider VWAP upper band as SL if it's above entry and tighter
        if vwap_info.get("vwap_upper") and vwap_info["vwap_upper"] > entry_price:
            vwap_sl = vwap_info["vwap_upper"] * 1.001
            if vwap_sl < sl:
                sl = vwap_sl

        sl_pct = (sl - entry_price) / entry_price * 100

    else:  # long
        # TP: target near the upper bound (opposite side of range)
        tp = upper * 0.998  # small buffer below range top
        tp_pct = (tp - entry_price) / entry_price * 100

        # SL: just below the lower bound
        swing_l = find_swing_low(lows, entry_idx, lookback=30)
        sl = lower * 0.997  # just below range bottom
        if swing_l is not None and swing_l < sl:
            sl = swing_l * 0.999

        # Consider VWAP lower band as SL if it's below entry and tighter
        if vwap_info.get("vwap_lower") and vwap_info["vwap_lower"] < entry_price:
            vwap_sl = vwap_info["vwap_lower"] * 0.999
            if vwap_sl > sl:
                sl = vwap_sl

        sl_pct = (entry_price - sl) / entry_price * 100

    # Enforce SL bounds
    if sl_pct < cfg.min_sl_pct:
        if side == "short":
            sl = entry_price * (1 + cfg.min_sl_pct / 100)
        else:
            sl = entry_price * (1 - cfg.min_sl_pct / 100)
        sl_pct = cfg.min_sl_pct
    if sl_pct > cfg.max_sl_pct:
        if side == "short":
            sl = entry_price * (1 + cfg.max_sl_pct / 100)
        else:
            sl = entry_price * (1 - cfg.max_sl_pct / 100)
        sl_pct = cfg.max_sl_pct

    # TP must be at least 0.5% to be worth it
    if tp_pct < 0.5:
        return None

    rr = tp_pct / sl_pct if sl_pct > 0 else 0

    # Require at least 1:1 RR
    if rr < 1.0:
        return None

    return {
        "sl": round(sl, 8),
        "tp": round(tp, 8),
        "sl_pct": round(sl_pct, 3),
        "tp_pct": round(tp_pct, 3),
        "rr": round(rr, 2),
    }


# ---------------------------------------------------------------------------
# Shared gate function — used by both backtest and live trader
# ---------------------------------------------------------------------------

def check_mr_gates_at_bar(df: pd.DataFrame, bar_idx: int,
                          cfg: MRSettings) -> dict:
    """
    Run all MR gates on a single bar. Returns a result dict with:
      - passed: bool
      - reason: str (why it failed, or 'all_gates_passed')
      - All setup fields (entry, sl, tp, dps, touches, etc.) if passed

    This is the single source of truth for MR detection logic,
    used by both scan_symbol() (backtest) and detect_mr_setup_live().
    """
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    i = bar_idx

    # Step 1: Detect choppy range
    range_info = detect_choppy_range(highs, lows, closes, i, cfg)
    if range_info is None:
        return {"passed": False, "reason": "no_range"}

    # Step 2: Pre-chop trend
    pre_trend = detect_pre_chop_trend(closes, range_info["start_idx"])

    # Step 3: Entry at range boundary
    entry_info = detect_range_entry(df, i, range_info, cfg)
    if entry_info is None:
        return {"passed": False, "reason": "no_entry_trigger"}

    # Filter: side must match pre-chop trend direction
    if pre_trend["preferred_side"] is not None:
        if entry_info["side"] != pre_trend["preferred_side"]:
            return {"passed": False, "reason": "side_vs_pre_trend"}

    # Filter: SMMA 30/120 trend — don't trade against the higher-timeframe trend
    smma30 = pd.Series(closes[:i+1]).ewm(alpha=1.0 / 30, adjust=False).mean().values
    smma120 = pd.Series(closes[:i+1]).ewm(alpha=1.0 / 120, adjust=False).mean().values
    window = min(120, i)
    smma30_slope = (smma30[-1] - smma30[-window]) / smma30[-window] * 100 if window > 0 else 0
    smma120_slope = (smma120[-1] - smma120[-window]) / smma120[-window] * 100 if window > 0 else 0
    if smma30_slope > 0 and smma120_slope > 0 and entry_info["side"] == "short":
        return {"passed": False, "reason": "smma_trend_up_vs_short"}
    if smma30_slope < 0 and smma120_slope < 0 and entry_info["side"] == "long":
        return {"passed": False, "reason": "smma_trend_down_vs_long"}

    # Step 4: Touch counting + level integrity
    entry_side = entry_info["level_side"]
    touches = count_level_touches(
        highs, lows, closes, opens,
        range_info["start_idx"], i,
        entry_info["level_price"], entry_side,
        debounce_bars=cfg.touch_cluster_bars,
    )

    if touches["count"] < cfg.min_touches:
        return {"passed": False, "reason": f"touches_{touches['count']}_lt_{cfg.min_touches}"}
    if touches["break_pct"] > 5.0:
        return {"passed": False, "reason": "break_pct_too_high"}

    # Step 5: Last touch recency
    if touches["last_touch_idx"] < 0:
        return {"passed": False, "reason": "no_last_touch"}
    bars_since_last = i - touches["last_touch_idx"]
    if bars_since_last > cfg.last_touch_max_bars:
        return {"passed": False, "reason": "last_touch_too_old"}

    # Step 5b: Opposite boundary check
    opp_check = check_opposite_bound_intact(
        highs, lows, closes,
        touches["last_touch_idx"], i,
        range_info, entry_side)
    if not opp_check["opposite_intact"]:
        return {"passed": False, "reason": "opposite_bound_broken"}

    # Step 5c: Post-touch trend contradiction
    if opp_check["post_touch_trend"] == "bearish" and entry_info["side"] == "long":
        return {"passed": False, "reason": "post_touch_bearish_vs_long"}
    if opp_check["post_touch_trend"] == "bullish" and entry_info["side"] == "short":
        return {"passed": False, "reason": "post_touch_bullish_vs_short"}

    # Step 6: VWAP bands
    vwap_info = compute_vwap_bands(df, i)

    # Step 7: SL/TP
    sl_tp = compute_sl_tp(df, i, entry_info["entry_price"],
                          entry_info["side"], range_info, vwap_info, cfg)
    if sl_tp is None:
        return {"passed": False, "reason": "sl_tp_invalid"}

    # Step 8: VWAP clear path
    vwap_check = check_vwap_clear_path(
        entry_info["entry_price"], sl_tp["tp"],
        entry_info["side"], vwap_info)
    if not vwap_check["vwap_clear"]:
        return {"passed": False, "reason": f"vwap_blocked_{vwap_check['blocking_band']}"}

    # Step 9: Noise classification
    noise = classify_noise(closes, i, cfg)

    # Step 10: DPS scoring
    dps = evaluate_dps(df, i, range_info, entry_info, cfg)

    # Build touch timestamps
    touch_timestamps = "|".join(
        str(df.iloc[idx]["timestamp"]) for idx in touches["touch_indices"]
    )

    return {
        "passed": True,
        "reason": "all_gates_passed",
        "side": entry_info["side"],
        "entry": round(entry_info["entry_price"], 8),
        "level_price": round(entry_info["level_price"], 8),
        "level_side": entry_side,
        "overshoot_pct": entry_info["overshoot_pct"],
        "range_upper": round(range_info["upper"], 8),
        "range_lower": round(range_info["lower"], 8),
        "range_width_pct": range_info["width_pct"],
        "range_duration_hrs": range_info["duration_hrs"],
        "range_containment_pct": range_info["containment_pct"],
        "range_efficiency": range_info["efficiency"],
        "touches": touches["count"],
        "break_count": touches["break_count"],
        "break_pct": touches["break_pct"],
        "bars_since_last_touch": bars_since_last,
        "post_touch_trend": opp_check["post_touch_trend"],
        "pre_chop_trend": pre_trend["trend"],
        "pre_chop_move_pct": pre_trend["move_pct"],
        "vwap": vwap_info.get("vwap"),
        "vwap_upper": vwap_info.get("vwap_upper"),
        "vwap_lower": vwap_info.get("vwap_lower"),
        **noise,
        **dps,
        **sl_tp,
        "touch_timestamps": touch_timestamps,
    }


# ---------------------------------------------------------------------------
# Strict MR — stricter touch counting
# ---------------------------------------------------------------------------

def count_level_touches_strict(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    opens: np.ndarray, start_idx: int, end_idx: int,
    level: float, side: str, opposite_level: float,
    at_level_pct: float = 0.05,
    debounce_bars: int = 5,
) -> dict:
    """
    Strict touch counting with two extra rules:

    Rule 1 — Swing-back: Only count a touch if price swung back to the
             opposite bound (TP side) since the previous touch.  The first
             touch is always counted.

    Rule 2 — Clean entries: Among the last 3 counted touches, no more than
             1 candle may have closed beyond the entry-side level.

    Returns the same shape as count_level_touches plus:
      clean_entries (bool), closes_beyond_count (int).
    """
    touches: list[int] = []
    breaks = 0
    last_touch_bar = -999
    at_level_tol = level * at_level_pct / 100.0
    opp_tol = opposite_level * at_level_pct / 100.0

    # After first touch, track whether price visited the opposite bound
    visited_opposite = True  # first touch doesn't need swing-back

    for i in range(start_idx, end_idx + 1):
        # --- track breaks (close beyond entry level) ---
        if side == "upper":
            if closes[i] > level + at_level_tol:
                breaks += 1
        else:
            if closes[i] < level - at_level_tol:
                breaks += 1

        # --- track swing-back to opposite bound ---
        if len(touches) > 0 and not visited_opposite:
            if side == "upper":
                # short entry → price must swing down to lower bound
                if lows[i] <= opposite_level + opp_tol:
                    visited_opposite = True
            else:
                # long entry → price must swing up to upper bound
                if highs[i] >= opposite_level - opp_tol:
                    visited_opposite = True

        # --- debounce ---
        if i - last_touch_bar < debounce_bars:
            continue

        # --- check touch ---
        if side == "upper":
            wick_touch = highs[i] >= level
            body_below = closes[i] < level
            close_at_level = abs(closes[i] - level) <= at_level_tol

            if (wick_touch and body_below) or close_at_level:
                if visited_opposite:
                    touches.append(i)
                    last_touch_bar = i
                    visited_opposite = False
        else:  # lower / support
            wick_touch = lows[i] <= level
            body_above = closes[i] > level
            close_at_level = abs(closes[i] - level) <= at_level_tol

            if (wick_touch and body_above) or close_at_level:
                if visited_opposite:
                    touches.append(i)
                    last_touch_bar = i
                    visited_opposite = False

    # --- Rule 2: last-3 close-beyond check ---
    last_3 = touches[-3:] if len(touches) >= 3 else touches
    closes_beyond = 0
    for idx in last_3:
        if side == "upper" and closes[idx] >= level:
            closes_beyond += 1
        elif side == "lower" and closes[idx] <= level:
            closes_beyond += 1
    clean_entries = closes_beyond <= 1

    n_bars = end_idx - start_idx + 1
    break_pct = breaks / n_bars * 100 if n_bars > 0 else 0

    return {
        "count": len(touches),
        "last_touch_idx": touches[-1] if touches else -1,
        "touch_indices": touches,
        "break_count": breaks,
        "break_pct": round(break_pct, 1),
        "clean_entries": clean_entries,
        "closes_beyond_count": closes_beyond,
    }


def check_strict_mr_gates_at_bar(
    df: pd.DataFrame, bar_idx: int, cfg: MRSettings
) -> dict:
    """
    Strict MR gate — identical to check_mr_gates_at_bar but with:
      1. Swing-back touch counting (price must visit opposite bound between touches)
      2. Clean-entry filter (≤1 of last 3 touches closed beyond entry level)
    """
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    i = bar_idx

    # Step 1: Detect choppy range
    range_info = detect_choppy_range(highs, lows, closes, i, cfg)
    if range_info is None:
        return {"passed": False, "reason": "no_range"}

    # Step 2: Pre-chop trend
    pre_trend = detect_pre_chop_trend(closes, range_info["start_idx"])

    # Step 3: Entry at range boundary
    entry_info = detect_range_entry(df, i, range_info, cfg)
    if entry_info is None:
        return {"passed": False, "reason": "no_entry_trigger"}

    # Filter: side must match pre-chop trend direction
    if pre_trend["preferred_side"] is not None:
        if entry_info["side"] != pre_trend["preferred_side"]:
            return {"passed": False, "reason": "side_vs_pre_trend"}

    # Filter: SMMA 30/120 trend
    smma30 = pd.Series(closes[:i+1]).ewm(alpha=1.0 / 30, adjust=False).mean().values
    smma120 = pd.Series(closes[:i+1]).ewm(alpha=1.0 / 120, adjust=False).mean().values
    window = min(120, i)
    smma30_slope = (smma30[-1] - smma30[-window]) / smma30[-window] * 100 if window > 0 else 0
    smma120_slope = (smma120[-1] - smma120[-window]) / smma120[-window] * 100 if window > 0 else 0
    if smma30_slope > 0 and smma120_slope > 0 and entry_info["side"] == "short":
        return {"passed": False, "reason": "smma_trend_up_vs_short"}
    if smma30_slope < 0 and smma120_slope < 0 and entry_info["side"] == "long":
        return {"passed": False, "reason": "smma_trend_down_vs_long"}

    # Step 4: STRICT touch counting + level integrity
    entry_side = entry_info["level_side"]
    opposite_level = (range_info["lower"] if entry_side == "upper"
                      else range_info["upper"])

    touches = count_level_touches_strict(
        highs, lows, closes, opens,
        range_info["start_idx"], i,
        entry_info["level_price"], entry_side,
        opposite_level=opposite_level,
        debounce_bars=cfg.touch_cluster_bars,
    )

    if touches["count"] < cfg.min_touches:
        return {"passed": False, "reason": f"touches_{touches['count']}_lt_{cfg.min_touches}"}
    if touches["break_pct"] > 5.0:
        return {"passed": False, "reason": "break_pct_too_high"}
    if not touches["clean_entries"]:
        return {"passed": False, "reason": f"closes_beyond_{touches['closes_beyond_count']}_in_last3"}

    # Step 5: Last touch recency
    if touches["last_touch_idx"] < 0:
        return {"passed": False, "reason": "no_last_touch"}
    bars_since_last = i - touches["last_touch_idx"]
    if bars_since_last > cfg.last_touch_max_bars:
        return {"passed": False, "reason": "last_touch_too_old"}

    # Step 5b: Opposite boundary check
    opp_check = check_opposite_bound_intact(
        highs, lows, closes,
        touches["last_touch_idx"], i,
        range_info, entry_side)
    if not opp_check["opposite_intact"]:
        return {"passed": False, "reason": "opposite_bound_broken"}

    # Step 5c: Post-touch trend contradiction
    if opp_check["post_touch_trend"] == "bearish" and entry_info["side"] == "long":
        return {"passed": False, "reason": "post_touch_bearish_vs_long"}
    if opp_check["post_touch_trend"] == "bullish" and entry_info["side"] == "short":
        return {"passed": False, "reason": "post_touch_bullish_vs_short"}

    # Step 6: VWAP bands
    vwap_info = compute_vwap_bands(df, i)

    # Step 7: SL/TP
    sl_tp = compute_sl_tp(df, i, entry_info["entry_price"],
                          entry_info["side"], range_info, vwap_info, cfg)
    if sl_tp is None:
        return {"passed": False, "reason": "sl_tp_invalid"}

    # Step 8: VWAP clear path
    vwap_check = check_vwap_clear_path(
        entry_info["entry_price"], sl_tp["tp"],
        entry_info["side"], vwap_info)
    if not vwap_check["vwap_clear"]:
        return {"passed": False, "reason": f"vwap_blocked_{vwap_check['blocking_band']}"}

    # Step 9: Noise classification
    noise = classify_noise(closes, i, cfg)

    # Step 10: DPS scoring
    dps = evaluate_dps(df, i, range_info, entry_info, cfg)

    # Build touch timestamps
    touch_timestamps = "|".join(
        str(df.iloc[idx]["timestamp"]) for idx in touches["touch_indices"]
    )

    return {
        "passed": True,
        "reason": "all_gates_passed",
        "side": entry_info["side"],
        "entry": round(entry_info["entry_price"], 8),
        "level_price": round(entry_info["level_price"], 8),
        "level_side": entry_side,
        "overshoot_pct": entry_info["overshoot_pct"],
        "range_upper": round(range_info["upper"], 8),
        "range_lower": round(range_info["lower"], 8),
        "range_width_pct": range_info["width_pct"],
        "range_duration_hrs": range_info["duration_hrs"],
        "range_containment_pct": range_info["containment_pct"],
        "range_efficiency": range_info["efficiency"],
        "touches": touches["count"],
        "break_count": touches["break_count"],
        "break_pct": touches["break_pct"],
        "closes_beyond_count": touches["closes_beyond_count"],
        "bars_since_last_touch": bars_since_last,
        "post_touch_trend": opp_check["post_touch_trend"],
        "pre_chop_trend": pre_trend["trend"],
        "pre_chop_move_pct": pre_trend["move_pct"],
        "vwap": vwap_info.get("vwap"),
        "vwap_upper": vwap_info.get("vwap_upper"),
        "vwap_lower": vwap_info.get("vwap_lower"),
        **noise,
        **dps,
        **sl_tp,
        "touch_timestamps": touch_timestamps,
    }


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def scan_symbol(df: pd.DataFrame, symbol: str, cfg: MRSettings) -> list[dict]:
    """Scan a single symbol for Choppy Range MR setups."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    setups = []
    cooldown_until = -1

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values

    # Need warmup for range detection
    start_bar = max(cfg.range_max_bars, cfg.noise_lookback_bars, 720)

    for i in range(start_bar, len(df) - 1):
        if i <= cooldown_until:
            continue

        result = check_mr_gates_at_bar(df, i, cfg)
        if not result["passed"]:
            continue

        entry_ts = df.iloc[i]["timestamp"]

        setup = {
            "symbol": symbol,
            "timestamp": str(entry_ts),
        }
        # Copy all fields from result except internal ones
        for k, v in result.items():
            if k not in ("passed", "reason"):
                setup[k] = v

        setups.append(setup)
        cooldown_until = i + cfg.cooldown_bars

    return setups


def main():
    parser = argparse.ArgumentParser(description="Choppy Range MR Scanner")
    parser.add_argument("--dataset-dir", required=True,
                        help="Directory with 1m CSV files")
    parser.add_argument("--out-prefix", default="mr_choppy",
                        help="Output file prefix")
    parser.add_argument("--settings", default=None,
                        help="JSON settings file (optional)")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbols to scan (default: all)")
    parser.add_argument("--exclude-symbols", default=None,
                        help="Comma-separated symbols to exclude")
    args = parser.parse_args()

    cfg = MRSettings.from_json(args.settings) if args.settings else MRSettings()

    dataset_dir = Path(args.dataset_dir)
    csv_files = sorted(dataset_dir.glob("*_1m*.csv"))

    if args.symbols:
        wanted = {s.strip() for s in args.symbols.split(",")}
        csv_files = [f for f in csv_files
                     if f.stem.split("_1m")[0] in wanted]

    exclude = set()
    if args.exclude_symbols:
        exclude = {s.strip() for s in args.exclude_symbols.split(",")}

    print(f"Scanning {len(csv_files)} files for Choppy Range MR setups...")
    print(f"Settings: min_touches={cfg.min_touches}, range_min={cfg.range_min_bars}bars, "
          f"last_touch_max={cfg.last_touch_max_bars}bars")

    all_setups = []
    for fi, csv_path in enumerate(csv_files):
        symbol = csv_path.stem.split("_1m")[0]
        if symbol in exclude:
            continue

        df = pd.read_csv(str(csv_path), parse_dates=["timestamp"])
        if len(df) < 1500:
            continue

        setups = scan_symbol(df, symbol, cfg)
        all_setups.extend(setups)

        if (fi + 1) % 20 == 0 or fi == len(csv_files) - 1:
            print(f"  [{fi+1}/{len(csv_files)}] {symbol:20s}  "
                  f"total setups so far: {len(all_setups)}")

    if not all_setups:
        print("\nNo setups found. Try adjusting range/touch thresholds.")
        return

    result_df = pd.DataFrame(all_setups)
    out_path = f"{args.out_prefix}_setups.csv"
    result_df.to_csv(out_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"CHOPPY RANGE MR RESULTS")
    print(f"{'='*60}")
    print(f"Total setups: {len(result_df)}")
    print(f"Symbols with setups: {result_df['symbol'].nunique()}")

    print(f"\nBy side:")
    print(result_df["side"].value_counts().to_string())

    print(f"\nBy DPS confidence:")
    print(result_df["dps_confidence"].value_counts().to_string())

    print(f"\nDPS score distribution:")
    print(result_df["dps_total"].value_counts().sort_index().to_string())

    print(f"\nBy noise level:")
    print(result_df["noise_level"].value_counts().to_string())

    print(f"\nRange duration (hours):")
    print(f"  Mean: {result_df['range_duration_hrs'].mean():.1f}")
    print(f"  Min:  {result_df['range_duration_hrs'].min():.1f}")
    print(f"  Max:  {result_df['range_duration_hrs'].max():.1f}")

    print(f"\nTouches on entry side:")
    print(f"  Mean: {result_df['touches'].mean():.1f}")
    print(f"  Min:  {result_df['touches'].min()}")
    print(f"  Max:  {result_df['touches'].max()}")

    print(f"\nVolume trend:")
    print(result_df["dps_v3_vol_trend"].value_counts().to_string())

    print(f"\nApproach quality:")
    print(result_df["dps_v2_label"].value_counts().to_string())

    # Show top setups by DPS
    if len(result_df) > 0:
        top = result_df.nlargest(10, "dps_total")
        print(f"\nTop 10 setups by DPS score:")
        print(f"{'Symbol':20s} {'Side':5s} {'DPS':3s} {'Conf':4s} "
              f"{'Dur':4s} {'Touch':5s} {'Noise':6s} {'RngW%':6s}")
        print("-" * 60)
        for _, s in top.iterrows():
            print(f"{s['symbol']:20s} {s['side']:5s} {s['dps_total']:3d} "
                  f"{s['dps_confidence']:4s} {s['range_duration_hrs']:4.1f}h "
                  f"{s['touches']:5d} {s['noise_level']:6s} {s['range_width_pct']:6.2f}")

    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
