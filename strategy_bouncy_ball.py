#!/usr/bin/env python3
"""
Bouncy Ball Mean Reversion Strategy

Detects coins oscillating between two proven levels (support/resistance).
Entry when price touches a boundary with confirmation.

Setup criteria:
  1. Choppy range with upper/lower boundaries touched 3+ times each
  2. Levels must be cleanly respected (low overshoot)
  3. Pre-chop trend direction filter (only trade WITH prior trend)
  4. Volume flat/decreasing before entry
  5. Entry at the level (limit order), TP at opposite boundary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BouncyBallSettings:
    """Settings for bouncy ball MR strategy."""
    # Range detection
    min_range_bars: int = 180       # Min bars for range (3 hours)
    max_range_bars: int = 480       # Max lookback (8 hours)
    min_range_pct: float = 1.0     # Min range width %
    max_range_pct: float = 5.0     # Max range width %
    percentile_upper: float = 92    # Percentile for upper boundary
    percentile_lower: float = 8     # Percentile for lower boundary

    # Level quality
    min_touches_per_side: int = 3   # Min touches on each boundary
    max_overshoot_pct: float = 0.25 # Max avg overshoot at levels
    min_inside_pct: float = 70      # Min % of bars inside range
    min_clean_score: int = 5        # Min range cleanliness score (0-10)

    # Entry/exit
    entry_buffer_pct: float = 0.0   # Entry at level (0 = exact level)
    sl_buffer_pct: float = 0.5      # SL this % beyond boundary
    min_sl_pct: float = 0.5         # Minimum SL distance %
    tp_buffer_pct: float = 0.3      # TP this % before opposite boundary
    min_rr: float = 1.0             # Minimum RR
    max_rr: float = 1.5             # Cap RR (if swing TP gives higher, cap to 1.25R)
    cap_rr: float = 1.25            # RR to cap to when original > max_rr
    min_tp_pct: float = 0.5         # Minimum TP distance %
    use_swing_tp: bool = True       # Use swing-depth TP instead of opposite boundary
    min_swing_travel_pct: float = 40  # Min % of range width for a real swing

    # Pre-trend
    pre_trend_lookback: int = 240   # Bars to look back for pre-range trend
    pre_trend_threshold: float = 0.5  # % move to classify as trending


def _count_touches(highs, lows, closes, level, min_gap=10):
    """Count touches at a level + measure overshoot quality."""
    n = len(highs)
    touches = 0
    overshoots = []
    last = -min_gap
    for j in range(n):
        if j - last < min_gap:
            continue
        near_high = abs(highs[j] - level) / level * 100 < 0.5
        near_low = abs(lows[j] - level) / level * 100 < 0.5
        close_at = abs(closes[j] - level) / level * 100 < 0.3
        if near_high or near_low or close_at:
            touches += 1
            os_h = max(0, (highs[j] - level) / level * 100) if highs[j] > level else 0
            os_l = max(0, (level - lows[j]) / level * 100) if lows[j] < level else 0
            overshoots.append(max(os_h, os_l))
            last = j
    avg_os = np.mean(overshoots) if overshoots else 999
    return touches, avg_os


def _range_cleanliness(highs, lows, closes, upper, lower):
    """Score how clean/organized the range is (0-10)."""
    n = len(closes)
    if n < 50:
        return 0

    inside = np.mean((closes >= lower * 0.998) & (closes <= upper * 1.002)) * 100
    blasts = np.sum(highs > upper * 1.005) + np.sum(lows < lower * 0.995)
    blast_pct = blasts / n * 100

    ret = np.diff(closes) / closes[:-1]
    ac = np.corrcoef(ret[:-1], ret[1:])[0, 1] if len(ret) > 20 else 0

    score = 0
    if inside > 90: score += 3
    elif inside > 80: score += 2
    elif inside > 70: score += 1

    if blast_pct < 2: score += 3
    elif blast_pct < 5: score += 2
    elif blast_pct < 10: score += 1

    if ac < -0.1: score += 2
    elif ac < 0: score += 1

    mid = (upper + lower) / 2
    end_pos = abs(closes[-1] - mid) / (upper - lower) if upper != lower else 1
    if end_pos < 0.3: score += 2
    elif end_pos < 0.5: score += 1

    return score


def _find_real_swing_touches(highs, lows, closes, level, start, end, range_width, side, min_travel_pct=40):
    """
    Find real swing touches — price must travel at least min_travel_pct% of range
    away from level and come back.
    Returns list of {'bar': int, 'retrace_depth': float} newest first.
    """
    min_travel = range_width * min_travel_pct / 100
    swings = []
    near_level = False
    farthest = 0

    for j in range(start, end):
        at_level = (abs(highs[j] - level) / level * 100 < 0.4 or
                    abs(lows[j] - level) / level * 100 < 0.4 or
                    abs(closes[j] - level) / level * 100 < 0.3)

        if side == "short":
            dist = level - lows[j]
        else:
            dist = highs[j] - level

        if at_level:
            if not near_level and farthest >= min_travel:
                swings.append({'bar': j, 'retrace_depth': farthest})
            near_level = True
            farthest = 0
        else:
            near_level = False
            if dist > farthest:
                farthest = dist

    swings.reverse()
    return swings


def _detect_pre_trend(closes, highs, lows, range_start, range_end, lookback=240):
    """
    Detect how price entered the range.

    Simple approach: look at the highest high and lowest low in a wider window
    (range + lookback before). If price dropped significantly from the peak
    to the current range → downtrend. If it rallied from a trough → uptrend.

    This catches: uptrend → spike → selloff → chop (= downtrend context)
    """
    # Look at wider window: lookback bars before range + the range itself
    wide_start = max(0, range_start - lookback)
    wide_highs = highs[wide_start:range_end]
    wide_lows = lows[wide_start:range_end]

    if len(wide_highs) < 30:
        return "flat", 0.0

    # Current price = average of last 30 bars in range
    current_avg = np.mean(closes[max(range_start, range_end - 30):range_end])

    # Highest high and lowest low in the wider window
    peak = np.max(wide_highs)
    trough = np.min(wide_lows)

    # How far is current price from peak vs trough?
    drop_from_peak = (peak - current_avg) / peak * 100
    rise_from_trough = (current_avg - trough) / trough * 100

    # If we're much closer to the trough → uptrend (came from below)
    # If we're much closer to the peak → downtrend (came from above)
    if drop_from_peak > 1.0 and drop_from_peak > rise_from_trough:
        return "down", -drop_from_peak
    elif rise_from_trough > 1.0 and rise_from_trough > drop_from_peak:
        return "up", rise_from_trough

    # Check range drift as fallback
    range_closes = closes[range_start:range_end]
    if len(range_closes) >= 60:
        first_avg = np.mean(range_closes[:60])
        last_avg = np.mean(range_closes[-60:])
        drift = (last_avg - first_avg) / first_avg * 100
        if drift > 0.5:
            return "up", drift
        elif drift < -0.5:
            return "down", drift

    return "flat", 0.0


def check_bouncy_ball_setup(df: pd.DataFrame, bar_idx: int,
                             cfg: BouncyBallSettings = None) -> dict:
    """
    Check for bouncy ball MR setup at the given bar.

    Returns dict with:
      - passed: bool
      - reason: str (if failed)
      - Setup fields if passed: side, entry, sl, tp, etc.
    """
    if cfg is None:
        cfg = BouncyBallSettings()

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    i = bar_idx

    if i < cfg.max_range_bars + cfg.pre_trend_lookback:
        return {"passed": False, "reason": "not_enough_bars"}

    # --- Step 1: Find choppy range ---
    best_range = None

    for duration in [cfg.min_range_bars, 360]:
        if duration > cfg.max_range_bars:
            continue
        s0 = i - duration
        if s0 < 0:
            continue

        ch = highs[s0:i]
        cl = lows[s0:i]
        cc = closes[s0:i]

        upper = np.percentile(ch, cfg.percentile_upper)
        lower = np.percentile(cl, cfg.percentile_lower)

        range_pct = (upper - lower) / lower * 100
        if range_pct < cfg.min_range_pct or range_pct > cfg.max_range_pct:
            continue

        # Inside check
        inside_pct = np.mean((cc >= lower * 0.998) & (cc <= upper * 1.002)) * 100
        if inside_pct < cfg.min_inside_pct:
            continue

        # Touch quality
        ut, uo = _count_touches(ch, cl, cc, upper)
        lt, lo = _count_touches(ch, cl, cc, lower)

        if ut < cfg.min_touches_per_side or lt < cfg.min_touches_per_side:
            continue

        avg_os = (uo + lo) / 2
        if avg_os > cfg.max_overshoot_pct:
            continue

        # Cleanliness
        clean = _range_cleanliness(ch, cl, cc, upper, lower)
        if clean < cfg.min_clean_score:
            continue

        score = ut + lt + clean
        if best_range is None or score > best_range['score']:
            best_range = {
                'start': s0, 'duration': duration,
                'upper': upper, 'lower': lower,
                'upper_touches': ut, 'lower_touches': lt,
                'upper_overshoot': uo, 'lower_overshoot': lo,
                'range_pct': range_pct, 'inside_pct': inside_pct,
                'clean_score': clean, 'score': score,
            }

    if best_range is None:
        return {"passed": False, "reason": "no_range"}

    upper = best_range['upper']
    lower = best_range['lower']

    # --- Step 2: Price at a boundary? ---
    at_upper = highs[i] >= upper * 0.997
    at_lower = lows[i] <= lower * 1.003

    if not at_upper and not at_lower:
        return {"passed": False, "reason": "not_at_boundary"}

    # --- Step 3: Pre-chop trend direction ---
    pre_trend, pre_trend_pct = _detect_pre_trend(
        closes, highs, lows, best_range['start'], i, cfg.pre_trend_lookback)

    if at_upper:
        side = "short"
        if pre_trend == "up":
            return {"passed": False, "reason": "short_vs_uptrend"}
    else:
        side = "long"
        if pre_trend == "down":
            return {"passed": False, "reason": "long_vs_downtrend"}

    # --- Step 4: Confirmation — current bar closes back inside ---
    if side == "short" and closes[i] >= upper:
        return {"passed": False, "reason": "no_confirm_short"}
    if side == "long" and closes[i] <= lower:
        return {"passed": False, "reason": "no_confirm_long"}

    # --- Step 5: Entry + SL ---
    range_width = upper - lower
    if side == "short":
        entry = upper
        sl = upper * (1 + cfg.sl_buffer_pct / 100)
    else:
        entry = lower
        sl = lower * (1 - cfg.sl_buffer_pct / 100)

    sl_dist = abs(entry - sl)
    sl_pct = sl_dist / entry * 100

    if sl_pct <= 0:
        return {"passed": False, "reason": "invalid_sl"}
    if sl_pct < cfg.min_sl_pct:
        return {"passed": False, "reason": f"sl_too_small_{sl_pct:.2f}pct"}

    # --- Step 6: TP from swing depth or opposite boundary ---
    swing_touches = []
    if cfg.use_swing_tp:
        swing_touches = _find_real_swing_touches(
            highs, lows, closes, entry,
            best_range['start'], i, range_width, side,
            cfg.min_swing_travel_pct)

    if cfg.use_swing_tp and len(swing_touches) >= 1:
        # TP = shallowest retrace of last 3 swings (95% of it)
        recent = swing_touches[:3]
        depths = [s['retrace_depth'] for s in recent]
        tp_depth = min(depths) * 0.95

        if side == "short":
            tp = entry - tp_depth
        else:
            tp = entry + tp_depth

        tp_pct = tp_depth / entry * 100
        rr = tp_pct / sl_pct

        # Cap RR: if > max_rr, reduce to cap_rr
        if rr > cfg.max_rr:
            capped_tp_dist = sl_dist * cfg.cap_rr
            if side == "short":
                tp = entry - capped_tp_dist
            else:
                tp = entry + capped_tp_dist
            tp_pct = capped_tp_dist / entry * 100
            rr = cfg.cap_rr

        last_swing_bar = swing_touches[0]['bar']
        last_swing_mins = i - last_swing_bar
    else:
        # Fallback: opposite boundary
        if side == "short":
            tp = lower * (1 + cfg.tp_buffer_pct / 100)
            tp_pct = (entry - tp) / entry * 100
        else:
            tp = upper * (1 - cfg.tp_buffer_pct / 100)
            tp_pct = (tp - entry) / entry * 100

        rr = tp_pct / sl_pct if sl_pct > 0 else 0

        # Cap RR for fallback too
        if rr > cfg.max_rr:
            capped_tp_dist = sl_dist * cfg.cap_rr
            if side == "short":
                tp = entry - capped_tp_dist
            else:
                tp = entry + capped_tp_dist
            tp_pct = capped_tp_dist / entry * 100
            rr = cfg.cap_rr

        last_swing_mins = 999
        swing_touches = []

    if tp_pct <= 0:
        return {"passed": False, "reason": "invalid_tp"}
    if tp_pct < cfg.min_tp_pct:
        return {"passed": False, "reason": f"tp_too_small_{tp_pct:.2f}pct"}
    if rr < cfg.min_rr:
        return {"passed": False, "reason": f"rr_too_low_{rr:.2f}"}

    return {
        "passed": True,
        "side": side,
        "entry": round(entry, 8),
        "sl": round(sl, 8),
        "tp": round(tp, 8),
        "sl_pct": round(sl_pct, 3),
        "tp_pct": round(tp_pct, 3),
        "rr": round(rr, 2),
        "range_upper": round(upper, 8),
        "range_lower": round(lower, 8),
        "range_pct": round(best_range['range_pct'], 2),
        "upper_touches": best_range['upper_touches'],
        "lower_touches": best_range['lower_touches'],
        "clean_score": best_range['clean_score'],
        "inside_pct": round(best_range['inside_pct'], 1),
        "pre_trend": pre_trend,
        "pre_trend_pct": round(pre_trend_pct, 2),
        "range_duration_bars": best_range['duration'],
        "n_swing_touches": len(swing_touches),
        "last_swing_mins": last_swing_mins,
    }
