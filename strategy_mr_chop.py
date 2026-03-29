#!/usr/bin/env python3
"""
Mean Reversion Chop Strategy v3

Level detection: Regression channel edges from the chop window
  - Low R² = sideways/choppy, channel width = range
  - Entry at channel edge, TP at opposite edge, SL beyond entry edge
Swing respect: validates that channel edges are actually respected
DPS: ZCT scoring (duration + approach + volume, side-specific)
Entry: limit order at level price, 3-min expiry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from scan_momo_quality import calc_regression as _calc_regression


@dataclass
class MRChopSettings:
    """Settings for MR chop strategy."""
    # Choppiness (R²-based)
    max_chop_r2: float = 0.3           # R² below this = choppy (no trend)
    max_chop_channel: float = 4.0      # max channel width % for chop range
    min_chop_channel: float = 1.0      # min channel — needs room for TP > SL
    min_chop_bars: int = 120           # min 2 hours of chop

    # Level respect
    max_breach_bars: int = 5
    min_respected_swings: int = 3       # need 3 prior bounces, enter on 4th+
    max_last_swing_mins: int = 30
    min_swing_depth_ratio: float = 0.3  # swing must bounce >= 30% of channel width
    min_containment_pct: float = 90.0   # % of bars that must be inside channel

    # Entry/exit
    sl_tp_ratio: float = 0.9           # SL = 90% of TP distance (RR ~1.11)
    tp_discount: float = 0.90          # TP targets 90% of channel width
    min_rr: float = 1.0
    min_tp_pct: float = 0.5

    # Pre-trend
    pre_trend_lookback: int = 240
    pre_trend_threshold: float = 1.0

    # Limit order
    limit_expiry_mins: int = 3


def _compute_channel(closes, highs, lows):
    """
    Compute regression R² for chop detection, plus percentile-based
    range bounds from actual highs/lows where price repeatedly turned.
    """
    n = len(closes)
    if n < 30:
        return None

    x = np.arange(n, dtype=float)
    xm, ym = np.mean(x), np.mean(closes)
    sxy = np.sum((x - xm) * (closes - ym))
    sxx = np.sum((x - xm) ** 2)
    syy = np.sum((closes - ym) ** 2)

    if sxx == 0 or syy == 0:
        return None

    slope = sxy / sxx
    r2 = (sxy ** 2) / (sxx * syy)

    # Range bounds from actual price action (percentile-based)
    # 95th percentile of highs = resistance, 5th percentile of lows = support
    # This captures where price actually turned, trimming spike wicks
    upper = np.percentile(highs, 95)
    lower = np.percentile(lows, 5)
    center = (upper + lower) / 2
    channel_pct = (upper - lower) / center * 100 if center > 0 else 0

    return {
        "r2": r2,
        "channel_pct": channel_pct,
        "center": center,
        "upper": upper,
        "lower": lower,
        "slope": slope,
        "slope_pct": slope / center * 100 if center > 0 else 0,
    }


def _detect_choppy(closes, highs, lows, end_bar, cfg: MRChopSettings):
    """
    Detect chop using R² regression — low R² = sideways/choppy.
    Walks backwards expanding the window to find where chop started.

    Returns (chop_start, chop_dur, channel_info) or None.
    """
    if end_bar < 240:
        return None

    # Step 1: Check if current 120-bar window is choppy
    ch = _compute_channel(closes[end_bar - 120:end_bar],
                          highs[end_bar - 120:end_bar],
                          lows[end_bar - 120:end_bar])
    if ch is None:
        return None
    if ch["r2"] >= cfg.max_chop_r2:
        return None
    if ch["channel_pct"] > cfg.max_chop_channel:
        return None
    if ch["channel_pct"] < cfg.min_chop_channel:
        return None

    # Step 2: Walk backwards to find where chop started
    chop_start = end_bar - 120
    max_lookback = min(end_bar, 720)

    for step_back in range(150, max_lookback, 30):
        s = end_bar - step_back
        if s < 0:
            break
        seg_ch = _compute_channel(closes[s:end_bar], highs[s:end_bar], lows[s:end_bar])
        if seg_ch is None:
            break
        if seg_ch["r2"] >= cfg.max_chop_r2:
            chop_start = s + 30
            break
        chop_start = s

    chop_dur = end_bar - chop_start
    if chop_dur < cfg.min_chop_bars:
        return None

    # Step 3: Recompute channel on the final chop window
    final = _compute_channel(closes[chop_start:end_bar],
                             highs[chop_start:end_bar],
                             lows[chop_start:end_bar])
    if final is None:
        return None
    if final["channel_pct"] > cfg.max_chop_channel:
        return None
    if final["channel_pct"] < cfg.min_chop_channel:
        return None

    return chop_start, chop_dur, final


def _detect_pre_trend(closes, highs, lows, chop_start, cfg: MRChopSettings):
    """How did price enter the chop zone?"""
    wide_start = max(0, chop_start - cfg.pre_trend_lookback)
    wide_end = min(len(closes), chop_start + 60)
    if wide_end - wide_start < 30:
        return "flat", 0.0

    peak = np.max(highs[wide_start:wide_end])
    trough = np.min(lows[wide_start:wide_end])
    current = np.mean(closes[chop_start:min(chop_start + 60, len(closes))])

    drop = (peak - current) / peak * 100
    rise = (current - trough) / trough * 100

    if drop > cfg.pre_trend_threshold and drop > rise:
        return "down", -drop
    if rise > cfg.pre_trend_threshold and rise > drop:
        return "up", rise
    return "flat", 0.0


def _find_respected_swings(highs, lows, closes, opens, level, start, end,
                            side, max_breach=5):
    """
    Find swings where price respected the level.
    For longs: price travels away from level upward, then returns to touch.
    For shorts: price travels away from level downward, then returns to touch.
    """
    swings = []
    farthest = 0
    breach_count = 0

    for j in range(start, end):
        c_val = closes[j]
        h_val = highs[j]
        l_val = lows[j]

        if side == "long":
            dist_away = h_val - level
            if dist_away > farthest:
                farthest = dist_away

            close_below = c_val < level
            wick_touched = l_val <= level
            close_at = abs(c_val - level) / level * 100 < 0.1

            if close_below:
                breach_count += 1
                if breach_count > max_breach:
                    farthest = 0
                    breach_count = 0
            elif wick_touched or close_at:
                if farthest > 0:
                    depth_pct = farthest / level * 100
                    touch = 'wick' if (not close_at and wick_touched) else 'body'
                    swings.append({
                        'bar': j, 'depth_pct': depth_pct,
                        'depth': farthest, 'touch_type': touch,
                    })
                farthest = 0
                breach_count = 0
            else:
                breach_count = 0

        else:  # short
            dist_away = level - l_val
            if dist_away > farthest:
                farthest = dist_away

            close_above = c_val > level
            wick_touched = h_val >= level
            close_at = abs(c_val - level) / level * 100 < 0.1

            if close_above:
                breach_count += 1
                if breach_count > max_breach:
                    farthest = 0
                    breach_count = 0
            elif wick_touched or close_at:
                if farthest > 0:
                    depth_pct = farthest / level * 100
                    touch = 'wick' if (not close_at and wick_touched) else 'body'
                    swings.append({
                        'bar': j, 'depth_pct': depth_pct,
                        'depth': farthest, 'touch_type': touch,
                    })
                farthest = 0
                breach_count = 0
            else:
                breach_count = 0

    return swings


def _score_volume(volumes, bar, side, lookback=60, skip=3):
    """ZCT MR volume scoring."""
    s = max(0, bar - lookback)
    e = bar - skip
    vol = volumes[s:e]
    if len(vol) < 20:
        return 1, "unclear"
    slope = np.polyfit(np.arange(len(vol)), vol, 1)[0]
    avg = np.mean(vol)
    if avg <= 0:
        return 1, "unclear"
    norm = slope / avg
    if norm > 0.005:
        vt = "increasing"
    elif norm < -0.005:
        vt = "decreasing"
    else:
        vt = "flat"
    if side == "long":
        return (2 if vt == "flat" else (1 if vt == "decreasing" else 0)), vt
    else:
        return (2 if vt == "decreasing" else (1 if vt == "flat" else 0)), vt


def check_range_shift_setup(df: pd.DataFrame, bar_idx: int,
                             cfg: MRChopSettings = None) -> dict:
    """
    Check for MR chop setup using regression channel edges.
    Entry at channel boundary, TP targets opposite side, SL beyond entry edge.
    """
    if cfg is None:
        cfg = MRChopSettings()

    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    o = df["open"].values
    v = df["volume"].values if "volume" in df.columns else np.ones(len(c))
    i = bar_idx

    if i < 600:
        return {"passed": False, "reason": "not_enough_bars"}

    # Choppy? Get channel bounds
    chop = _detect_choppy(c, h, l, i, cfg)
    if chop is None:
        return {"passed": False, "reason": "not_choppy"}
    chop_start, chop_dur, ch_info = chop

    # Pre-trend
    pre, pre_pct = _detect_pre_trend(c, h, l, chop_start, cfg)

    # Channel levels
    upper_level = ch_info["upper"]
    lower_level = ch_info["lower"]
    center = ch_info["center"]
    channel_width_pct = ch_info["channel_pct"]

    # Containment check: 90%+ of bars must be inside the channel
    chop_h = h[chop_start:i]
    chop_l = l[chop_start:i]
    buf = (upper_level - lower_level) * 0.05  # 5% buffer on each side
    inside = np.sum((chop_h <= upper_level + buf) & (chop_l >= lower_level - buf))
    containment = inside / len(chop_h) * 100 if len(chop_h) > 0 else 0
    if containment < cfg.min_containment_pct:
        return {"passed": False, "reason": f"containment_{containment:.0f}pct"}

    # Build candidates: long at lower channel, short at upper channel
    candidates = []
    if pre != "down":  # don't long if pre-trend is down
        candidates.append({"price": lower_level, "side": "long", "source": "ch_lower"})
    if pre != "up":    # don't short if pre-trend is up
        candidates.append({"price": upper_level, "side": "short", "source": "ch_upper"})

    if not candidates:
        return {"passed": False, "reason": "no_aligned_levels"}

    for cand in candidates:
        level = cand["price"]
        side = cand["side"]
        source = cand["source"]

        # Respected swings at this channel edge
        swings = _find_respected_swings(h, l, c, o, level, chop_start, i,
                                         side, cfg.max_breach_bars)

        # Filter swings by minimum bounce depth (% of channel width)
        min_depth = channel_width_pct * cfg.min_swing_depth_ratio
        swings = [s for s in swings if s['depth_pct'] >= min_depth]

        if len(swings) < cfg.min_respected_swings:
            continue

        # Last swing within 30 min
        last_sw_mins = i - swings[-1]['bar']
        if last_sw_mins > cfg.max_last_swing_mins:
            continue

        # Current bar: wick must reach level, close on correct side
        if side == "short":
            if h[i] < level * 0.998:
                continue
            if c[i] >= level:
                continue
        else:
            if l[i] > level * 1.002:
                continue
            if c[i] <= level:
                continue

        # --- TP/SL from channel ---
        entry = level
        channel_dist = upper_level - lower_level  # full channel width in price

        # TP: target opposite channel edge, discounted
        tp_dist = channel_dist * cfg.tp_discount
        if side == "long":
            tp = entry + tp_dist
        else:
            tp = entry - tp_dist

        # SL: 90% of TP distance, placed beyond entry edge
        sl_dist_price = tp_dist * cfg.sl_tp_ratio
        if side == "long":
            sl = entry - sl_dist_price
        else:
            sl = entry + sl_dist_price

        # Compute percentages
        tp_p = tp_dist / entry * 100
        sl_p = sl_dist_price / entry * 100
        rr = tp_p / sl_p if sl_p > 0 else 0

        # Filters
        if rr < cfg.min_rr or tp_p < cfg.min_tp_pct or sl_p < 0.1:
            continue

        # Approach quality
        last_swing = swings[-1]
        last_swing_bar = last_swing['bar']
        prev_touch_bar = swings[-2]['bar'] if len(swings) >= 2 else max(chop_start, last_swing_bar - 60)

        if side == "long":
            segment_h = h[prev_touch_bar:last_swing_bar]
            if len(segment_h) > 0:
                peak_idx = prev_touch_bar + np.argmax(segment_h)
                journey_bars = last_swing_bar - peak_idx
                journey_pct = (h[peak_idx] - level) / level * 100
            else:
                journey_bars, journey_pct = 1, 0
        else:
            segment_l = l[prev_touch_bar:last_swing_bar]
            if len(segment_l) > 0:
                trough_idx = prev_touch_bar + np.argmin(segment_l)
                journey_bars = last_swing_bar - trough_idx
                journey_pct = (level - l[trough_idx]) / level * 100
            else:
                journey_bars, journey_pct = 1, 0

        if journey_bars > 0 and journey_pct > 0:
            if journey_bars <= 3 and journey_pct >= 0.5:
                approach_type = "spike"
                approach_qual = 2
            elif journey_pct / journey_bars >= 0.1:
                approach_type = "unclear"
                approach_qual = 1
            else:
                approach_type = "grind"
                approach_qual = 0
        else:
            approach_type = "unclear"
            approach_qual = 1

        # Recency
        recency_score = 2 if last_sw_mins <= 5 else (1 if last_sw_mins <= 15 else 0)

        # DPS = duration + approach + volume
        dur_score = 2 if chop_dur / 60 >= 4 else (1 if chop_dur / 60 >= 2 else 0)
        vol_score, vol_type = _score_volume(v, i, side)
        dps = dur_score + approach_qual + vol_score

        recent = swings[-3:]
        depth_str = "/".join(f"{d['depth_pct']:.1f}%" for d in recent)
        touch_str = "/".join(s['touch_type'] for s in recent)

        return {
            "passed": True,
            "strategy_variant": "mr_chop_v3",
            "side": side,
            "entry": round(entry, 8),
            "tp": round(tp, 8),
            "sl": round(sl, 8),
            "sl_pct": round(sl_p, 3),
            "tp_pct": round(tp_p, 3),
            "rr": round(rr, 2),
            "n_swings": len(swings),
            "last_swing_mins": last_sw_mins,
            "recency_score": recency_score,
            "swing_depths": depth_str,
            "touch_types": touch_str,
            "approach_type": approach_type,
            "approach_score": approach_qual,
            "level_source": source,
            "chop_hrs": round(chop_dur / 60, 1),
            "containment_pct": round(containment, 1),
            "channel_pct": round(channel_width_pct, 3),
            "r2": round(ch_info["r2"], 4),
            "ch_upper": round(upper_level, 8),
            "ch_lower": round(lower_level, 8),
            "pre_trend": pre,
            "pre_trend_pct": round(pre_pct, 2),
            "dps_total": dps,
            "dps_dur": dur_score,
            "dps_app": approach_qual,
            "dps_vol": vol_score,
            "vol_type": vol_type,
        }

    return {"passed": False, "reason": "no_valid_setup"}


def check_one_sided_chop_setup(df: pd.DataFrame, bar_idx: int,
                                cfg: MRChopSettings = None) -> dict:
    """Placeholder — v3 unified logic handles both via check_range_shift_setup."""
    return {"passed": False, "reason": "use_range_shift_v3"}
