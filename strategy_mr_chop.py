#!/usr/bin/env python3
"""
Mean Reversion Chop Strategy v2

Level detection: 15-min and 1-hour high/low candle bodies
Swing respect: max 5 consecutive closes beyond level, then recovery
TP: shallowest of last 3 respected swing depths × 0.95
SL: 1% beyond level
DPS: ZCT scoring (duration + approach + volume, side-specific)
Entry: limit order at level price, 3-min expiry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MRChopSettings:
    """Settings for MR chop strategy."""
    # Choppiness
    min_chop_ratio: float = 0.25
    chop_durations: tuple = (360, 180)

    # Level respect
    max_breach_bars: int = 5
    min_respected_swings: int = 3
    max_last_swing_mins: int = 30

    # Entry/exit
    sl_pct: float = 1.0              # fallback SL if < 3 swings
    min_rr: float = 1.0
    max_rr: float = 1.5
    min_tp_pct: float = 0.5

    # Pre-trend
    pre_trend_lookback: int = 240
    pre_trend_threshold: float = 1.0

    # Limit order
    limit_expiry_mins: int = 3


def _detect_choppy(closes, end_bar, cfg: MRChopSettings):
    """Detect if recent price action is choppy."""
    for dur in cfg.chop_durations:
        s = end_bar - dur
        if s < 0:
            continue
        cc = closes[s:end_bar]
        if len(cc) < 100:
            continue
        smooth = np.convolve(cc, np.ones(10) / 10, mode='valid')
        if len(smooth) < 30:
            continue
        diffs = np.diff(smooth)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        ratio = sign_changes / len(diffs)
        if ratio > cfg.min_chop_ratio:
            return s, dur
    return None


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


def _find_tf_levels(highs, lows, closes, opens, end_bar):
    """Find levels from 15-min and 1-hour candle highs/lows."""
    levels = []

    for tf_bars, tf_name in [(15, '15m'), (60, '1h')]:
        s = max(0, end_bar - tf_bars)
        h_max = np.max(highs[s:end_bar])
        l_min = np.min(lows[s:end_bar])

        # Resistance: bodies of bars at the high
        h_bodies = [max(opens[j], closes[j]) for j in range(s, end_bar)
                     if highs[j] >= h_max * 0.999]
        if h_bodies:
            levels.append({
                'price': np.mean(h_bodies),
                'source': f'{tf_name}_high',
                'side': 'short',
            })

        # Support: bodies of bars at the low
        l_bodies = [min(opens[j], closes[j]) for j in range(s, end_bar)
                     if lows[j] <= l_min * 1.001]
        if l_bodies:
            levels.append({
                'price': np.mean(l_bodies),
                'source': f'{tf_name}_low',
                'side': 'long',
            })

    return levels


def _find_respected_swings(highs, lows, closes, opens, level, start, end,
                            side, max_breach=5):
    """
    Find swings where price respected the level.
    For longs: max N closes below level, then recovery above.
    For shorts: max N closes above level, then recovery below.
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
    Check for MR chop setup using 15m/1h levels + respected swings.
    This replaces both range_shift and one_sided_chop with unified v2 logic.
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

    # Choppy?
    chop = _detect_choppy(c, i, cfg)
    if chop is None:
        return {"passed": False, "reason": "not_choppy"}
    chop_start, chop_dur = chop

    # Pre-trend
    pre, pre_pct = _detect_pre_trend(c, h, l, chop_start, cfg)

    # Get candidate levels
    tf_levels = _find_tf_levels(h, l, c, o, i)
    if not tf_levels:
        return {"passed": False, "reason": "no_levels"}

    # Filter by pre-trend
    if pre == "up":
        candidates = [lv for lv in tf_levels if lv['side'] == 'long']
    elif pre == "down":
        candidates = [lv for lv in tf_levels if lv['side'] == 'short']
    else:
        candidates = tf_levels

    if not candidates:
        return {"passed": False, "reason": "no_aligned_levels"}

    for cand in candidates:
        level = cand['price']
        side = cand['side']
        source = cand['source']

        # Respected swings
        swings = _find_respected_swings(h, l, c, o, level, chop_start, i,
                                         side, cfg.max_breach_bars)
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

        # TP from swing depth (no cap)
        recent = swings[-3:]
        depths = [s['depth_pct'] for s in recent]
        tp_depth_pct = min(depths) * 0.95

        entry = level

        # SL at second swing low/high, capped at 1.5%
        max_sl_pct = 1.5
        if len(swings) >= 3:
            s1_bar = swings[-3]['bar']
            s2_bar = swings[-2]['bar']
            if side == "short":
                swing_extreme = np.max(h[s1_bar:s2_bar]) if s2_bar > s1_bar else h[s1_bar]
                sl = swing_extreme * 1.002
            else:
                swing_extreme = np.min(l[s1_bar:s2_bar]) if s2_bar > s1_bar else l[s1_bar]
                sl = swing_extreme * 0.998
            # If SL > 1.5%, use SL = 90% of TP instead
            sl_check = abs(entry - sl) / entry * 100
            if sl_check > max_sl_pct:
                fallback_sl_dist = tp_depth_pct * 0.95 * 0.9 / 100 * entry
                if side == "short":
                    sl = entry + fallback_sl_dist
                else:
                    sl = entry - fallback_sl_dist
        else:
            if side == "short":
                sl = level * (1 + cfg.sl_pct / 100)
            else:
                sl = level * (1 - cfg.sl_pct / 100)

        if side == "short":
            tp = level * (1 - tp_depth_pct / 100)
        else:
            tp = level * (1 + tp_depth_pct / 100)

        sl_dist = abs(entry - sl)
        sl_p = sl_dist / entry * 100
        tp_p = tp_depth_pct * 0.95
        rr = tp_p / sl_p if sl_p > 0 else 0

        # RR filter: 1.0 - 1.5
        if rr < 1.0 or rr > 1.5 or tp_p < cfg.min_tp_pct or sl_p < 0.3:
            continue

        # Approach quality: full swing from opposite extreme to level
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

        depth_str = "/".join(f"{d:.1f}%" for d in depths)
        touch_str = "/".join(s['touch_type'] for s in recent)

        return {
            "passed": True,
            "strategy_variant": "mr_chop_v2",
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
    """Placeholder — v2 unified logic handles both via check_range_shift_setup."""
    return {"passed": False, "reason": "use_range_shift_v2"}
