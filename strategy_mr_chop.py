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
    sl_pct: float = 1.0
    min_rr: float = 0.8
    max_rr: float = 1.5
    cap_rr: float = 1.25
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

        # TP from swing depth (match backtest exactly)
        recent = swings[-3:]
        depths = [s['depth_pct'] for s in recent]
        tp_depth_pct = min(depths) * 0.95

        entry = level
        sl_pct = cfg.sl_pct
        if side == "short":
            sl = level * (1 + sl_pct / 100)
            tp = level * (1 - tp_depth_pct / 100)
        else:
            sl = level * (1 - sl_pct / 100)
            tp = level * (1 + tp_depth_pct / 100)

        # TP% and RR calculated same as backtest
        tp_p = tp_depth_pct * 0.95
        rr = tp_p / sl_pct if sl_pct > 0 else 0

        # RR cap
        if rr > cfg.max_rr:
            rr = cfg.cap_rr
            cap_dist = abs(entry - sl) * cfg.cap_rr
            if side == "short":
                tp = entry - cap_dist
            else:
                tp = entry + cap_dist
            tp_p = cap_dist / entry * 100

        if rr < cfg.min_rr or tp_p < cfg.min_tp_pct:
            continue

        # DPS
        dur_score = 2 if chop_dur / 60 >= 4 else (1 if chop_dur / 60 >= 2 else 0)
        app_score = 2 if last_sw_mins <= 5 else (1 if last_sw_mins <= 15 else 0)
        vol_score, vol_type = _score_volume(v, i, side)
        dps = dur_score + app_score + vol_score

        depth_str = "/".join(f"{d:.1f}%" for d in depths)
        touch_str = "/".join(s['touch_type'] for s in recent)

        return {
            "passed": True,
            "strategy_variant": "mr_chop_v2",
            "side": side,
            "entry": round(entry, 8),
            "tp": round(tp, 8),
            "sl": round(sl, 8),
            "sl_pct": round(sl_pct, 3),
            "tp_pct": round(tp_p, 3),
            "rr": round(rr, 2),
            "n_swings": len(swings),
            "last_swing_mins": last_sw_mins,
            "swing_depths": depth_str,
            "touch_types": touch_str,
            "level_source": source,
            "chop_hrs": round(chop_dur / 60, 1),
            "pre_trend": pre,
            "pre_trend_pct": round(pre_pct, 2),
            "dps_total": dps,
            "dps_dur": dur_score,
            "dps_app": app_score,
            "dps_vol": vol_score,
            "vol_type": vol_type,
        }

    return {"passed": False, "reason": "no_valid_setup"}


def check_one_sided_chop_setup(df: pd.DataFrame, bar_idx: int,
                                cfg: MRChopSettings = None) -> dict:
    """Placeholder — v2 unified logic handles both via check_range_shift_setup."""
    return {"passed": False, "reason": "use_range_shift_v2"}
