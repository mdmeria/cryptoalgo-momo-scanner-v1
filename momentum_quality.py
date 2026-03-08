"""Momentum setup quality filter based on playbook-style checklist rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import requests


@dataclass
class MomentumQualityResult:
    """Output container for momentum setup validation."""

    direction: str
    score: float
    passed: bool
    checks: Dict[str, bool]
    metrics: Dict[str, float]
    quality_tier: str = "unknown"  # "high" or "low"


def _smma(values: pd.Series, length: int) -> pd.Series:
    """Smoothed moving average (SMMA/RMA style)."""
    return values.ewm(alpha=1.0 / length, adjust=False).mean()


def _fetch_ticker_24h(symbol: str) -> dict:
    """Fetch 24h ticker stats for a symbol from Binance Vision-compatible endpoint."""
    url = "https://data-api.binance.vision/api/v3/ticker/24hr"
    response = requests.get(url, params={"symbol": symbol.upper()}, timeout=8)
    if response.status_code != 200:
        return {}
    return response.json()


def _fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch klines and return OHLCV DataFrame."""
    url = "https://data-api.binance.vision/api/v3/klines"
    response = requests.get(
        url,
        params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
        timeout=8,
    )
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


def _session_vwap(df: pd.DataFrame) -> float:
    """Compute volume-weighted average price using typical price."""
    if df.empty:
        return float("nan")
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    denom = float(vol.sum())
    if denom <= 0:
        return float("nan")
    return float((tp * vol).sum() / denom)


def _linear_slope(values: pd.Series) -> float:
    """Return normalized slope per bar for the given series."""
    clean = values.dropna()
    if len(clean) < 5:
        return 0.0
    x = np.arange(len(clean), dtype=float)
    y = clean.to_numpy(dtype=float)
    if np.allclose(y, y[0]):
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope / max(abs(np.mean(y)), 1e-9))


def _detect_counter_retracements(
    df: pd.DataFrame, 
    direction: str, 
    lookback_bars: int = 120,
    window_size: int = 20,
    retracement_threshold_pct: float = 2.0
) -> tuple[bool, int]:
    """
    Detect counter-trend retracements that would threaten tight stops.
    
    For SHORT setups: looks for upward rallies from local lows
    For LONG setups: looks for downward drops from local highs
    
    Returns:
        (has_retracements, num_retracements)
        - has_retracements: True if any significant counter-moves detected
        - num_retracements: count of retracement periods found
    """
    if df is None or len(df) < lookback_bars:
        return False, 0
    
    is_long = direction.lower() == "long"
    data = df.iloc[-lookback_bars:].copy()
    
    retracements_found = 0
    threshold = retracement_threshold_pct / 100.0  # Convert to decimal
    
    # Slide through the lookback period in windows
    for i in range(0, len(data) - window_size + 1, window_size // 2):  # 50% overlap
        window = data.iloc[i:i+window_size]
        
        if len(window) < window_size:
            continue
            
        if is_long:
            # For LONG: check if price dropped significantly from local high
            local_high = float(window["high"].max())
            local_low = float(window["low"].min())
            
            if local_high > 0:
                retracement_pct = (local_high - local_low) / local_high
                if retracement_pct >= threshold:
                    retracements_found += 1
        else:
            # For SHORT: check if price rallied significantly from local low
            local_low = float(window["low"].min())
            local_high = float(window["high"].max())
            
            if local_low > 0:
                retracement_pct = (local_high - local_low) / local_low
                if retracement_pct >= threshold:
                    retracements_found += 1
    
    return retracements_found > 0, retracements_found


def _analyze_8min_grind_quality(df: pd.DataFrame, direction: str) -> Dict[str, float]:
    """
    Analyze 8-minute windows to distinguish true directional grind from spike/sideways action.
    
    Grind characteristics:
    - Multiple 8-min windows showing consistent directional progress (not one spike)
    - Average impulse in "grind zone" (0.25-0.80%), not too weak or explosive
    - Low variance across windows (consistency, not erratic)
    - Movement distributed across bars within windows (not concentrated in 1-2 bars)
    """
    if df is None or len(df) < 120:
        return {
            "grind_windows_count": 0,
            "avg_grind_impulse_pct": 0.0,
            "grind_impulse_std": 0.0,
            "grind_bar_participation": 0.0,
        }
    
    is_long = direction.lower() == "long"
    left = df.iloc[-120:].copy()
    close = left["close"]
    
    # Calculate all 8-minute directional moves
    move_8m = close.pct_change(8).dropna()
    dir_move_8m = move_8m if is_long else -move_8m
    
    # Filter to meaningful directional windows (>= 0.2% - lowered for distributed grinds)
    grind_threshold = 0.002
    meaningful_moves = dir_move_8m[dir_move_8m >= grind_threshold]
    grind_windows_count = len(meaningful_moves)
    
    # Calculate average and std dev of meaningful moves
    if len(meaningful_moves) > 0:
        avg_grind_impulse = float(meaningful_moves.mean())
        grind_impulse_std = float(meaningful_moves.std()) if len(meaningful_moves) > 1 else 0.0
    else:
        avg_grind_impulse = 0.0
        grind_impulse_std = 0.0
    
    # Analyze bar participation: for windows with good impulse, 
    # check what % of bars contributed to directional movement
    bar_participations = []
    for i in range(8, len(close)):
        window_pct = (close.iloc[i] - close.iloc[i-8]) / max(abs(close.iloc[i-8]), 1e-9)
        dir_window_pct = window_pct if is_long else -window_pct
        
        # Only analyze windows with meaningful directional movement
        if dir_window_pct >= grind_threshold:
            window_bars = close.iloc[i-8:i+1]
            bar_changes = window_bars.diff().dropna()
            dir_bars = (bar_changes > 0).sum() if is_long else (bar_changes < 0).sum()
            participation = float(dir_bars) / len(bar_changes) if len(bar_changes) > 0 else 0.0
            bar_participations.append(participation)
    
    avg_bar_participation = float(np.mean(bar_participations)) if bar_participations else 0.0
    
    return {
        "grind_windows_count": grind_windows_count,
        "avg_grind_impulse_pct": avg_grind_impulse * 100.0,
        "grind_impulse_std": grind_impulse_std,
        "grind_bar_participation": avg_bar_participation,
    }


def _balanced_momo_profile_2h(df: pd.DataFrame, direction: str) -> tuple[bool, Dict[str, float]]:
    """
    Validate 2h momentum profile sits between sideways and spike behavior.

    The profile should be directional and structured (not chop), but also not
    so explosive that it resembles exhaustion-style spike action.
    
    Enhanced with 8-minute grind quality analysis to avoid both spikes and sideways chop.
    """
    if df is None or len(df) < 120:
        return False, {
            "dir_move_2h_pct": 0.0,
            "dir_bar_ratio_2h": 0.0,
            "max_dir_impulse_8m_pct": 0.0,
            "efficiency_2h": 0.0,
            "grind_windows_count": 0,
            "avg_grind_impulse_pct": 0.0,
            "grind_impulse_std": 0.0,
            "grind_bar_participation": 0.0,
        }

    is_long = direction.lower() == "long"
    left = df.iloc[-120:].copy()
    close = left["close"]
    start = float(close.iloc[0])
    end = float(close.iloc[-1])

    raw_move = (end - start) / max(abs(start), 1e-9)
    dir_move = raw_move if is_long else -raw_move

    diff = close.diff().fillna(0.0)
    dir_bars = (diff > 0).sum() if is_long else (diff < 0).sum()
    dir_bar_ratio = float(dir_bars) / max(len(diff), 1)

    move_8m = close.pct_change(8).dropna()
    if len(move_8m) == 0:
        max_dir_impulse_8m = 0.0
    else:
        dir_move_8m = move_8m if is_long else -move_8m
        max_dir_impulse_8m = float(max(0.0, dir_move_8m.max()))

    path = close.pct_change().abs().fillna(0.0).sum()
    efficiency = float(dir_move / max(float(path), 1e-9))

    # Get detailed 8-minute grind analysis
    grind_metrics = _analyze_8min_grind_quality(df, direction)

    # 2h profile must show:
    # 1. Meaningful directional movement (0.4-12%)
    # 2. Not over-choppy (<=90% directional bars)
    # 3. Max 8-min impulse <= 1.5% (not a single explosive spike)
    # 4. Efficiency between 10-92%
    # 5. Grind quality: majority of windows show progress (>=50% of ~14 windows at 0.2% threshold)
    # 6. Average grind impulse in realistic range (0.3-1.2% - calibrated from real grinds)
    # 7. Moderate consistency (std <=0.45% - allows natural variation)
    # 8. Decent bar participation (>=40% bars contributing in grind windows)
    balanced_ok = (
        0.004 <= dir_move <= 0.120
        and dir_bar_ratio <= 0.90
        and max_dir_impulse_8m <= 0.015  # Raised from 1.0% to 1.5%
        and 0.10 <= efficiency <= 0.92
        and grind_metrics["grind_windows_count"] >= 7  # 50% of ~14 possible
        and 0.30 <= grind_metrics["avg_grind_impulse_pct"] <= 1.20  # Real grinds are 0.9-1.05%
        and grind_metrics["grind_impulse_std"] <= 0.0045  # 0.45% std (real grinds: 0.33-0.40%)
        and grind_metrics["grind_bar_participation"] >= 0.40  # Real grinds: 47-65%
    )

    return balanced_ok, {
        "dir_move_2h_pct": float(dir_move * 100.0),
        "dir_bar_ratio_2h": float(dir_bar_ratio),
        "max_dir_impulse_8m_pct": float(max_dir_impulse_8m * 100.0),
        "efficiency_2h": float(efficiency),
        "grind_windows_count": grind_metrics["grind_windows_count"],
        "avg_grind_impulse_pct": grind_metrics["avg_grind_impulse_pct"],
        "grind_impulse_std": grind_metrics["grind_impulse_std"],
        "grind_bar_participation": grind_metrics["grind_bar_participation"],
    }


def _price_parallel_to_smma30_2h(df: pd.DataFrame, direction: str) -> tuple[bool, Dict[str, float]]:
    """Check price has stayed mostly on one side of 30SMMA for last 2h with limited crossovers."""
    if df is None or len(df) < 120:
        return False, {
            "smma30_crosses_2h": 999.0,
            "smma30_trend_side_ratio_2h": 0.0,
        }

    left = df.iloc[-120:]
    is_long = direction.lower() == "long"

    # Treat tiny deviations as "touches" so minor contacts are allowed.
    tol = left["close"] * 0.0015  # 0.15%
    rel = left["close"] - left["smma30"]
    state = pd.Series(np.where(rel > tol, 1, np.where(rel < -tol, -1, 0)), index=left.index)

    # Forward-fill touch bars with previous side to avoid false cross counts.
    state_filled = state.replace(0, np.nan).ffill().fillna(0)
    transitions = np.diff(state_filled.to_numpy(dtype=float))
    cross_count = int(np.sum(np.abs(transitions) == 2))

    if is_long:
        trend_side_ratio = float((state_filled >= 0).sum()) / max(len(state_filled), 1)
    else:
        trend_side_ratio = float((state_filled <= 0).sum()) / max(len(state_filled), 1)

    parallel_ok = cross_count <= 4 and trend_side_ratio >= 0.85

    return parallel_ok, {
        "smma30_crosses_2h": float(cross_count),
        "smma30_trend_side_ratio_2h": float(trend_side_ratio),
    }


def _smma_spread_slowly_increasing_2h(df: pd.DataFrame, direction: str) -> tuple[bool, Dict[str, float]]:
    """Check the gap between 30SMMA and 120SMMA is gradually widening in signal direction."""
    if df is None or len(df) < 120:
        return False, {
            "smma_spread_slope_2h": 0.0,
            "smma_spread_up_ratio_2h": 0.0,
        }

    left = df.iloc[-120:]
    is_long = direction.lower() == "long"

    if is_long:
        spread = (left["smma30"] - left["smma120"]).clip(lower=0)
    else:
        spread = (left["smma120"] - left["smma30"]).clip(lower=0)

    spread_norm = spread / left["close"].replace(0, np.nan)
    spread_norm = spread_norm.fillna(0.0)
    spread_slope = _linear_slope(spread_norm)
    spread_delta = spread_norm.diff().fillna(0.0)
    spread_up_ratio = float((spread_delta > 0).sum()) / max(len(spread_delta), 1)

    spread_ok = spread_slope > 0 and spread_up_ratio >= 0.52

    return spread_ok, {
        "smma_spread_slope_2h": float(spread_slope),
        "smma_spread_up_ratio_2h": float(spread_up_ratio),
    }


def evaluate_momentum_setup(
    df: pd.DataFrame,
    direction: str,
    min_quality_score: float = 0.60,
    symbol: str | None = None,
    enforce_extended_rules: bool = False,
) -> MomentumQualityResult:
    """
    Evaluate whether the current momentum signal is high quality.

    Rules approximate the documented momentum checklist:
    - clean approach to level over last 1-10 minutes
    - staircase-style context on the left side (roughly 2h)
    - avoid decreasing volume profile
    - reject obvious chop
    """
    if df is None or len(df) < 120:
        return MomentumQualityResult(
            direction=direction,
            score=0.0,
            passed=False,
            checks={
                "enough_data": False,
                "slow_grind_approach": False,
                "left_side_staircase": False,
                "volume_not_decreasing": False,
                "not_choppy": False,
                "balanced_momo_2h": False,
                "parallel_to_smma30_2h": False,
                "smma_spread_increasing_2h": False,
            },
            metrics={"bars": float(len(df) if df is not None else 0)},
        )

    side = direction.lower()
    is_long = side == "long"

    work = df.copy()
    work["smma30"] = _smma(work["close"], 30)
    work["smma120"] = _smma(work["close"], 120)
    work["vol_ma20"] = work["volume"].rolling(20).mean()

    # Rule 1: Approach quality (last ~10 bars should grind in the signal direction)
    approach = work.iloc[-10:]
    close_start = float(approach["close"].iloc[0])
    close_end = float(approach["close"].iloc[-1])
    net_move = (close_end - close_start) / max(abs(close_start), 1e-9)

    if is_long:
        opposite_candles = int((approach["close"].diff() < 0).sum())
        approach_ok = net_move > 0 and opposite_candles <= 4
    else:
        opposite_candles = int((approach["close"].diff() > 0).sum())
        approach_ok = net_move < 0 and opposite_candles <= 4

    # Rule 2: Left side staircase context (roughly last 2h)
    left = work.iloc[-120:]
    smma30_slope = _linear_slope(left["smma30"])
    smma120_slope = _linear_slope(left["smma120"])
    if is_long:
        staircase_bars = int((left["close"] > left["smma30"]).sum())
        trend_stack_bars = int((left["smma30"] > left["smma120"]).sum())
        staircase_ok = (
            smma30_slope > 0
            and smma120_slope > 0
            and trend_stack_bars >= 84  # >=70% of last 120 bars
            and staircase_bars >= 72     # >=60% of bars above 30SMMA
            and float(left["close"].iloc[-1]) > float(left["smma30"].iloc[-1])
        )
    else:
        staircase_bars = int((left["close"] < left["smma30"]).sum())
        trend_stack_bars = int((left["smma30"] < left["smma120"]).sum())
        staircase_ok = (
            smma30_slope < 0
            and smma120_slope < 0
            and trend_stack_bars >= 84  # >=70% of last 120 bars
            and staircase_bars >= 72     # >=60% of bars below 30SMMA
            and float(left["close"].iloc[-1]) < float(left["smma30"].iloc[-1])
        )

    # Rule 3: Volume profile should not be clearly decaying
    vol_slice = work["vol_ma20"].iloc[-30:]
    vol_slope = _linear_slope(vol_slice)
    volume_ok = vol_slope > -0.02

    # Rule 4: Anti-chop filter
    # If price keeps crossing 30SMMA repeatedly, it behaves like chop.
    side_of_smma = np.sign(work["close"].iloc[-60:] - work["smma30"].iloc[-60:])
    side_changes = int(np.sum(np.abs(np.diff(side_of_smma.fillna(0).to_numpy())) > 0))
    choppy = side_changes >= 14
    not_choppy = not choppy

    # Rule 5: 2h profile should be between sideways and spike behavior.
    balanced_momo_2h, balanced_metrics = _balanced_momo_profile_2h(work, side)

    # Rule 6: Price should stay parallel to 30SMMA (limited criss-cross).
    parallel_to_smma30_2h, parallel_metrics = _price_parallel_to_smma30_2h(work, side)

    # Rule 7: Distance between 30SMMA and 120SMMA should gradually increase.
    smma_spread_increasing_2h, spread_metrics = _smma_spread_slowly_increasing_2h(work, side)

    checks = {
        "enough_data": True,
        "slow_grind_approach": approach_ok,
        "left_side_staircase": staircase_ok,
        "volume_not_decreasing": volume_ok,
        "not_choppy": not_choppy,
        "balanced_momo_2h": balanced_momo_2h,
        "parallel_to_smma30_2h": parallel_to_smma30_2h,
        "smma_spread_increasing_2h": smma_spread_increasing_2h,
        "day_change_ok": True,
        "vwap_side_ok": True,
        "first_2h_prev_day_vwap_ok": True,
        "entry_not_crossed_6h": True,
    }

    day_change_pct = float("nan")
    current_vwap = float("nan")
    prev_day_vwap = float("nan")
    first_2h = False
    crossed_count_6h = -1

    if enforce_extended_rules:
        if symbol is None:
            checks["day_change_ok"] = False
            checks["vwap_side_ok"] = False
            checks["first_2h_prev_day_vwap_ok"] = False
        else:
            # Rule A: day change threshold (>= +5% for long, <= -5% for short)
            ticker = _fetch_ticker_24h(symbol)
            if ticker:
                day_change_pct = float(ticker.get("priceChangePercent", 0.0))
            else:
                day_change_pct = float("nan")

            if is_long:
                checks["day_change_ok"] = bool(day_change_pct >= 5.0)
            else:
                checks["day_change_ok"] = bool(day_change_pct <= -5.0)

            # Rule B/C: VWAP side + first 2h previous day VWAP requirement
            vwap_df = _fetch_klines(symbol, interval="5m", limit=600)
            if not vwap_df.empty:
                now_ts = pd.Timestamp.utcnow()
                if now_ts.tzinfo is None:
                    now_ts = now_ts.tz_localize("UTC")

                today = now_ts.date()
                yesterday = (now_ts - pd.Timedelta(days=1)).date()

                today_df = vwap_df[vwap_df.index.date == today]
                prev_df = vwap_df[vwap_df.index.date == yesterday]

                current_vwap = _session_vwap(today_df)
                prev_day_vwap = _session_vwap(prev_df)
                close_now = float(df["close"].iloc[-1])

                if is_long:
                    checks["vwap_side_ok"] = bool(np.isfinite(current_vwap) and close_now > current_vwap)
                else:
                    checks["vwap_side_ok"] = bool(np.isfinite(current_vwap) and close_now < current_vwap)

                first_2h = now_ts.hour < 2
                if first_2h:
                    if is_long:
                        checks["first_2h_prev_day_vwap_ok"] = bool(
                            np.isfinite(prev_day_vwap) and close_now > prev_day_vwap and day_change_pct >= 5.0
                        )
                    else:
                        checks["first_2h_prev_day_vwap_ok"] = bool(
                            np.isfinite(prev_day_vwap) and close_now < prev_day_vwap and day_change_pct <= -5.0
                        )
                else:
                    checks["first_2h_prev_day_vwap_ok"] = True
            else:
                checks["vwap_side_ok"] = False
                checks["first_2h_prev_day_vwap_ok"] = False

        # Rule D: entry price should not have been crossed in last 6h.
        # Approximation: breakout entry level from last 10 bars must be untouched in prior 6h window.
        if len(work) >= 370:
            if is_long:
                entry_price = float(work["high"].iloc[-10:].max())
            else:
                entry_price = float(work["low"].iloc[-10:].min())

            prior_window = work.iloc[-370:-10]
            crossed = (prior_window["low"] <= entry_price) & (prior_window["high"] >= entry_price)
            crossed_count_6h = int(crossed.sum())
            checks["entry_not_crossed_6h"] = crossed_count_6h == 0
        else:
            checks["entry_not_crossed_6h"] = False

    weights = {
        "slow_grind_approach": 0.35,
        "left_side_staircase": 0.35,
        "volume_not_decreasing": 0.15,
        "not_choppy": 0.15,
    }

    score = 0.0
    for key, weight in weights.items():
        if checks[key]:
            score += weight

    # Extended rules are additive, not a replacement.
    # Core momentum checklist must pass first.
    core_ok = (
        checks["slow_grind_approach"]
        and checks["left_side_staircase"]
        and checks["volume_not_decreasing"]
        and checks["not_choppy"]
        and checks["balanced_momo_2h"]
        and checks["parallel_to_smma30_2h"]
        and checks["smma_spread_increasing_2h"]
        and score >= min_quality_score
    )

    # Hard requirement: left-side staircase must be present for 2h context.
    if enforce_extended_rules:
        extended_ok = (
            checks["day_change_ok"]
            and checks["vwap_side_ok"]
            and checks["first_2h_prev_day_vwap_ok"]
            and checks["entry_not_crossed_6h"]
        )
    else:
        extended_ok = True

    passed = core_ok and extended_ok

    # Determine quality tier based on counter-trend retracements
    has_retracements, num_retracements = _detect_counter_retracements(
        df=df,
        direction=side,
        lookback_bars=120,  # 2-hour window
        window_size=20,      # ~20-minute sliding windows
        retracement_threshold_pct=2.0  # 2% threshold
    )
    
    if passed:
        quality_tier = "low" if has_retracements else "high"
    else:
        quality_tier = "failed"

    metrics = {
        "net_move_10": float(net_move),
        "opposite_candles_10": float(opposite_candles),
        "smma30_slope": float(smma30_slope),
        "smma120_slope": float(smma120_slope),
        "staircase_bars_120": float(staircase_bars),
        "trend_stack_bars_120": float(trend_stack_bars),
        "vol_slope": float(vol_slope),
        "smma_side_changes_60": float(side_changes),
        "day_change_pct": float(day_change_pct),
        "current_day_vwap": float(current_vwap),
        "prev_day_vwap": float(prev_day_vwap),
        "first_2h_session": float(1 if first_2h else 0),
        "entry_cross_count_6h": float(crossed_count_6h),
        "retracements_found": float(num_retracements),
        "dir_move_2h_pct": float(balanced_metrics["dir_move_2h_pct"]),
        "dir_bar_ratio_2h": float(balanced_metrics["dir_bar_ratio_2h"]),
        "max_dir_impulse_8m_pct": float(balanced_metrics["max_dir_impulse_8m_pct"]),
        "efficiency_2h": float(balanced_metrics["efficiency_2h"]),
        "grind_windows_count": float(balanced_metrics["grind_windows_count"]),
        "avg_grind_impulse_pct": float(balanced_metrics["avg_grind_impulse_pct"]),
        "grind_impulse_std": float(balanced_metrics["grind_impulse_std"]),
        "grind_bar_participation": float(balanced_metrics["grind_bar_participation"]),
        "smma30_crosses_2h": float(parallel_metrics["smma30_crosses_2h"]),
        "smma30_trend_side_ratio_2h": float(parallel_metrics["smma30_trend_side_ratio_2h"]),
        "smma_spread_slope_2h": float(spread_metrics["smma_spread_slope_2h"]),
        "smma_spread_up_ratio_2h": float(spread_metrics["smma_spread_up_ratio_2h"]),
    }

    return MomentumQualityResult(
        direction=side,
        score=float(score),
        passed=passed,
        checks=checks,
        metrics=metrics,
        quality_tier=quality_tier,
    )
