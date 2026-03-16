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


@dataclass
class MomentumCheckConfig:
    """Feature flags for momentum checks to allow isolated tuning."""

    require_grind_subchecks_in_balanced_2h: bool = True
    enforce_geometry_2h_gate: bool = False


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simple ATR series (rolling mean of true range)."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


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


def _fetch_klines(symbol: str, interval: str, limit: int, end_time_ms: int | None = None) -> pd.DataFrame:
    """Fetch klines and return OHLCV DataFrame."""
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)

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


def _nama(src: pd.Series, seed_len: int = 60) -> pd.Series:
    """Port of Pine Script nama(): seed with SMA(seed_len), then slow EWM.

    alpha = 1.618 / 100 = 0.01618 (fixed, independent of seed_len).
    Equivalent to Pine Script:
        nama(src, len) =>
            alpha = 1.618 / 100
            ma = ta.sma(src, len)
            ma := na(ma[1]) ? ma : alpha * src + (1 - alpha) * ma[1]
    """
    alpha = 1.618 / 100.0
    out = src.copy().astype(float)
    sma_seed = src.rolling(seed_len).mean()
    # Find first valid SMA position
    first_valid = sma_seed.first_valid_index()
    if first_valid is None:
        return out * float("nan")
    iloc_start = src.index.get_loc(first_valid)
    out.iloc[:iloc_start] = float("nan")
    out.iloc[iloc_start] = sma_seed.iloc[iloc_start]
    for i in range(iloc_start + 1, len(src)):
        out.iloc[i] = alpha * src.iloc[i] + (1.0 - alpha) * out.iloc[i - 1]
    return out


def _vol_usd_rising(df: pd.DataFrame, seed_len: int = 60, lookback: int = 10) -> bool:
    """Return True when nama(volume*close, seed_len) is rising vs lookback bars ago.

    Implements the Pine Script VolUsd gate:
        vol = volume * close
        ma  = nama(vol, len)   // very slow alpha=1.618/100
        gate passes when ma > ma[lookback bars ago]
    """
    if len(df) < seed_len + lookback + 5:
        return True  # not enough data — don't reject
    vol_usd = df["volume"] * df["close"]
    ma = _nama(vol_usd, seed_len=seed_len)
    if ma.isna().all():
        return True
    ma_now = float(ma.iloc[-1])
    ma_prev = float(ma.iloc[-(lookback + 1)])
    if np.isnan(ma_now) or np.isnan(ma_prev):
        return True
    return ma_now > ma_prev


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


def _regression_r2(values: pd.Series) -> float:
    """Return linear-regression R^2 for the series."""
    clean = values.dropna()
    if len(clean) < 5:
        return 0.0
    x = np.arange(len(clean), dtype=float)
    y = clean.to_numpy(dtype=float)
    if np.allclose(y, y[0]):
        return 0.0
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = intercept + slope * x
    sse = float(np.sum((y - y_hat) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    if sst <= 1e-12:
        return 0.0
    r2 = 1.0 - (sse / sst)
    return float(max(0.0, min(1.0, r2)))


def _momentum_geometry_2h(df: pd.DataFrame, direction: str) -> tuple[bool, Dict[str, float]]:
    """
    Quantify 2h "clean momentum" geometry using slope + linearity + low noise.

    This converts visual "~45 degree, low-noise" intuition into scale-invariant metrics.
    """
    if df is None or len(df) < 120:
        return False, {
            "geom_norm_slope_2h": 0.0,
            "geom_r2_2h": 0.0,
            "geom_er_2h": 0.0,
            "geom_dir_consistency_2h": 0.0,
            "geom_pullback_atr_2h": 999.0,
            "geom_score_2h": 0.0,
        }

    is_long = direction.lower() == "long"
    left = df.iloc[-120:].copy()
    close = left["close"].astype(float)

    atr_series = _atr(left, period=14)
    atr_now = float(atr_series.iloc[-1]) if len(atr_series) > 0 else float("nan")
    price_now = float(close.iloc[-1])

    x = np.arange(len(close), dtype=float)
    y = close.to_numpy(dtype=float)
    if np.allclose(y, y[0]):
        slope = 0.0
    else:
        slope, _ = np.polyfit(x, y, 1)

    # Convert slope to ATR units per bar to make it robust across symbols.
    norm_slope = 0.0
    if np.isfinite(atr_now) and atr_now > 1e-9:
        slope_signed = float(slope if is_long else -slope)
        norm_slope = float(slope_signed / atr_now)

    r2 = _regression_r2(close)

    net_move_abs = abs(float(close.iloc[-1] - close.iloc[0]))
    path = float(close.diff().abs().fillna(0.0).sum())
    er = float(net_move_abs / max(path, 1e-9))

    diff = close.diff().fillna(0.0)
    dir_bars = int((diff > 0).sum()) if is_long else int((diff < 0).sum())
    dir_consistency = float(dir_bars) / max(len(diff), 1)

    # Worst counter-trend pullback from rolling extreme, normalized by ATR.
    if is_long:
        rolling_peak = close.cummax()
        pullback = (rolling_peak - close).max()
    else:
        rolling_trough = close.cummin()
        pullback = (close - rolling_trough).max()

    pullback_atr = 999.0
    if np.isfinite(atr_now) and atr_now > 1e-9:
        pullback_atr = float(pullback / atr_now)

    # Weights intentionally balanced between trend steepness, linearity, and noise control.
    geom_score = (
        0.30 * max(0.0, norm_slope)
        + 0.25 * r2
        + 0.25 * er
        + 0.20 * dir_consistency
        - 0.20 * min(2.0, max(0.0, pullback_atr))
    )

    # Permissive defaults for 2h 1m-bar windows; tune with your backtests.
    geometry_ok = (
        norm_slope >= 0.015
        and r2 >= 0.55
        and er >= 0.35
        and dir_consistency >= 0.52
        and pullback_atr <= 1.20
    )

    return geometry_ok, {
        "geom_norm_slope_2h": float(norm_slope),
        "geom_r2_2h": float(r2),
        "geom_er_2h": float(er),
        "geom_dir_consistency_2h": float(dir_consistency),
        "geom_pullback_atr_2h": float(pullback_atr),
        "geom_score_2h": float(geom_score),
    }


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


def _balanced_momo_profile_2h(
    df: pd.DataFrame,
    direction: str,
    require_grind_subchecks: bool = True,
) -> tuple[bool, Dict[str, float]]:
    """
    Validate 2h momentum profile is directional and stable.

    This check is intentionally focused on 2-hour directional structure only.
    Grind-quality subchecks are handled in the last 10 minutes near entry.
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

    # Keep grind metrics for diagnostics only (not used for 2h pass/fail gating).
    grind_metrics = _analyze_8min_grind_quality(df, direction)

    # Check regime age and stability
    regime_age_bars, regime_age_ok = _detect_regime_age_2h(df, direction)
    recent_break_detected, regime_stability_ok = _check_recent_high_low_break_2h(df, direction)
    
    # 2h profile must show directional action, but stay permissive enough
    # to keep valid momentum regimes from being over-filtered.
    base_balanced_ok = (
        0.004 <= dir_move <= 0.350
        and 0.45 <= dir_bar_ratio <= 0.98
        and max_dir_impulse_8m <= 0.025
        and 0.12 <= efficiency <= 0.90
        and regime_age_bars >= 60
    )

    # Intentionally ignore periodic 2h grind subchecks.
    balanced_ok = base_balanced_ok

    return balanced_ok, {
        "dir_move_2h_pct": float(dir_move * 100.0),
        "dir_bar_ratio_2h": float(dir_bar_ratio),
        "max_dir_impulse_8m_pct": float(max_dir_impulse_8m * 100.0),
        "efficiency_2h": float(efficiency),
        "grind_windows_count": grind_metrics["grind_windows_count"],
        "avg_grind_impulse_pct": grind_metrics["avg_grind_impulse_pct"],
        "grind_impulse_std": grind_metrics["grind_impulse_std"],
        "grind_bar_participation": grind_metrics["grind_bar_participation"],
        "regime_age_bars": float(regime_age_bars),
        "regime_age_ok": regime_age_ok,
        "recent_high_low_break": recent_break_detected,
        "regime_stability_ok": regime_stability_ok,
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

    parallel_ok = cross_count <= 2 and trend_side_ratio >= 0.90

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


def _momentum_noise_class_2h(smma30_slope: float, smma30_crosses_2h: float) -> tuple[str, str]:
    """Classify momentum noise using 30 SMMA direction and crossover count."""
    # Sideways MA direction is always high-noise for momentum.
    if abs(float(smma30_slope)) <= 0.0005:
        return "sideways", "high"

    direction = "up" if smma30_slope > 0 else "down"
    crosses = int(round(float(smma30_crosses_2h)))

    if crosses <= 3:
        noise = "low"
    elif crosses <= 6:
        noise = "medium"
    else:
        noise = "high"

    return direction, noise


def _calculate_entry_bar_concentration_30m(df: pd.DataFrame, direction: str) -> tuple[float, bool]:
    """
    Calculate what % of the 30m move came from the last 10 bars.
    
    High concentration (>60%) indicates a spike pattern that likely leads to pullback/entry failure.
    Returns (concentration_pct, passes_filter)
    """
    if df is None or len(df) < 30:
        return 0.0, False
    
    is_long = direction.lower() == "long"
    tail = df.iloc[-30:]
    close = tail["close"]
    
    # Total 30m move
    start_30m = float(close.iloc[0])
    end_30m = float(close.iloc[-1])
    total_move_30m = (end_30m - start_30m) / max(abs(start_30m), 1e-9)
    
    # Move in last 10 bars
    start_10b = float(close.iloc[-10])
    end_10b = float(close.iloc[-1])
    move_last_10b = (end_10b - start_10b) / max(abs(start_30m), 1e-9)
    
    # Avoid division by zero/near-zero moves
    if abs(total_move_30m) < 1e-5:
        concentration = 0.0
    else:
        concentration = abs(move_last_10b / total_move_30m)
    
    # Spike filter: if >60% of move is in final 10 bars, likely a spike
    passes = concentration <= 0.60
    
    return concentration, passes


def _detect_regime_age_2h(df: pd.DataFrame, direction: str) -> tuple[int, bool]:
    """
    Detect how old the current 2-hour directional regime is (in minutes/bars).
    
    Finds the bar where the 2h move started and returns age.
    Relaxed check: flags if <10 bars old (very fresh, high risk), passes otherwise.
    Returns (age_in_bars, passes_filter)
    """
    if df is None or len(df) < 120:
        return 0, True  # Default to pass if insufficient data
    
    is_long = direction.lower() == "long"
    left = df.iloc[-120:]
    close = left["close"]
    
    # Find the bar where the current directional move initiated
    # Walk backwards from current and find where trend reversed
    regime_start_idx = 0
    for i in range(len(close) - 1, 0, -1):
        if is_long:
            # For long: find where we went below previous bar
            if close.iloc[i] < close.iloc[i-1]:
                regime_start_idx = i + 1
                break
        else:
            # For short: find where we went above previous bar
            if close.iloc[i] > close.iloc[i-1]:
                regime_start_idx = i + 1
                break
    
    age_in_bars = len(close) - 1 - regime_start_idx
    
    # Require at least 60 bars (~1h) of established regime before entry.
    passes = age_in_bars >= 60
    
    return age_in_bars, passes


def _check_recent_high_low_break_2h(df: pd.DataFrame, direction: str) -> tuple[bool, bool]:
    """
    Check if extreme from earlier in 2h window got broken recently.
    
    Only flags regime instability if a SIGNIFICANT extreme (top 10% of moves)
    from the middle window was broken in the final window.
    Returns (recent_break_detected, passes_filter)
    """
    if df is None or len(df) < 120:
        return False, True
    
    is_long = direction.lower() == "long"
    left = df.iloc[-120:].copy()
    
    # First, calculate what a "significant" extreme would be
    all_high = left["high"]
    all_low = left["low"]
    
    if is_long:
        # For long: track lows, significance = bottom 10th percentile
        significant_threshold = all_low.quantile(0.10)
    else:
        # For short: track highs, significance = top 90th percentile  
        significant_threshold = all_high.quantile(0.90)
    
    # Mid-window (bars 40-80 from end, i.e. indices 40-80 in 120-bar window)
    mid_window = left.iloc[40:80]
    
    if is_long:
        # For long setup: check if support below significance was broken
        mid_significant = (mid_window["low"] <= significant_threshold).any()
        if not mid_significant:
            return False, True  # No significant low to watch
        
        # Recent window (bars 100-120, i.e. last 20 bars)
        recent_window = left.iloc[-20:]
        # Pass if we DON'T break that support
        recent_break = (recent_window["low"] < significant_threshold).any()
    else:
        # For short setup: check if resistance above significance was broken
        mid_significant = (mid_window["high"] >= significant_threshold).any()
        if not mid_significant:
            return False, True  # No significant high to watch
        
        # Recent window (bars 100-120, i.e. last 20 bars)
        recent_window = left.iloc[-20:]
        # Pass if we DON'T break that resistance
        recent_break = (recent_window["high"] > significant_threshold).any()
    
    # Pass filter if NO significant break detected, or if no significant extreme to watch
    passes = not recent_break
    
    return recent_break, passes


def _pre_entry_directional_30m(df: pd.DataFrame, direction: str) -> tuple[bool, Dict[str, float]]:
    """
    Require directional structure into entry, with grind quality focused on last 10 minutes.
    """
    if df is None or len(df) < 30:
        return False, {
            "pre_entry_move_30m_pct": 0.0,
            "pre_entry_efficiency_30m": 0.0,
            "pre_entry_dir_bar_ratio_30m": 0.0,
            "pre_entry_spike_concentration_pct": 0.0,
            "pre_entry_bar_concentration_ok": False,
            "pre_entry_move_10m_pct": 0.0,
            "pre_entry_efficiency_10m": 0.0,
            "pre_entry_dir_bar_ratio_10m": 0.0,
            "pre_entry_opp_candles_10m": 0.0,
            "pre_entry_grind_10m_ok": False,
        }

    is_long = direction.lower() == "long"
    tail = df.iloc[-30:]
    close = tail["close"]

    start = float(close.iloc[0])
    end = float(close.iloc[-1])
    raw_move = (end - start) / max(abs(start), 1e-9)
    dir_move = raw_move if is_long else -raw_move

    diff = close.diff().fillna(0.0)
    dir_bars = (diff > 0).sum() if is_long else (diff < 0).sum()
    dir_bar_ratio = float(dir_bars) / max(len(diff), 1)

    path = close.pct_change().abs().fillna(0.0).sum()
    efficiency = float(dir_move / max(float(path), 1e-9))

    # Last 10m grind-quality check (entry-critical).
    tail10 = tail.iloc[-10:]
    close10 = tail10["close"]
    start10 = float(close10.iloc[0])
    end10 = float(close10.iloc[-1])
    raw_move_10m = (end10 - start10) / max(abs(start10), 1e-9)
    dir_move_10m = raw_move_10m if is_long else -raw_move_10m

    diff10 = close10.diff().fillna(0.0)
    dir_bars_10m = (diff10 > 0).sum() if is_long else (diff10 < 0).sum()
    dir_bar_ratio_10m = float(dir_bars_10m) / max(len(diff10), 1)
    opp_candles_10m = int((diff10 < 0).sum()) if is_long else int((diff10 > 0).sum())

    path10 = close10.pct_change().abs().fillna(0.0).sum()
    efficiency10 = float(dir_move_10m / max(float(path10), 1e-9))

    grind_10m_ok = (
        dir_move_10m >= 0.0020
        and dir_bar_ratio_10m >= 0.60
        and 0.30 <= efficiency10 <= 0.95
        and opp_candles_10m <= 3
    )

    # NEW: Spike concentration check
    spike_concentration, concentration_ok = _calculate_entry_bar_concentration_30m(df, direction)
    
    # Entry check now prioritizes last-10-minute grind quality plus anti-spike distribution.
    pre_entry_ok = (
        dir_move >= 0.004
        and dir_bar_ratio >= 0.50  # TIGHTENED from 0.40
        and efficiency >= 0.20
        and concentration_ok  # NEW: must not be spike pattern
        and grind_10m_ok
    )

    return pre_entry_ok, {
        "pre_entry_move_30m_pct": float(dir_move * 100.0),
        "pre_entry_efficiency_30m": float(efficiency),
        "pre_entry_dir_bar_ratio_30m": float(dir_bar_ratio),
        "pre_entry_spike_concentration_pct": float(spike_concentration * 100.0),
        "pre_entry_bar_concentration_ok": concentration_ok,
        "pre_entry_move_10m_pct": float(dir_move_10m * 100.0),
        "pre_entry_efficiency_10m": float(efficiency10),
        "pre_entry_dir_bar_ratio_10m": float(dir_bar_ratio_10m),
        "pre_entry_opp_candles_10m": float(opp_candles_10m),
        "pre_entry_grind_10m_ok": grind_10m_ok,
    }


def evaluate_momentum_setup(
    df: pd.DataFrame,
    direction: str,
    min_quality_score: float = 0.60,
    symbol: str | None = None,
    enforce_extended_rules: bool = False,
    eval_time: pd.Timestamp | None = None,
    check_config: MomentumCheckConfig | None = None,
) -> MomentumQualityResult:
    """
    Evaluate whether the current momentum signal is high quality.

    Rules approximate the documented momentum checklist:
    - clean approach to level over last 1-10 minutes
    - staircase-style context on the left side (roughly 2h)
    - avoid decreasing volume profile
    - reject obvious chop
    """
    cfg = check_config or MomentumCheckConfig()

    if df is None or len(df) < 120:
        return MomentumQualityResult(
            direction=direction,
            score=0.0,
            passed=False,
            checks={
                "enough_data": False,
                "slow_grind_approach": False,
                "pre_entry_directional_30m": False,
                "left_side_staircase": False,
                "volume_not_decreasing": False,
                "not_choppy": False,
                "balanced_momo_2h": False,
                "parallel_to_smma30_2h": False,
                "smma_spread_increasing_2h": False,
                "momentum_geometry_2h": False,
            },
            metrics={"bars": float(len(df) if df is not None else 0)},
        )

    side = direction.lower()
    is_long = side == "long"

    work = df.copy()
    work["smma30"] = _smma(work["close"], 30)
    work["smma120"] = _smma(work["close"], 120)

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

    # Rule 1b: Entry window should not be sideways over the last 30m.
    pre_entry_directional_30m, pre_entry_metrics = _pre_entry_directional_30m(work, side)

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

    # Rule 3: VolUsd MA (nama, alpha=1.618/100) must be rising vs 10 bars ago.
    # Implements the Pine Script VolUsd indicator: vol=volume*close, ma=nama(vol,60),
    # gate passes when ma > ma[10 bars ago].
    volume_ok = _vol_usd_rising(work, seed_len=60, lookback=10)

    # Rule 5: 2h profile should be between sideways and spike behavior.
    balanced_momo_2h, balanced_metrics = _balanced_momo_profile_2h(
        work,
        side,
        require_grind_subchecks=cfg.require_grind_subchecks_in_balanced_2h,
    )

    # Rule 6: Price should stay parallel to 30SMMA (limited criss-cross).
    parallel_to_smma30_2h, parallel_metrics = _price_parallel_to_smma30_2h(work, side)

    # Rule 4: Noise model from 30 SMMA PDF (direction + cross-count).
    smma30_direction_2h, noise_class_momentum = _momentum_noise_class_2h(
        smma30_slope=smma30_slope,
        smma30_crosses_2h=parallel_metrics["smma30_crosses_2h"],
    )
    not_choppy = noise_class_momentum == "low"

    # Rule 7: Distance between 30SMMA and 120SMMA should gradually increase.
    smma_spread_increasing_2h, spread_metrics = _smma_spread_slowly_increasing_2h(work, side)

    # Rule 8: Geometric trend quality over full 2h (slope + linearity + low noise).
    momentum_geometry_2h, geometry_metrics = _momentum_geometry_2h(work, side)

    checks = {
        "enough_data": True,
        "slow_grind_approach": approach_ok,
        "pre_entry_directional_30m": pre_entry_directional_30m,
        "left_side_staircase": staircase_ok,
        "volume_not_decreasing": volume_ok,
        "not_choppy": not_choppy,
        "balanced_momo_2h": balanced_momo_2h,
        "parallel_to_smma30_2h": parallel_to_smma30_2h,
        "smma_spread_increasing_2h": smma_spread_increasing_2h,
        "momentum_geometry_2h": momentum_geometry_2h,
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
            if eval_time is None:
                ticker = _fetch_ticker_24h(symbol)
                if ticker:
                    day_change_pct = float(ticker.get("priceChangePercent", 0.0))
                else:
                    day_change_pct = float("nan")
                eval_ts = pd.Timestamp.utcnow()
                if eval_ts.tzinfo is None:
                    eval_ts = eval_ts.tz_localize("UTC")
                end_time_ms = None
            else:
                eval_ts = pd.Timestamp(eval_time)
                if eval_ts.tzinfo is None:
                    eval_ts = eval_ts.tz_localize("UTC")
                else:
                    eval_ts = eval_ts.tz_convert("UTC")
                end_time_ms = int(eval_ts.timestamp() * 1000)

                # Historical 24h change approximation from 5m closes ending at eval time.
                day_change_df = _fetch_klines(symbol, interval="5m", limit=320, end_time_ms=end_time_ms)
                if not day_change_df.empty:
                    cutoff_ts = eval_ts - pd.Timedelta(hours=24)
                    last_24h = day_change_df[(day_change_df.index >= cutoff_ts) & (day_change_df.index <= eval_ts)]
                    if len(last_24h) >= 2:
                        start_px = float(last_24h["close"].iloc[0])
                        end_px = float(last_24h["close"].iloc[-1])
                        day_change_pct = ((end_px - start_px) / max(abs(start_px), 1e-9)) * 100.0
                    else:
                        day_change_pct = float("nan")
                else:
                    day_change_pct = float("nan")

            if is_long:
                checks["day_change_ok"] = bool(day_change_pct >= 5.0)
            else:
                checks["day_change_ok"] = bool(day_change_pct <= -5.0)

            # Rule B/C: VWAP side + first 2h previous day VWAP requirement
            vwap_df = _fetch_klines(symbol, interval="5m", limit=600, end_time_ms=end_time_ms)
            if not vwap_df.empty:
                now_ts = eval_ts

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
        and checks["pre_entry_directional_30m"]
        and checks["left_side_staircase"]
        and checks["volume_not_decreasing"]
        and checks["not_choppy"]
        and checks["balanced_momo_2h"]
        and checks["parallel_to_smma30_2h"]
        and checks["smma_spread_increasing_2h"]
        and score >= min_quality_score
        and ((not cfg.enforce_geometry_2h_gate) or checks["momentum_geometry_2h"])
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
        "pre_entry_move_30m_pct": float(pre_entry_metrics["pre_entry_move_30m_pct"]),
        "pre_entry_efficiency_30m": float(pre_entry_metrics["pre_entry_efficiency_30m"]),
        "pre_entry_dir_bar_ratio_30m": float(pre_entry_metrics["pre_entry_dir_bar_ratio_30m"]),
        "pre_entry_spike_concentration_pct": float(pre_entry_metrics["pre_entry_spike_concentration_pct"]),
        "pre_entry_bar_concentration_ok": float(1 if pre_entry_metrics["pre_entry_bar_concentration_ok"] else 0),
        "pre_entry_move_10m_pct": float(pre_entry_metrics["pre_entry_move_10m_pct"]),
        "pre_entry_efficiency_10m": float(pre_entry_metrics["pre_entry_efficiency_10m"]),
        "pre_entry_dir_bar_ratio_10m": float(pre_entry_metrics["pre_entry_dir_bar_ratio_10m"]),
        "pre_entry_opp_candles_10m": float(pre_entry_metrics["pre_entry_opp_candles_10m"]),
        "pre_entry_grind_10m_ok": float(1 if pre_entry_metrics["pre_entry_grind_10m_ok"] else 0),
        "smma30_slope": float(smma30_slope),
        "smma120_slope": float(smma120_slope),
        "staircase_bars_120": float(staircase_bars),
        "trend_stack_bars_120": float(trend_stack_bars),
        "vol_usd_rising": float(1 if volume_ok else 0),
        "smma_side_changes_60": float(parallel_metrics["smma30_crosses_2h"]),
        "smma30_direction_2h": 1.0 if smma30_direction_2h == "up" else (-1.0 if smma30_direction_2h == "down" else 0.0),
        "noise_class_momentum": 0.0 if noise_class_momentum == "low" else (1.0 if noise_class_momentum == "medium" else 2.0),
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
        "regime_age_bars": float(balanced_metrics["regime_age_bars"]),
        "regime_age_ok": float(1 if balanced_metrics["regime_age_ok"] else 0),
        "recent_high_low_break": float(1 if balanced_metrics["recent_high_low_break"] else 0),
        "regime_stability_ok": float(1 if balanced_metrics["regime_stability_ok"] else 0),
        "grind_windows_count": float(balanced_metrics["grind_windows_count"]),
        "avg_grind_impulse_pct": float(balanced_metrics["avg_grind_impulse_pct"]),
        "grind_impulse_std": float(balanced_metrics["grind_impulse_std"]),
        "grind_bar_participation": float(balanced_metrics["grind_bar_participation"]),
        "smma30_crosses_2h": float(parallel_metrics["smma30_crosses_2h"]),
        "smma30_trend_side_ratio_2h": float(parallel_metrics["smma30_trend_side_ratio_2h"]),
        "smma_spread_slope_2h": float(spread_metrics["smma_spread_slope_2h"]),
        "smma_spread_up_ratio_2h": float(spread_metrics["smma_spread_up_ratio_2h"]),
        "geom_norm_slope_2h": float(geometry_metrics["geom_norm_slope_2h"]),
        "geom_r2_2h": float(geometry_metrics["geom_r2_2h"]),
        "geom_er_2h": float(geometry_metrics["geom_er_2h"]),
        "geom_dir_consistency_2h": float(geometry_metrics["geom_dir_consistency_2h"]),
        "geom_pullback_atr_2h": float(geometry_metrics["geom_pullback_atr_2h"]),
        "geom_score_2h": float(geometry_metrics["geom_score_2h"]),
        "grind_subchecks_enabled": float(1 if cfg.require_grind_subchecks_in_balanced_2h else 0),
        "geometry_2h_gate_enabled": float(1 if cfg.enforce_geometry_2h_gate else 0),
    }

    return MomentumQualityResult(
        direction=side,
        score=float(score),
        passed=passed,
        checks=checks,
        metrics=metrics,
        quality_tier=quality_tier,
    )
