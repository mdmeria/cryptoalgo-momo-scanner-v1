#!/usr/bin/env python3
"""
Depth-of-Book Strategy

Pure order book wall strategy — no candle-based signals.

Entry logic:
  - Fetch depth, find significant walls (resistance above, support below)
  - For long: SL behind support wall, TP just under resistance wall
  - For short: SL behind resistance wall, TP just above support wall
  - Direction chosen by depth imbalance (more buy depth → long, more sell → short)

Filters:
  - SL must be >= 1% from entry
  - TP:SL ratio must be > 1.0 (RR > 1)
  - Wall backing the SL must be >= min_wall_strength_x (default 3x avg)
  - Minimum wall USD value for SL backing
  - Excluded symbols (majors, commodities)

ZCT Alignment:
  - After depth setup passes, run MR and Momo gates on same data
  - Tag trade with closest matching ZCT strategy + DPS score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from depth_tp_sl_analyzer import (
    cluster_walls,
    compute_depth_tp_sl,
)
from live_data_collector import analyze_depth

# Symbols excluded from depth strategy (majors, commodities — walls behave differently)
DEPTH_EXCLUDED_SYMBOLS = {
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
    "XAUUSDT", "XAGUSDT", "PAXGUSDT",
}


@dataclass
class DepthStrategySettings:
    """Settings for the depth-of-book strategy."""
    min_sl_pct: float = 1.0           # Minimum SL distance %
    max_sl_pct: float = 3.0           # Maximum SL distance %
    min_rr: float = 1.2               # Minimum TP:SL ratio (covers fees + slippage)
    min_wall_strength: float = 3.0    # Min wall strength (x avg) for SL backing
    min_wall_usd: float = 5000.0      # Min wall USD value for SL backing
    min_imbalance: float = 0.05       # Min depth imbalance to pick direction
    tp_buffer_pct: float = 0.1        # Place TP this % before the wall
    sl_buffer_pct: float = 0.1        # Place SL this % beyond the wall
    cooldown_bars: int = 30           # Min bars between trades on same symbol


def check_depth_setup(depth_data: dict, current_price: float,
                      cfg: DepthStrategySettings = None) -> dict:
    """
    Check if depth structure supports a trade.

    Returns dict with 'passed', 'reason' (if failed), and trade details if passed.
    """
    if cfg is None:
        cfg = DepthStrategySettings()

    if not depth_data:
        return {"passed": False, "reason": "no_depth_data"}

    asks = depth_data.get("asks", [])
    bids = depth_data.get("bids", [])
    if not asks or not bids:
        return {"passed": False, "reason": "empty_order_book"}

    analysis = analyze_depth(depth_data, current_price)

    # Get walls
    res_walls = analysis.get("tp_sl_walls_resistance", [])
    sup_walls = analysis.get("tp_sl_walls_support", [])

    if not res_walls and not sup_walls:
        return {"passed": False, "reason": "no_walls_found"}

    # Cluster walls into zones
    res_clusters = cluster_walls(res_walls)
    sup_clusters = cluster_walls(sup_walls)

    # Determine direction from depth imbalance
    imbalance_1pct = analysis.get("imbalance_1pct", 0)
    imbalance_2pct = analysis.get("imbalance_2pct", 0)
    avg_imbalance = (imbalance_1pct + imbalance_2pct) / 2

    if abs(avg_imbalance) < cfg.min_imbalance:
        return {"passed": False, "reason": f"imbalance_too_low_{avg_imbalance:+.3f}"}

    # Positive imbalance = more buy pressure = long
    # Negative imbalance = more sell pressure = short
    side = "long" if avg_imbalance > 0 else "short"

    # --- Build TP/SL based on walls ---
    if side == "long":
        # SL: behind strongest support wall (bid wall)
        sl_wall = _find_best_sl_wall(sup_clusters, current_price, side, cfg)
        if sl_wall is None:
            return {"passed": False, "reason": "no_support_wall_for_sl"}

        # TP: just before strongest resistance wall (ask wall)
        tp_wall = _find_best_tp_wall(res_clusters, current_price, side, cfg, sl_wall["dist_pct"])
        if tp_wall is None:
            return {"passed": False, "reason": "no_resistance_wall_for_tp"}

        sl_price = sl_wall["price_lo"] * (1 - cfg.sl_buffer_pct / 100)
        tp_price = tp_wall["price_lo"] * (1 - cfg.tp_buffer_pct / 100)

        sl_pct = (current_price - sl_price) / current_price * 100
        tp_pct = (tp_price - current_price) / current_price * 100

    else:  # short
        # SL: behind strongest resistance wall (ask wall)
        sl_wall = _find_best_sl_wall(res_clusters, current_price, side, cfg)
        if sl_wall is None:
            return {"passed": False, "reason": "no_resistance_wall_for_sl"}

        # TP: just before strongest support wall (bid wall)
        tp_wall = _find_best_tp_wall(sup_clusters, current_price, side, cfg, sl_wall["dist_pct"])
        if tp_wall is None:
            return {"passed": False, "reason": "no_support_wall_for_tp"}

        sl_price = sl_wall["price_hi"] * (1 + cfg.sl_buffer_pct / 100)
        tp_price = tp_wall["price_hi"] * (1 + cfg.tp_buffer_pct / 100)

        sl_pct = (sl_price - current_price) / current_price * 100
        tp_pct = (current_price - tp_price) / current_price * 100

    # --- Validate filters ---
    if sl_pct < cfg.min_sl_pct:
        return {"passed": False, "reason": f"sl_too_close_{sl_pct:.2f}pct"}

    if sl_pct > cfg.max_sl_pct:
        return {"passed": False, "reason": f"sl_too_far_{sl_pct:.2f}pct"}

    rr = tp_pct / sl_pct if sl_pct > 0 else 0
    if rr < cfg.min_rr:
        return {"passed": False, "reason": f"rr_too_low_{rr:.2f}"}

    return {
        "passed": True,
        "side": side,
        "entry": current_price,
        "sl": round(sl_price, 8),
        "tp": round(tp_price, 8),
        "sl_pct": round(sl_pct, 3),
        "tp_pct": round(tp_pct, 3),
        "rr": round(rr, 2),
        "sl_wall_usd": round(sl_wall["usd_value"], 2),
        "sl_wall_strength": sl_wall["strength"],
        "tp_wall_usd": round(tp_wall["usd_value"], 2),
        "tp_wall_strength": tp_wall["strength"],
        "imbalance_1pct": round(imbalance_1pct, 4),
        "imbalance_2pct": round(imbalance_2pct, 4),
        "buy_depth_1pct": analysis.get("buy_depth_1pct", 0),
        "sell_depth_1pct": analysis.get("sell_depth_1pct", 0),
    }


# ---------------------------------------------------------------------------
# ZCT Alignment — check if depth trade aligns with MR or Momo
# ---------------------------------------------------------------------------

def evaluate_zct_alignment(df: pd.DataFrame, side: str,
                           mr_cfg=None, momo_cfg=None) -> dict:
    """
    Run MR and Momo gates on the same bar to see if depth trade
    aligns with a ZCT strategy.

    Returns dict with:
      alignment: "MR" | "Momo" | "unclear"
      mr_reason: gate failure reason (or "passed")
      momo_reason: gate failure reason (or "passed")
      dps_total: DPS score from aligned strategy (0 if unclear)
      dps_confidence: confidence label
    """
    from scan_mean_reversion import check_mr_gates_at_bar, MRSettings
    from backtest_momo_vwap_grind15_full import (
        check_momo_gates_at_bar, prepare_features, GateSettings,
    )

    result = {
        "alignment": "unclear",
        "mr_reason": "not_checked",
        "momo_reason": "not_checked",
        "dps_total": 0,
        "dps_confidence": "unclear",
    }

    # --- Check MR ---
    if mr_cfg is None:
        mr_cfg = MRSettings()

    min_bars_mr = max(mr_cfg.range_max_bars, mr_cfg.noise_lookback_bars, 720) + 10
    if len(df) >= min_bars_mr:
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        i = len(df_sorted) - 1
        mr_result = check_mr_gates_at_bar(df_sorted, i, mr_cfg)

        if mr_result["passed"]:
            # Full MR pass — check if direction matches
            if mr_result["side"] == side:
                result["alignment"] = "MR"
                result["mr_reason"] = "passed"
                result["dps_total"] = mr_result["dps_total"]
                result["dps_confidence"] = mr_result["dps_confidence"]
                return result
            else:
                result["mr_reason"] = f"passed_but_{mr_result['side']}_not_{side}"
        else:
            result["mr_reason"] = mr_result.get("reason", "unknown")

            # Check if "near-MR" — got past range detection
            near_mr_reasons = {"smma_trend", "too_few_touches", "break_pct",
                              "last_touch_too_old", "opposite_bound_broken",
                              "vwap_blocked", "post_touch_trend", "noise_high"}
            if any(result["mr_reason"].startswith(r) for r in near_mr_reasons):
                result["alignment"] = "near-MR"
    else:
        result["mr_reason"] = "insufficient_bars"

    # --- Check Momo ---
    if momo_cfg is None:
        momo_cfg = GateSettings.from_json("momo_gate_settings.json")

    if len(df) >= 500:
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        df_idx = df_sorted.set_index("timestamp").copy()
        df_idx.index = pd.to_datetime(df_idx.index, utc=True)
        try:
            df_prepped = prepare_features(df_idx)
            momo_result = check_momo_gates_at_bar(df_prepped, side, momo_cfg)

            if momo_result["passed"]:
                result["alignment"] = "Momo"
                result["momo_reason"] = "passed"
                result["dps_total"] = momo_result["dps_total"]
                result["dps_confidence"] = momo_result["dps_confidence"]
                return result
            else:
                result["momo_reason"] = momo_result.get("reason", "unknown")

                # Check if "near-Momo" — got past early gates
                near_momo_reasons = {"staircase", "last15m", "ema7", "rr=",
                                    "min_tp_sl", "2h_"}
                if any(result["momo_reason"].startswith(r) for r in near_momo_reasons):
                    if result["alignment"] == "unclear":
                        result["alignment"] = "near-Momo"
        except Exception:
            result["momo_reason"] = "error"
    else:
        result["momo_reason"] = "insufficient_bars"

    return result


def _find_best_sl_wall(clusters: list[dict], current_price: float,
                       side: str, cfg: DepthStrategySettings) -> Optional[dict]:
    """Find the best wall to back our SL against."""
    for wall in clusters:
        # Must meet minimum strength and USD
        if wall["strength"] < cfg.min_wall_strength:
            continue
        if wall["usd_value"] < cfg.min_wall_usd:
            continue
        # Must be within SL range
        if wall["dist_pct"] < cfg.min_sl_pct * 0.5:  # too close
            continue
        if wall["dist_pct"] > cfg.max_sl_pct:
            continue
        return wall
    return None


def _find_best_tp_wall(clusters: list[dict], current_price: float,
                       side: str, cfg: DepthStrategySettings,
                       sl_dist_pct: float) -> Optional[dict]:
    """Find the best wall to place our TP against, ensuring RR > min_rr."""
    for wall in clusters:
        # TP must be far enough to beat RR requirement
        if wall["dist_pct"] < sl_dist_pct * cfg.min_rr:
            continue
        # Any wall with meaningful size works for TP
        if wall["usd_value"] < cfg.min_wall_usd * 0.5:
            continue
        return wall
    return None
