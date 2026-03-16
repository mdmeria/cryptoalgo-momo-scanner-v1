#!/usr/bin/env python3
"""
Depth-Based TP/SL Analyzer

Analyzes order book depth data to suggest optimal TP/SL levels for both
Momentum and Mean Reversion trades.

Modes:
  --live     Fetch fresh depth + candles for a symbol and analyze now
  --report   Analyze accumulated depth data and produce a summary report
  --scan     Scan all active coins and produce TP/SL suggestions for each

Usage:
  python depth_tp_sl_analyzer.py --live BTCUSDT
  python depth_tp_sl_analyzer.py --report ETHUSDT
  python depth_tp_sl_analyzer.py --scan --min-vol 200000
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import shared functions from live_data_collector
from live_data_collector import (
    fetch_orion_active_coins,
    fetch_bitunix_depth,
    fetch_bitunix_klines,
    analyze_depth,
    LIVE_DIR,
    DEPTH_DIR,
    DEPTH_SUMMARY_DIR,
    CANDLES_DIR,
)


# ---------------------------------------------------------------------------
# Depth TP/SL Engine
# ---------------------------------------------------------------------------

def cluster_walls(walls: list[dict], cluster_pct: float = 0.15) -> list[dict]:
    """
    Cluster nearby walls into zones.
    Walls within cluster_pct% of each other get merged.
    Returns list of clustered walls sorted by total USD value.
    """
    if not walls:
        return []

    # Sort by price
    sorted_walls = sorted(walls, key=lambda w: w["price"])
    clusters = []
    current_cluster = [sorted_walls[0]]

    for w in sorted_walls[1:]:
        prev_price = current_cluster[-1]["price"]
        gap_pct = abs(w["price"] - prev_price) / prev_price * 100
        if gap_pct <= cluster_pct:
            current_cluster.append(w)
        else:
            clusters.append(current_cluster)
            current_cluster = [w]
    clusters.append(current_cluster)

    # Merge each cluster into a single zone
    merged = []
    for cluster in clusters:
        total_qty = sum(w["qty"] for w in cluster)
        total_usd = sum(w.get("usd_value", w["price"] * w["qty"]) for w in cluster)
        # Weighted average price
        wavg_price = sum(w["price"] * w["qty"] for w in cluster) / total_qty
        avg_dist = sum(w["dist_pct"] for w in cluster) / len(cluster)
        max_strength = max(w.get("strength", 1.0) for w in cluster)

        merged.append({
            "price": round(wavg_price, 6),
            "qty": round(total_qty, 4),
            "usd_value": round(total_usd, 2),
            "dist_pct": round(avg_dist, 3),
            "strength": round(max_strength, 1),
            "n_levels": len(cluster),
            "price_lo": cluster[0]["price"],
            "price_hi": cluster[-1]["price"],
        })

    merged.sort(key=lambda x: x["usd_value"], reverse=True)
    return merged


def find_thin_zones(levels: list, current_price: float,
                    side: str, max_dist_pct: float = 3.0,
                    window: int = 10) -> list[dict]:
    """
    Find thin zones in the order book where price could move fast.
    A thin zone is a stretch of price levels with below-average quantity.

    side: 'above' (for long TP / short SL) or 'below' (for short TP / long SL)
    """
    if not levels or len(levels) < window:
        return []

    # Filter to relevant side within max_dist_pct
    filtered = []
    for price, qty in levels:
        dist = (price - current_price) / current_price * 100
        if side == "above" and 0 < dist <= max_dist_pct:
            filtered.append((price, qty, dist))
        elif side == "below" and 0 < -dist <= max_dist_pct:
            filtered.append((price, qty, abs(dist)))

    if len(filtered) < window:
        return []

    # Sort by distance from price
    filtered.sort(key=lambda x: x[2])
    avg_qty = np.mean([f[1] for f in filtered])

    # Sliding window to find thin stretches
    thin_zones = []
    for i in range(len(filtered) - window + 1):
        window_levels = filtered[i:i + window]
        window_avg = np.mean([w[1] for w in window_levels])
        if window_avg < avg_qty * 0.4:  # Less than 40% of average = thin
            thin_zones.append({
                "start_price": window_levels[0][0],
                "end_price": window_levels[-1][0],
                "start_dist_pct": round(window_levels[0][2], 3),
                "end_dist_pct": round(window_levels[-1][2], 3),
                "avg_qty_ratio": round(window_avg / avg_qty, 2),
            })

    # Deduplicate overlapping zones
    if not thin_zones:
        return []

    merged = [thin_zones[0]]
    for z in thin_zones[1:]:
        if z["start_dist_pct"] <= merged[-1]["end_dist_pct"]:
            merged[-1]["end_price"] = z["end_price"]
            merged[-1]["end_dist_pct"] = z["end_dist_pct"]
            merged[-1]["avg_qty_ratio"] = min(merged[-1]["avg_qty_ratio"],
                                               z["avg_qty_ratio"])
        else:
            merged.append(z)

    return merged


def compute_depth_tp_sl(depth_data: dict, current_price: float,
                         side: str, strategy: str,
                         min_tp_pct: float = 1.0,
                         min_rr: float = 1.0) -> dict:
    """
    Compute TP/SL suggestions based on order book depth.

    Args:
        depth_data: Raw depth from Bitunix {asks: [...], bids: [...]}
        current_price: Current mid price
        side: 'long' or 'short'
        strategy: 'momentum' or 'mean_reversion'
        min_tp_pct: Minimum TP distance in % (default 1%)
        min_rr: Minimum reward/risk ratio (default 1.0)

    Returns dict with suggested levels and reasoning.
    """
    analysis = analyze_depth(depth_data, current_price)

    asks = [[float(a[0]), float(a[1])] for a in depth_data.get("asks", [])]
    bids = [[float(b[0]), float(b[1])] for b in depth_data.get("bids", [])]
    asks.sort(key=lambda x: x[0])
    bids.sort(key=lambda x: x[0], reverse=True)

    res_walls = analysis.get("tp_sl_walls_resistance", [])
    sup_walls = analysis.get("tp_sl_walls_support", [])

    # Cluster walls into zones
    res_clusters = cluster_walls(res_walls)
    sup_clusters = cluster_walls(sup_walls)

    # Find thin zones
    thin_above = find_thin_zones(asks, current_price, "above")
    thin_below = find_thin_zones(bids, current_price, "below")

    result = {
        "symbol": None,  # filled by caller
        "price": current_price,
        "side": side,
        "strategy": strategy,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if side == "long":
        # --- LONG TRADE ---
        # SL: Place behind strongest support wall (bid wall)
        # TP: Place before strongest resistance wall (ask wall), or in thin zone above

        # SL selection
        sl_candidates = []
        for wall in sup_clusters:
            if wall["dist_pct"] < 0.2:
                continue
            # SL goes just below the support wall
            sl_price = wall["price_lo"] * 0.999  # tiny buffer below wall
            sl_dist = (current_price - sl_price) / current_price * 100
            if sl_dist > 3.0:
                continue
            sl_candidates.append({
                "price": round(sl_price, 6),
                "dist_pct": round(sl_dist, 3),
                "wall_usd": wall["usd_value"],
                "wall_strength": wall["strength"],
                "wall_levels": wall["n_levels"],
                "reason": f"Behind ${wall['usd_value']:,.0f} bid wall "
                          f"({wall['strength']}x avg, {wall['n_levels']} levels)",
            })

        # If no wall-backed SL, use nearest support
        if not sl_candidates:
            ns = analysis.get("nearest_support")
            if ns:
                sl_price = ns["price"] * 0.999
                sl_dist = (current_price - sl_price) / current_price * 100
                sl_candidates.append({
                    "price": round(sl_price, 6),
                    "dist_pct": round(sl_dist, 3),
                    "wall_usd": ns["qty"] * ns["price"],
                    "wall_strength": 0,
                    "wall_levels": 1,
                    "reason": f"Behind nearest support at {ns['price']:.6g}",
                })

        # TP selection for long
        tp_candidates = []

        if strategy == "momentum":
            # Momentum: TP at thin zones (price runs through easily)
            # or just before the first major resistance wall
            for zone in thin_above:
                if zone["end_dist_pct"] >= min_tp_pct:
                    tp_candidates.append({
                        "price": round(zone["end_price"], 6),
                        "dist_pct": round(zone["end_dist_pct"], 3),
                        "reason": f"Thin zone {zone['start_dist_pct']:.1f}-{zone['end_dist_pct']:.1f}% "
                                  f"(depth {zone['avg_qty_ratio']:.0%} of avg)",
                        "type": "thin_zone",
                    })

            # Also consider space before first big resistance wall
            for wall in res_clusters:
                if wall["dist_pct"] >= min_tp_pct:
                    tp_price = wall["price_lo"] * 0.999  # just before wall
                    tp_dist = (tp_price - current_price) / current_price * 100
                    if tp_dist >= min_tp_pct:
                        tp_candidates.append({
                            "price": round(tp_price, 6),
                            "dist_pct": round(tp_dist, 3),
                            "reason": f"Before ${wall['usd_value']:,.0f} ask wall "
                                      f"({wall['strength']}x avg)",
                            "type": "before_wall",
                        })

        else:  # mean_reversion
            # MR: TP at opposite boundary. Check if path is clear.
            # Find gaps in resistance — places with low sell depth
            # Use the first moderate resistance wall as TP (price bounces to it)
            for wall in res_clusters:
                if wall["dist_pct"] >= min_tp_pct:
                    tp_candidates.append({
                        "price": round(wall["price"], 6),
                        "dist_pct": round(wall["dist_pct"], 3),
                        "reason": f"Resistance zone ${wall['usd_value']:,.0f} "
                                  f"({wall['n_levels']} levels clustered)",
                        "type": "wall_target",
                    })

            # Also thin zones as bonus targets
            for zone in thin_above:
                if zone["start_dist_pct"] >= min_tp_pct:
                    tp_candidates.append({
                        "price": round(zone["start_price"], 6),
                        "dist_pct": round(zone["start_dist_pct"], 3),
                        "reason": f"Clear path starts at {zone['start_dist_pct']:.1f}%",
                        "type": "clear_path",
                    })

    else:
        # --- SHORT TRADE ---
        # SL: Place behind strongest resistance wall (ask wall)
        # TP: Place before strongest support wall (bid wall), or in thin zone below

        sl_candidates = []
        for wall in res_clusters:
            if wall["dist_pct"] < 0.2:
                continue
            sl_price = wall["price_hi"] * 1.001
            sl_dist = (sl_price - current_price) / current_price * 100
            if sl_dist > 3.0:
                continue
            sl_candidates.append({
                "price": round(sl_price, 6),
                "dist_pct": round(sl_dist, 3),
                "wall_usd": wall["usd_value"],
                "wall_strength": wall["strength"],
                "wall_levels": wall["n_levels"],
                "reason": f"Behind ${wall['usd_value']:,.0f} ask wall "
                          f"({wall['strength']}x avg, {wall['n_levels']} levels)",
            })

        if not sl_candidates:
            nr = analysis.get("nearest_resistance")
            if nr:
                sl_price = nr["price"] * 1.001
                sl_dist = (sl_price - current_price) / current_price * 100
                sl_candidates.append({
                    "price": round(sl_price, 6),
                    "dist_pct": round(sl_dist, 3),
                    "wall_usd": nr["qty"] * nr["price"],
                    "wall_strength": 0,
                    "wall_levels": 1,
                    "reason": f"Behind nearest resistance at {nr['price']:.6g}",
                })

        tp_candidates = []

        if strategy == "momentum":
            for zone in thin_below:
                if zone["end_dist_pct"] >= min_tp_pct:
                    tp_candidates.append({
                        "price": round(zone["end_price"], 6),
                        "dist_pct": round(zone["end_dist_pct"], 3),
                        "reason": f"Thin zone {zone['start_dist_pct']:.1f}-{zone['end_dist_pct']:.1f}% "
                                  f"(depth {zone['avg_qty_ratio']:.0%} of avg)",
                        "type": "thin_zone",
                    })

            for wall in sup_clusters:
                if wall["dist_pct"] >= min_tp_pct:
                    tp_price = wall["price_hi"] * 1.001
                    tp_dist = (current_price - tp_price) / current_price * 100
                    if tp_dist >= min_tp_pct:
                        tp_candidates.append({
                            "price": round(tp_price, 6),
                            "dist_pct": round(tp_dist, 3),
                            "reason": f"Before ${wall['usd_value']:,.0f} bid wall "
                                      f"({wall['strength']}x avg)",
                            "type": "before_wall",
                        })
        else:  # mean_reversion
            for wall in sup_clusters:
                if wall["dist_pct"] >= min_tp_pct:
                    tp_candidates.append({
                        "price": round(wall["price"], 6),
                        "dist_pct": round(wall["dist_pct"], 3),
                        "reason": f"Support zone ${wall['usd_value']:,.0f} "
                                  f"({wall['n_levels']} levels clustered)",
                        "type": "wall_target",
                    })

            for zone in thin_below:
                if zone["start_dist_pct"] >= min_tp_pct:
                    tp_candidates.append({
                        "price": round(zone["start_price"], 6),
                        "dist_pct": round(zone["start_dist_pct"], 3),
                        "reason": f"Clear path starts at {zone['start_dist_pct']:.1f}%",
                        "type": "clear_path",
                    })

    # --- Select best TP/SL combo meeting RR requirement ---
    best_combo = None
    for sl in sl_candidates:
        for tp in tp_candidates:
            rr = tp["dist_pct"] / sl["dist_pct"] if sl["dist_pct"] > 0 else 0
            if tp["dist_pct"] >= min_tp_pct and rr >= min_rr:
                score = rr * (1 + sl.get("wall_strength", 0) * 0.1)
                if best_combo is None or score > best_combo["score"]:
                    best_combo = {
                        "tp": tp,
                        "sl": sl,
                        "rr": round(rr, 2),
                        "score": round(score, 2),
                    }

    result["sl_candidates"] = sl_candidates[:5]
    result["tp_candidates"] = tp_candidates[:5]
    result["best_combo"] = best_combo
    result["depth_imbalance_1pct"] = analysis.get("imbalance_1pct", 0)
    result["depth_imbalance_2pct"] = analysis.get("imbalance_2pct", 0)
    result["buy_depth_1pct"] = analysis.get("buy_depth_1pct", 0)
    result["sell_depth_1pct"] = analysis.get("sell_depth_1pct", 0)
    result["resistance_clusters"] = res_clusters[:5]
    result["support_clusters"] = sup_clusters[:5]
    result["thin_zones_above"] = thin_above[:3]
    result["thin_zones_below"] = thin_below[:3]
    result["spread_pct"] = analysis.get("spread_pct", 0)

    return result


# ---------------------------------------------------------------------------
# Price Action Context (from 1m candles)
# ---------------------------------------------------------------------------

def get_price_context(symbol: str, candles_dir: Path = CANDLES_DIR) -> dict:
    """
    Get recent price action context from stored 1m candles.
    Returns trend, volatility, recent S/R levels from price action.
    """
    fpath = candles_dir / f"{symbol}_1m.csv"
    if not fpath.exists():
        return {}

    df = pd.read_csv(str(fpath), parse_dates=["timestamp"])
    if len(df) < 20:
        return {}

    df = df.sort_values("timestamp").tail(120)  # Last 2 hours
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    # Trend: linear regression slope on last 60 bars
    n = min(60, len(closes))
    x = np.arange(n)
    slope = np.polyfit(x, closes[-n:], 1)[0]
    slope_pct = slope / closes[-1] * 100 * 60  # % per hour

    # Volatility: ATR-like measure
    ranges = highs[-n:] - lows[-n:]
    avg_range_pct = np.mean(ranges) / closes[-1] * 100

    # Recent swing highs/lows (simple peak detection)
    swing_highs = []
    swing_lows = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and \
           highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            swing_highs.append(float(highs[i]))
        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and \
           lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            swing_lows.append(float(lows[i]))

    current_price = float(closes[-1])

    # Find nearest swing levels
    res_from_swings = [h for h in swing_highs if h > current_price]
    sup_from_swings = [l for l in swing_lows if l < current_price]

    nearest_swing_res = min(res_from_swings) if res_from_swings else None
    nearest_swing_sup = max(sup_from_swings) if sup_from_swings else None

    trend = "up" if slope_pct > 0.05 else ("down" if slope_pct < -0.05 else "flat")

    return {
        "current_price": current_price,
        "trend_1h": trend,
        "slope_pct_per_hr": round(slope_pct, 4),
        "avg_bar_range_pct": round(avg_range_pct, 4),
        "n_swing_highs": len(swing_highs),
        "n_swing_lows": len(swing_lows),
        "nearest_swing_res": nearest_swing_res,
        "nearest_swing_sup": nearest_swing_sup,
        "nearest_swing_res_pct": round((nearest_swing_res - current_price) / current_price * 100, 3) if nearest_swing_res else None,
        "nearest_swing_sup_pct": round((current_price - nearest_swing_sup) / current_price * 100, 3) if nearest_swing_sup else None,
    }


# ---------------------------------------------------------------------------
# Historical Depth Persistence
# ---------------------------------------------------------------------------

def check_wall_persistence(symbol: str, wall_price: float,
                            side: str, tolerance_pct: float = 0.3,
                            summary_dir: Path = DEPTH_SUMMARY_DIR) -> dict:
    """
    Check if a wall at a given price has persisted over multiple snapshots.
    Persistent walls are more reliable for TP/SL placement.
    """
    fpath = summary_dir / f"{symbol}_depth_summary.csv"
    if not fpath.exists():
        return {"persistent": False, "snapshots": 0, "appearances": 0}

    df = pd.read_csv(str(fpath))
    if len(df) == 0:
        return {"persistent": False, "snapshots": 0, "appearances": 0}

    total = len(df)
    col = "top_resistance_price" if side == "resistance" else "top_support_price"
    if col not in df.columns:
        return {"persistent": False, "snapshots": total, "appearances": 0}

    # Count how many snapshots have a wall near this price
    appearances = 0
    for _, row in df.iterrows():
        p = row.get(col)
        if pd.notna(p) and abs(float(p) - wall_price) / wall_price * 100 <= tolerance_pct:
            appearances += 1

    pct = appearances / total * 100 if total > 0 else 0

    return {
        "persistent": pct >= 30,  # appears in 30%+ of snapshots
        "snapshots": total,
        "appearances": appearances,
        "persistence_pct": round(pct, 1),
    }


# ---------------------------------------------------------------------------
# Full Analysis for a Symbol
# ---------------------------------------------------------------------------

def analyze_symbol(symbol: str, depth_data: Optional[dict] = None,
                   min_tp_pct: float = 1.0, min_rr: float = 1.0) -> dict:
    """
    Full depth TP/SL analysis for a symbol.
    Fetches fresh depth if not provided.
    Combines depth analysis with price context and wall persistence.
    """
    if depth_data is None:
        depth_data = fetch_bitunix_depth(symbol, limit="max")
        if not depth_data:
            return {"symbol": symbol, "error": "No depth data"}

    asks = depth_data.get("asks", [])
    bids = depth_data.get("bids", [])
    if not asks or not bids:
        return {"symbol": symbol, "error": "Empty order book"}

    best_ask = float(asks[0][0]) if isinstance(asks[0], list) else float(asks[0])
    best_bid = float(bids[0][0]) if isinstance(bids[0], list) else float(bids[0])
    current_price = (best_ask + best_bid) / 2

    # Get price context
    price_ctx = get_price_context(symbol)

    # Compute depth TP/SL for all 4 combos
    combos = {}
    for strategy in ["momentum", "mean_reversion"]:
        for side in ["long", "short"]:
            key = f"{strategy}_{side}"
            result = compute_depth_tp_sl(
                depth_data, current_price,
                side=side, strategy=strategy,
                min_tp_pct=min_tp_pct, min_rr=min_rr,
            )
            result["symbol"] = symbol

            # Check wall persistence for best combo
            if result.get("best_combo"):
                sl_price = result["best_combo"]["sl"]["price"]
                tp_price = result["best_combo"]["tp"]["price"]
                sl_side = "support" if side == "long" else "resistance"
                tp_side = "resistance" if side == "long" else "support"
                result["sl_persistence"] = check_wall_persistence(
                    symbol, sl_price, sl_side)
                result["tp_persistence"] = check_wall_persistence(
                    symbol, tp_price, tp_side)

            combos[key] = result

    return {
        "symbol": symbol,
        "price": current_price,
        "price_context": price_ctx,
        "combos": combos,
        "depth_imbalance_1pct": combos.get("momentum_long", {}).get("depth_imbalance_1pct", 0),
        "depth_imbalance_2pct": combos.get("momentum_long", {}).get("depth_imbalance_2pct", 0),
    }


# ---------------------------------------------------------------------------
# Output Formatters
# ---------------------------------------------------------------------------

def format_analysis(result: dict) -> str:
    """Format a single symbol analysis for console output."""
    lines = []
    sym = result["symbol"]
    price = result.get("price", 0)
    ctx = result.get("price_context", {})
    imb1 = result.get("depth_imbalance_1pct", 0)
    imb2 = result.get("depth_imbalance_2pct", 0)

    lines.append(f"{'=' * 70}")
    lines.append(f" {sym}  |  Price: {price:.6g}  |  Imbalance 1%: {imb1:+.3f}  2%: {imb2:+.3f}")

    if ctx:
        trend = ctx.get("trend_1h", "?")
        slope = ctx.get("slope_pct_per_hr", 0)
        vol = ctx.get("avg_bar_range_pct", 0)
        swing_r = ctx.get("nearest_swing_res_pct")
        swing_s = ctx.get("nearest_swing_sup_pct")
        lines.append(f" Trend: {trend} ({slope:+.3f}%/hr)  |  "
                     f"Bar range: {vol:.3f}%  |  "
                     f"Swing R: {f'{swing_r:.2f}%' if swing_r else 'none'}  "
                     f"S: {f'{swing_s:.2f}%' if swing_s else 'none'}")
    lines.append(f"{'=' * 70}")

    for strategy in ["momentum", "mean_reversion"]:
        strat_label = "MOMENTUM" if strategy == "momentum" else "MEAN REVERSION"
        lines.append(f"\n  --- {strat_label} ---")

        for side in ["long", "short"]:
            key = f"{strategy}_{side}"
            combo_data = result.get("combos", {}).get(key, {})
            best = combo_data.get("best_combo")

            side_label = "LONG" if side == "long" else "SHORT"

            if best:
                tp = best["tp"]
                sl = best["sl"]
                rr = best["rr"]

                # Persistence indicators
                sl_persist = combo_data.get("sl_persistence", {})
                tp_persist = combo_data.get("tp_persistence", {})
                sl_p = f" [persistent {sl_persist['persistence_pct']:.0f}%]" \
                    if sl_persist.get("persistent") else ""
                tp_p = f" [persistent {tp_persist['persistence_pct']:.0f}%]" \
                    if tp_persist.get("persistent") else ""

                lines.append(f"  {side_label}:")
                lines.append(f"    TP: {tp['price']:.6g} ({tp['dist_pct']:.2f}%)  "
                             f"- {tp['reason']}{tp_p}")
                lines.append(f"    SL: {sl['price']:.6g} ({sl['dist_pct']:.2f}%)  "
                             f"- {sl['reason']}{sl_p}")
                lines.append(f"    RR: {rr:.2f}  |  Score: {best['score']:.1f}")
            else:
                n_tp = len(combo_data.get("tp_candidates", []))
                n_sl = len(combo_data.get("sl_candidates", []))
                lines.append(f"  {side_label}: No valid combo (TP>={combo_data.get('min_tp_pct', 1.0)}%, "
                             f"RR>=1.0)  [{n_tp} TP / {n_sl} SL candidates]")

    # Show wall clusters
    combos = result.get("combos", {})
    any_combo = next(iter(combos.values()), {})
    res_c = any_combo.get("resistance_clusters", [])
    sup_c = any_combo.get("support_clusters", [])

    if res_c:
        lines.append(f"\n  Resistance walls:")
        for w in res_c[:3]:
            lines.append(f"    {w['price']:.6g} ({w['dist_pct']:.2f}%)  "
                         f"${w['usd_value']:,.0f}  {w['strength']}x avg  "
                         f"{w['n_levels']} levels")
    if sup_c:
        lines.append(f"  Support walls:")
        for w in sup_c[:3]:
            lines.append(f"    {w['price']:.6g} ({w['dist_pct']:.2f}%)  "
                         f"${w['usd_value']:,.0f}  {w['strength']}x avg  "
                         f"{w['n_levels']} levels")

    # Show thin zones
    thin_a = any_combo.get("thin_zones_above", [])
    thin_b = any_combo.get("thin_zones_below", [])
    if thin_a or thin_b:
        lines.append(f"\n  Thin zones (fast-move potential):")
        for z in thin_a[:2]:
            lines.append(f"    ABOVE: {z['start_dist_pct']:.1f}%-{z['end_dist_pct']:.1f}%  "
                         f"(depth {z['avg_qty_ratio']:.0%} of avg)")
        for z in thin_b[:2]:
            lines.append(f"    BELOW: {z['start_dist_pct']:.1f}%-{z['end_dist_pct']:.1f}%  "
                         f"(depth {z['avg_qty_ratio']:.0%} of avg)")

    return "\n".join(lines)


def save_scan_results(results: list[dict], output_file: str = "depth_tp_sl_scan.csv"):
    """Save scan results to CSV for review."""
    rows = []
    for r in results:
        if "error" in r:
            continue
        sym = r["symbol"]
        price = r.get("price", 0)
        ctx = r.get("price_context", {})

        for strategy in ["momentum", "mean_reversion"]:
            for side in ["long", "short"]:
                key = f"{strategy}_{side}"
                combo_data = r.get("combos", {}).get(key, {})
                best = combo_data.get("best_combo")
                if not best:
                    continue

                sl_persist = combo_data.get("sl_persistence", {})
                tp_persist = combo_data.get("tp_persistence", {})

                rows.append({
                    "timestamp": combo_data.get("timestamp"),
                    "symbol": sym,
                    "price": price,
                    "strategy": strategy,
                    "side": side,
                    "tp_price": best["tp"]["price"],
                    "tp_pct": best["tp"]["dist_pct"],
                    "tp_reason": best["tp"]["reason"],
                    "tp_type": best["tp"].get("type", ""),
                    "sl_price": best["sl"]["price"],
                    "sl_pct": best["sl"]["dist_pct"],
                    "sl_reason": best["sl"]["reason"],
                    "sl_wall_usd": best["sl"].get("wall_usd", 0),
                    "sl_wall_strength": best["sl"].get("wall_strength", 0),
                    "rr": best["rr"],
                    "score": best["score"],
                    "depth_imbalance_1pct": combo_data.get("depth_imbalance_1pct", 0),
                    "depth_imbalance_2pct": combo_data.get("depth_imbalance_2pct", 0),
                    "sl_persistent": sl_persist.get("persistent", False),
                    "sl_persist_pct": sl_persist.get("persistence_pct", 0),
                    "tp_persistent": tp_persist.get("persistent", False),
                    "tp_persist_pct": tp_persist.get("persistence_pct", 0),
                    "trend_1h": ctx.get("trend_1h", ""),
                    "slope_pct_hr": ctx.get("slope_pct_per_hr", 0),
                    "avg_bar_range_pct": ctx.get("avg_bar_range_pct", 0),
                })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(df)} suggestions to {output_file}")
    else:
        print("No valid TP/SL combos found.")


# ---------------------------------------------------------------------------
# CLI Modes
# ---------------------------------------------------------------------------

def mode_live(symbol: str, min_tp_pct: float, min_rr: float):
    """Fetch fresh data and analyze a single symbol."""
    print(f"Fetching live depth for {symbol}...")
    result = analyze_symbol(symbol, min_tp_pct=min_tp_pct, min_rr=min_rr)
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    print(format_analysis(result))


def mode_report(symbol: str):
    """Analyze accumulated depth data for a symbol over time."""
    fpath = DEPTH_SUMMARY_DIR / f"{symbol}_depth_summary.csv"
    if not fpath.exists():
        print(f"No depth history for {symbol}. Run collector first.")
        return

    df = pd.read_csv(str(fpath))
    print(f"\n{'=' * 60}")
    print(f"DEPTH HISTORY REPORT: {symbol}")
    print(f"{'=' * 60}")
    print(f"Snapshots: {len(df)}")
    if len(df) == 0:
        return

    print(f"Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"\nPrice: {df['price'].iloc[-1]:.6g} "
          f"(range: {df['price'].min():.6g} - {df['price'].max():.6g})")

    print(f"\n--- Depth Imbalance ---")
    print(f"  1% band: mean={df['imbalance_1pct'].mean():+.3f}  "
          f"last={df['imbalance_1pct'].iloc[-1]:+.3f}")
    print(f"  2% band: mean={df['imbalance_2pct'].mean():+.3f}  "
          f"last={df['imbalance_2pct'].iloc[-1]:+.3f}")

    buy_mean = df['buy_depth_1pct'].mean()
    sell_mean = df['sell_depth_1pct'].mean()
    print(f"\n--- Avg Depth in 1% Band ---")
    print(f"  Buy:  ${buy_mean:,.0f}")
    print(f"  Sell: ${sell_mean:,.0f}")

    # Resistance wall persistence
    res_prices = df['top_resistance_price'].dropna()
    sup_prices = df['top_support_price'].dropna()

    if len(res_prices) > 0:
        print(f"\n--- Top Resistance Walls ---")
        # Find most common resistance levels
        rounded = res_prices.round(2)
        top_levels = rounded.value_counts().head(5)
        for price, count in top_levels.items():
            pct = count / len(df) * 100
            dist = (price - df['price'].iloc[-1]) / df['price'].iloc[-1] * 100
            print(f"  {price:.6g}  ({dist:+.2f}%)  appeared {count}x ({pct:.0f}% of snapshots)")

    if len(sup_prices) > 0:
        print(f"\n--- Top Support Walls ---")
        rounded = sup_prices.round(2)
        top_levels = rounded.value_counts().head(5)
        for price, count in top_levels.items():
            pct = count / len(df) * 100
            dist = (df['price'].iloc[-1] - price) / df['price'].iloc[-1] * 100
            print(f"  {price:.6g}  ({dist:+.2f}%)  appeared {count}x ({pct:.0f}% of snapshots)")

    print(f"\n--- Spread ---")
    print(f"  Mean: {df['spread_pct'].mean():.4f}%  Max: {df['spread_pct'].max():.4f}%")


def mode_scan(min_vol: float, top_n: int, min_tp_pct: float, min_rr: float):
    """Scan all active coins from Orion and produce TP/SL suggestions."""
    import time

    print("Fetching active coins from Orion...")
    coins = fetch_orion_active_coins(min_vol_5m=min_vol, top_n=top_n)
    if not coins:
        print("No coins found.")
        return

    print(f"Found {len(coins)} coins. Analyzing depth for each...\n")

    results = []
    for i, c in enumerate(coins):
        sym = c["symbol"]
        print(f"  [{i + 1}/{len(coins)}] {sym}...", end="", flush=True)

        depth = fetch_bitunix_depth(sym, limit="max")
        if not depth:
            print(" no depth")
            continue

        result = analyze_symbol(sym, depth_data=depth,
                                min_tp_pct=min_tp_pct, min_rr=min_rr)
        if "error" not in result:
            results.append(result)
            # Count valid combos
            n_valid = sum(1 for k, v in result.get("combos", {}).items()
                          if v.get("best_combo"))
            print(f" {n_valid}/4 valid combos")
        else:
            print(f" {result['error']}")

        time.sleep(0.15)  # Rate limit

    # Print top results
    print(f"\n{'=' * 70}")
    print(f"DEPTH TP/SL SCAN RESULTS — {len(results)} coins analyzed")
    print(f"{'=' * 70}")

    # Rank by best RR across all combos
    ranked = []
    for r in results:
        for key, combo_data in r.get("combos", {}).items():
            best = combo_data.get("best_combo")
            if best:
                ranked.append({
                    "symbol": r["symbol"],
                    "price": r["price"],
                    "strategy": key.rsplit("_", 1)[0],
                    "side": key.rsplit("_", 1)[1],
                    "rr": best["rr"],
                    "score": best["score"],
                    "tp_pct": best["tp"]["dist_pct"],
                    "sl_pct": best["sl"]["dist_pct"],
                    "tp_reason": best["tp"]["reason"][:50],
                    "sl_wall_usd": best["sl"].get("wall_usd", 0),
                    "imbalance": combo_data.get("depth_imbalance_1pct", 0),
                })

    if ranked:
        ranked.sort(key=lambda x: x["score"], reverse=True)

        print(f"\nTop 20 by score (RR * wall strength):")
        print(f"{'Symbol':>14s}  {'Strategy':>5s}  {'Side':>5s}  "
              f"{'TP%':>5s}  {'SL%':>5s}  {'RR':>5s}  {'Score':>5s}  "
              f"{'SL Wall$':>10s}  {'Imb':>6s}")
        print("-" * 85)

        for r in ranked[:20]:
            strat = "MOMO" if r["strategy"] == "momentum" else "MR"
            print(f"{r['symbol']:>14s}  {strat:>5s}  {r['side']:>5s}  "
                  f"{r['tp_pct']:5.2f}  {r['sl_pct']:5.2f}  {r['rr']:5.2f}  "
                  f"{r['score']:5.1f}  {r['sl_wall_usd']:>10,.0f}  "
                  f"{r['imbalance']:+.3f}")

    # Save all results
    save_scan_results(results, "depth_tp_sl_scan.csv")

    # Also print detailed analysis for top 3 coins by score
    if ranked:
        top_syms = list(dict.fromkeys(r["symbol"] for r in ranked[:3]))
        print(f"\n\n{'#' * 70}")
        print(f"DETAILED ANALYSIS — Top {len(top_syms)} coins")
        print(f"{'#' * 70}")
        for r in results:
            if r["symbol"] in top_syms:
                print(format_analysis(r))
                print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Depth-Based TP/SL Analyzer")
    parser.add_argument("--live", type=str, metavar="SYMBOL",
                        help="Analyze live depth for a symbol (e.g., BTCUSDT)")
    parser.add_argument("--report", type=str, metavar="SYMBOL",
                        help="Report on accumulated depth history for a symbol")
    parser.add_argument("--scan", action="store_true",
                        help="Scan all active Orion coins")
    parser.add_argument("--min-vol", type=float, default=200_000,
                        help="Min 5min volume for scan mode (default: 200000)")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Max coins for scan (default: 30)")
    parser.add_argument("--min-tp", type=float, default=1.0,
                        help="Min TP distance %% (default: 1.0)")
    parser.add_argument("--min-rr", type=float, default=1.0,
                        help="Min reward/risk ratio (default: 1.0)")

    args = parser.parse_args()

    if args.live:
        mode_live(args.live.upper(), args.min_tp, args.min_rr)
    elif args.report:
        mode_report(args.report.upper())
    elif args.scan:
        mode_scan(args.min_vol, args.top_n, args.min_tp, args.min_rr)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python depth_tp_sl_analyzer.py --live BTCUSDT")
        print("  python depth_tp_sl_analyzer.py --report ETHUSDT")
        print("  python depth_tp_sl_analyzer.py --scan --min-vol 200000")


if __name__ == "__main__":
    main()
