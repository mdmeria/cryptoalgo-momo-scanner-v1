#!/usr/bin/env python3
"""
Live Data Collector

Continuously collects:
  1. Active coins from Orion Terminal (5min vol > $200k)
  2. Order book depth snapshots from Bitunix
  3. 1-minute candles from Bitunix

Stores data in datasets/live/ with rolling files per symbol.

Usage:
  python live_data_collector.py [--interval 60] [--min-vol 200000]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

handler = logging.StreamHandler(stream=sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s",
                                        datefmt="%H:%M:%S"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BITUNIX_BASE = "https://fapi.bitunix.com"
ORION_URL = "https://screener.orionterminal.com/api/screener"

LIVE_DIR = Path("datasets/live")
DEPTH_DIR = LIVE_DIR / "depth"
CANDLES_DIR = LIVE_DIR / "candles_1m"
DEPTH_SUMMARY_DIR = LIVE_DIR / "depth_summary"
TP_SL_DIR = LIVE_DIR / "tp_sl_suggestions"


# ---------------------------------------------------------------------------
# Orion Screener
# ---------------------------------------------------------------------------

def fetch_orion_active_coins(min_vol_5m: float = 200_000,
                              top_n: int = 50) -> list[dict]:
    """Fetch active coins from Orion Terminal screener."""
    try:
        resp = requests.get(ORION_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Orion fetch failed: %s", e)
        return []

    tickers = data.get("tickers", [])
    results = []

    for t in tickers:
        symbol = str(t.get("symbol", "")).upper()
        if not symbol.endswith("USDT"):
            continue
        if not symbol.isascii():
            continue

        tf5m = t.get("tf5m") or {}
        volume = float(tf5m.get("volume", 0) or 0)
        trades = int(tf5m.get("trades", 0) or 0)
        volatility = float(tf5m.get("volatility", 0) or 0)
        oi_change = float(tf5m.get("oiChange", 0) or 0)
        vdelta = float(tf5m.get("vdelta", 0) or 0)

        if volume < min_vol_5m:
            continue

        results.append({
            "symbol": symbol,
            "volume_5m": volume,
            "trades_5m": trades,
            "volatility_5m": volatility,
            "oi_change_5m": oi_change,
            "vdelta_5m": vdelta,
        })

    results.sort(key=lambda x: x["volume_5m"], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Bitunix API
# ---------------------------------------------------------------------------

def fetch_bitunix_depth(symbol: str, limit: str = "50") -> Optional[dict]:
    """
    Fetch order book depth from Bitunix.
    Returns {asks: [[price, qty], ...], bids: [[price, qty], ...]}
    """
    url = f"{BITUNIX_BASE}/api/v1/futures/market/depth"
    try:
        resp = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            logger.warning("Bitunix depth error for %s: %s", symbol, data.get("msg"))
            return None
        return data.get("data")
    except Exception as e:
        logger.error("Bitunix depth failed for %s: %s", symbol, e)
        return None


def fetch_bitunix_klines(symbol: str, interval: str = "1m",
                          limit: int = 100) -> Optional[list[dict]]:
    """
    Fetch recent klines from Bitunix.
    Returns list of {time, open, high, low, close, quoteVol}.
    """
    url = f"{BITUNIX_BASE}/api/v1/futures/market/kline"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - limit * 60 * 1000  # limit minutes back

    try:
        resp = requests.get(url, params={
            "symbol": symbol,
            "interval": interval,
            "startTime": str(start_ms),
            "endTime": str(now_ms),
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            logger.warning("Bitunix klines error for %s: %s", symbol, data.get("msg"))
            return None
        return data.get("data", [])
    except Exception as e:
        logger.error("Bitunix klines failed for %s: %s", symbol, e)
        return None


# ---------------------------------------------------------------------------
# Depth Analysis
# ---------------------------------------------------------------------------

def analyze_depth(depth_data: dict, current_price: float) -> dict:
    """
    Analyze order book depth to find key levels for TP/SL placement.

    Identifies:
    - Thick bid walls (support) — large buy orders below price
    - Thick ask walls (resistance) — large sell orders above price
    - Thin zones (gaps) — where price could move fast
    - Cumulative depth imbalance (buy vs sell pressure)
    """
    asks = depth_data.get("asks", [])
    bids = depth_data.get("bids", [])

    if not asks or not bids:
        return {}

    # Convert to float
    asks = [[float(a[0]), float(a[1])] for a in asks]
    bids = [[float(b[0]), float(b[1])] for b in bids]

    # Sort: asks ascending (lowest first), bids descending (highest first)
    asks.sort(key=lambda x: x[0])
    bids.sort(key=lambda x: x[0], reverse=True)

    # --- Find thick walls (top N by USD value, within 5% of price) ---
    def find_walls(levels: list, n: int = 5, max_dist_pct: float = 5.0) -> list:
        if not levels:
            return []
        # Filter to levels within max_dist_pct and sort by USD value
        nearby = []
        for l in levels:
            dist = abs(l[0] - current_price) / current_price * 100
            if dist <= max_dist_pct:
                nearby.append(l)
        sorted_by_usd = sorted(nearby, key=lambda x: x[0] * x[1], reverse=True)
        return [{"price": l[0], "qty": l[1],
                 "usd_value": round(l[0] * l[1], 2),
                 "dist_pct": abs(l[0] - current_price) / current_price * 100}
                for l in sorted_by_usd[:n]]

    ask_walls = find_walls(asks, 5)
    bid_walls = find_walls(bids, 5)

    # --- Cumulative depth within % bands ---
    def cum_depth_in_band(levels: list, pct: float) -> float:
        total = 0.0
        for price, qty in levels:
            if abs(price - current_price) / current_price * 100 <= pct:
                total += qty * price  # USD value
        return total

    buy_depth_1pct = cum_depth_in_band(bids, 1.0)
    sell_depth_1pct = cum_depth_in_band(asks, 1.0)
    buy_depth_2pct = cum_depth_in_band(bids, 2.0)
    sell_depth_2pct = cum_depth_in_band(asks, 2.0)

    # Imbalance: positive = more buy pressure, negative = more sell pressure
    imbalance_1pct = (buy_depth_1pct - sell_depth_1pct) / (buy_depth_1pct + sell_depth_1pct) \
        if (buy_depth_1pct + sell_depth_1pct) > 0 else 0
    imbalance_2pct = (buy_depth_2pct - sell_depth_2pct) / (buy_depth_2pct + sell_depth_2pct) \
        if (buy_depth_2pct + sell_depth_2pct) > 0 else 0

    # --- Compute average USD value per level within 5% band (for fair comparison) ---
    asks_nearby_usd = [a[0] * a[1] for a in asks
                       if 0 < (a[0] - current_price) / current_price * 100 <= 5.0]
    bids_nearby_usd = [b[0] * b[1] for b in bids
                       if 0 < (current_price - b[0]) / current_price * 100 <= 5.0]
    avg_ask_usd = sum(asks_nearby_usd) / len(asks_nearby_usd) if asks_nearby_usd else 0
    avg_bid_usd = sum(bids_nearby_usd) / len(bids_nearby_usd) if bids_nearby_usd else 0

    # --- Find nearest thick wall on each side (using USD value) ---
    nearest_resistance = None
    for a in asks:
        dist = (a[0] - current_price) / current_price * 100
        usd_val = a[0] * a[1]
        if dist >= 0.3 and avg_ask_usd > 0 and usd_val >= avg_ask_usd * 2:
            nearest_resistance = {"price": a[0], "qty": a[1],
                                  "usd_value": round(usd_val, 2),
                                  "dist_pct": round(dist, 3)}
            break

    nearest_support = None
    for b in bids:
        dist = (current_price - b[0]) / current_price * 100
        usd_val = b[0] * b[1]
        if dist >= 0.3 and avg_bid_usd > 0 and usd_val >= avg_bid_usd * 2:
            nearest_support = {"price": b[0], "qty": b[1],
                               "usd_value": round(usd_val, 2),
                               "dist_pct": round(dist, 3)}
            break

    # --- Find all significant walls within 3% for TP/SL decisions ---
    wall_threshold = 3.0  # multiplier of average USD value to count as a wall
    tp_sl_walls_resistance = []
    for a in asks:
        dist = (a[0] - current_price) / current_price * 100
        usd_val = a[0] * a[1]
        if 0.2 <= dist <= 3.0 and avg_ask_usd > 0 and usd_val >= avg_ask_usd * wall_threshold:
            tp_sl_walls_resistance.append({
                "price": a[0], "qty": a[1],
                "usd_value": round(usd_val, 2),
                "dist_pct": round(dist, 3),
                "strength": round(usd_val / avg_ask_usd, 1),
            })
    tp_sl_walls_resistance.sort(key=lambda x: x["usd_value"], reverse=True)

    tp_sl_walls_support = []
    for b in bids:
        dist = (current_price - b[0]) / current_price * 100
        usd_val = b[0] * b[1]
        if 0.2 <= dist <= 3.0 and avg_bid_usd > 0 and usd_val >= avg_bid_usd * wall_threshold:
            tp_sl_walls_support.append({
                "price": b[0], "qty": b[1],
                "usd_value": round(usd_val, 2),
                "dist_pct": round(dist, 3),
                "strength": round(usd_val / avg_bid_usd, 1),
            })
    tp_sl_walls_support.sort(key=lambda x: x["usd_value"], reverse=True)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_price": current_price,
        "ask_walls": ask_walls,
        "bid_walls": bid_walls,
        "buy_depth_1pct": round(buy_depth_1pct, 2),
        "sell_depth_1pct": round(sell_depth_1pct, 2),
        "buy_depth_2pct": round(buy_depth_2pct, 2),
        "sell_depth_2pct": round(sell_depth_2pct, 2),
        "imbalance_1pct": round(imbalance_1pct, 4),
        "imbalance_2pct": round(imbalance_2pct, 4),
        "nearest_resistance": nearest_resistance,
        "nearest_support": nearest_support,
        "tp_sl_walls_resistance": tp_sl_walls_resistance[:10],
        "tp_sl_walls_support": tp_sl_walls_support[:10],
        "spread_pct": round((asks[0][0] - bids[0][0]) / current_price * 100, 4) if asks and bids else 0,
        "n_ask_levels": len(asks),
        "n_bid_levels": len(bids),
    }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_depth_snapshot(symbol: str, depth_data: dict, analysis: dict):
    """Save raw depth + analysis to daily file."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw_dir = DEPTH_DIR / today
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Save raw depth (append to JSONL)
    raw_file = raw_dir / f"{symbol}_depth.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asks": depth_data.get("asks", [])[:50],  # top 50 levels nearest to mid
        "bids": depth_data.get("bids", [])[:50],
        "ask_walls": analysis.get("ask_walls", []),
        "bid_walls": analysis.get("bid_walls", []),
    }
    with open(raw_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Save analysis summary (append to CSV)
    summary_dir = DEPTH_SUMMARY_DIR
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / f"{symbol}_depth_summary.csv"

    # Top resistance/support walls within 3% for TP/SL
    res_walls = analysis.get("tp_sl_walls_resistance", [])
    sup_walls = analysis.get("tp_sl_walls_support", [])

    flat = {
        "timestamp": analysis.get("timestamp"),
        "price": analysis.get("current_price"),
        "spread_pct": analysis.get("spread_pct"),
        "buy_depth_1pct": analysis.get("buy_depth_1pct"),
        "sell_depth_1pct": analysis.get("sell_depth_1pct"),
        "buy_depth_2pct": analysis.get("buy_depth_2pct"),
        "sell_depth_2pct": analysis.get("sell_depth_2pct"),
        "imbalance_1pct": analysis.get("imbalance_1pct"),
        "imbalance_2pct": analysis.get("imbalance_2pct"),
        "n_resistance_walls": len(res_walls),
        "n_support_walls": len(sup_walls),
        "top_resistance_price": res_walls[0]["price"] if res_walls else None,
        "top_resistance_dist_pct": res_walls[0]["dist_pct"] if res_walls else None,
        "top_resistance_usd": res_walls[0]["usd_value"] if res_walls else None,
        "top_support_price": sup_walls[0]["price"] if sup_walls else None,
        "top_support_dist_pct": sup_walls[0]["dist_pct"] if sup_walls else None,
        "top_support_usd": sup_walls[0]["usd_value"] if sup_walls else None,
    }

    write_header = not summary_file.exists()
    with open(summary_file, "a") as f:
        if write_header:
            f.write(",".join(flat.keys()) + "\n")
        f.write(",".join(str(v) if v is not None else "" for v in flat.values()) + "\n")


def save_candles(symbol: str, klines: list[dict]):
    """Save/update 1m candle file for symbol."""
    CANDLES_DIR.mkdir(parents=True, exist_ok=True)
    candle_file = CANDLES_DIR / f"{symbol}_1m.csv"

    new_rows = []
    for k in klines:
        ts = k.get("time")
        if ts is None:
            continue
        new_rows.append({
            "timestamp": pd.Timestamp(int(ts), unit="ms", tz="UTC"),
            "open": float(k.get("open", 0)),
            "high": float(k.get("high", 0)),
            "low": float(k.get("low", 0)),
            "close": float(k.get("close", 0)),
            "volume": float(k.get("quoteVol", 0)),
        })

    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows)

    if candle_file.exists():
        existing = pd.read_csv(str(candle_file), parse_dates=["timestamp"])
        combined = pd.concat([existing, new_df]).drop_duplicates(subset=["timestamp"])
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        combined = new_df.sort_values("timestamp").reset_index(drop=True)

    combined.to_csv(str(candle_file), index=False)
    return len(new_rows)


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def save_tp_sl_suggestion(symbol: str, depth_data: dict, current_price: float):
    """Run depth TP/SL analysis and save suggestions."""
    try:
        from depth_tp_sl_analyzer import compute_depth_tp_sl
    except ImportError:
        return

    TP_SL_DIR.mkdir(parents=True, exist_ok=True)
    ts_now = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_file = TP_SL_DIR / f"{today}_suggestions.csv"
    write_header = not out_file.exists()

    rows = []
    for strategy in ["momentum", "mean_reversion"]:
        for side in ["long", "short"]:
            result = compute_depth_tp_sl(
                depth_data, current_price,
                side=side, strategy=strategy,
                min_tp_pct=1.0, min_rr=1.0,
            )
            best = result.get("best_combo")
            if not best:
                continue

            rows.append({
                "timestamp": ts_now,
                "symbol": symbol,
                "price": current_price,
                "strategy": strategy,
                "side": side,
                "tp_price": best["tp"]["price"],
                "tp_pct": best["tp"]["dist_pct"],
                "sl_price": best["sl"]["price"],
                "sl_pct": best["sl"]["dist_pct"],
                "rr": best["rr"],
                "score": best["score"],
                "tp_reason": best["tp"].get("reason", ""),
                "sl_wall_usd": best["sl"].get("wall_usd", 0),
                "sl_wall_strength": best["sl"].get("wall_strength", 0),
                "imbalance_1pct": result.get("depth_imbalance_1pct", 0),
            })

    if rows:
        with open(out_file, "a") as f:
            if write_header:
                f.write(",".join(rows[0].keys()) + "\n")
            for row in rows:
                vals = []
                for v in row.values():
                    s = str(v) if v is not None else ""
                    if "," in s:
                        s = f'"{s}"'
                    vals.append(s)
                f.write(",".join(vals) + "\n")


def collection_cycle(min_vol: float, top_n: int, depth_limit: str):
    """Run one collection cycle."""
    # Step 1: Get active coins from Orion
    coins = fetch_orion_active_coins(min_vol_5m=min_vol, top_n=top_n)

    if not coins:
        logger.warning("No coins found from Orion screener")
        return 0

    logger.info("Found %d coins with 5m vol > $%s", len(coins), f"{min_vol:,.0f}")

    # Log top 5
    for i, c in enumerate(coins[:5]):
        logger.info("  %d. %s  vol=$%s  trades=%d",
                     i + 1, c["symbol"], f"{c['volume_5m']:,.0f}", c["trades_5m"])

    # Save screener snapshot
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    screener_file = LIVE_DIR / "screener_log.csv"
    ts_now = datetime.now(timezone.utc).isoformat()
    write_header = not screener_file.exists()
    with open(screener_file, "a") as f:
        if write_header:
            f.write("timestamp,symbol,volume_5m,trades_5m,volatility_5m,oi_change_5m,vdelta_5m\n")
        for c in coins:
            f.write(f"{ts_now},{c['symbol']},{c['volume_5m']},{c['trades_5m']},"
                    f"{c['volatility_5m']},{c['oi_change_5m']},{c['vdelta_5m']}\n")

    # Step 2: For each coin, fetch depth + candles from Bitunix
    depth_count = 0
    candle_count = 0

    for c in coins:
        sym = c["symbol"]

        # Fetch depth
        depth = fetch_bitunix_depth(sym, limit=depth_limit)
        if depth:
            # Need current price for analysis
            asks = depth.get("asks", [])
            bids = depth.get("bids", [])
            if asks and bids:
                best_ask = float(asks[0][0]) if isinstance(asks[0], list) else float(asks[0])
                best_bid = float(bids[0][0]) if isinstance(bids[0], list) else float(bids[0])
                current_price = (best_ask + best_bid) / 2

                analysis = analyze_depth(depth, current_price)
                save_depth_snapshot(sym, depth, analysis)
                save_tp_sl_suggestion(sym, depth, current_price)
                depth_count += 1

                # Log key levels
                sup = analysis.get("nearest_support")
                res = analysis.get("nearest_resistance")
                imb = analysis.get("imbalance_1pct", 0)
                sup_str = f"S@{sup['price']:.6g}({sup['dist_pct']:.2f}%)" if sup else "none"
                res_str = f"R@{res['price']:.6g}({res['dist_pct']:.2f}%)" if res else "none"
                logger.debug("  %s depth: %s | %s | imb=%.3f", sym, sup_str, res_str, imb)

        # Fetch candles
        klines = fetch_bitunix_klines(sym, interval="1m", limit=100)
        if klines:
            n_saved = save_candles(sym, klines)
            candle_count += n_saved

        # Small delay to respect rate limits (10 req/s)
        time.sleep(0.15)

    logger.info("Cycle done: %d depth snapshots, %d candle bars saved for %d coins",
                depth_count, candle_count, len(coins))
    return len(coins)


def main():
    parser = argparse.ArgumentParser(description="Live Data Collector")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between collection cycles (default: 60)")
    parser.add_argument("--min-vol", type=float, default=200_000,
                        help="Min 5-minute volume in USD (default: 200000)")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Max coins to collect (default: 30)")
    parser.add_argument("--depth-limit", type=str, default="max",
                        help="Depth levels: 1/5/15/50/max (default: max)")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (no loop)")
    args = parser.parse_args()

    logger.info("Starting Live Data Collector")
    logger.info("  Interval: %ds | Min vol: $%s | Top N: %d | Depth: %s",
                args.interval, f"{args.min_vol:,.0f}", args.top_n, args.depth_limit)
    logger.info("  Output: %s", LIVE_DIR.resolve())

    # Create dirs
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    CANDLES_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    TP_SL_DIR.mkdir(parents=True, exist_ok=True)

    if args.once:
        collection_cycle(args.min_vol, args.top_n, args.depth_limit)
        return

    cycle = 0
    while True:
        cycle += 1
        logger.info("=== Cycle %d ===", cycle)
        try:
            collection_cycle(args.min_vol, args.top_n, args.depth_limit)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error("Cycle error: %s", e)

        logger.info("Sleeping %ds...", args.interval)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break


if __name__ == "__main__":
    main()
