#!/usr/bin/env python3
"""
Depth Snapshot — capture orderbook metrics at trade entry and close.

Provides structured depth data for post-trade analysis:
  - Does favorable imbalance predict winners?
  - Do blocking walls between entry-TP predict losers?
  - Do protection walls behind SL predict winners?
  - Are walls real or spoofed (present at entry but gone at close)?
"""

import numpy as np
from typing import Optional


def compute_depth_snapshot(depth_data: dict, entry_price: float,
                           tp_price: float, sl_price: float,
                           side: str) -> Optional[dict]:
    """
    Compute a structured depth snapshot for a trade.

    Args:
        depth_data: {asks: [[price, qty], ...], bids: [[price, qty], ...]}
        entry_price: trade entry price
        tp_price: take profit price
        sl_price: stop loss price
        side: "long" or "short"

    Returns dict with all depth metrics for logging, or None if no data.
    """
    if not depth_data:
        return None

    asks_raw = depth_data.get("asks", [])
    bids_raw = depth_data.get("bids", [])

    if not asks_raw or not bids_raw:
        return None

    # Parse to floats
    asks = [(float(a[0]), float(a[1])) for a in asks_raw]
    bids = [(float(b[0]), float(b[1])) for b in bids_raw]

    # Sort: asks ascending, bids descending
    asks.sort(key=lambda x: x[0])
    bids.sort(key=lambda x: x[0], reverse=True)

    best_ask = asks[0][0] if asks else entry_price
    best_bid = bids[0][0] if bids else entry_price
    spread = best_ask - best_bid
    spread_pct = spread / entry_price * 100 if entry_price > 0 else 0

    # ── Band volumes: bid/ask volume within 0.5%, 1%, 2% of entry ──
    ask_vol_05 = sum(p * q for p, q in asks if p <= entry_price * 1.005)
    ask_vol_1 = sum(p * q for p, q in asks if p <= entry_price * 1.01)
    ask_vol_2 = sum(p * q for p, q in asks if p <= entry_price * 1.02)
    bid_vol_05 = sum(p * q for p, q in bids if p >= entry_price * 0.995)
    bid_vol_1 = sum(p * q for p, q in bids if p >= entry_price * 0.99)
    bid_vol_2 = sum(p * q for p, q in bids if p >= entry_price * 0.98)

    # ── Imbalance (positive = more bids = bullish) ──
    imb_05 = (bid_vol_05 - ask_vol_05) / (bid_vol_05 + ask_vol_05) if (bid_vol_05 + ask_vol_05) > 0 else 0
    imb_1 = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1) if (bid_vol_1 + ask_vol_1) > 0 else 0
    imb_2 = (bid_vol_2 - ask_vol_2) / (bid_vol_2 + ask_vol_2) if (bid_vol_2 + ask_vol_2) > 0 else 0

    # ── Average level USD (for wall strength calculation) ──
    all_levels_usd = [p * q for p, q in asks[:20]] + [p * q for p, q in bids[:20]]
    avg_level_usd = np.mean(all_levels_usd) if all_levels_usd else 1.0

    # ── Walls between entry and TP (blocking walls) ──
    if side == "long":
        # Resistance walls (asks) between entry and TP
        blocking_walls = [(p, q, p * q) for p, q in asks if entry_price < p < tp_price]
    else:
        # Support walls (bids) between TP and entry
        blocking_walls = [(p, q, p * q) for p, q in bids if tp_price < p < entry_price]

    blocking_walls.sort(key=lambda x: x[2], reverse=True)  # by USD size

    # Top 3 blocking walls
    block_1_price, block_1_usd, block_1_str, block_1_dist = 0, 0, 0, 0
    block_2_price, block_2_usd, block_2_str = 0, 0, 0
    block_3_price, block_3_usd, block_3_str = 0, 0, 0

    if len(blocking_walls) >= 1:
        bw = blocking_walls[0]
        block_1_price = bw[0]
        block_1_usd = bw[2]
        block_1_str = bw[2] / avg_level_usd if avg_level_usd > 0 else 0
        block_1_dist = abs(bw[0] - entry_price) / entry_price * 100
    if len(blocking_walls) >= 2:
        bw = blocking_walls[1]
        block_2_price = bw[0]
        block_2_usd = bw[2]
        block_2_str = bw[2] / avg_level_usd if avg_level_usd > 0 else 0
    if len(blocking_walls) >= 3:
        bw = blocking_walls[2]
        block_3_price = bw[0]
        block_3_usd = bw[2]
        block_3_str = bw[2] / avg_level_usd if avg_level_usd > 0 else 0

    n_blocking = len(blocking_walls)
    n_thick_blocking = sum(1 for _, _, usd in blocking_walls if usd > avg_level_usd * 2)
    total_blocking_usd = sum(usd for _, _, usd in blocking_walls)

    # ── Protection walls behind SL ──
    if side == "long":
        # Support walls (bids) between SL and entry
        protect_walls = [(p, q, p * q) for p, q in bids if sl_price < p < entry_price]
    else:
        # Resistance walls (asks) between entry and SL
        protect_walls = [(p, q, p * q) for p, q in asks if entry_price < p < sl_price]

    protect_walls.sort(key=lambda x: x[2], reverse=True)

    protect_1_price, protect_1_usd, protect_1_str, protect_1_dist = 0, 0, 0, 0
    if len(protect_walls) >= 1:
        pw = protect_walls[0]
        protect_1_price = pw[0]
        protect_1_usd = pw[2]
        protect_1_str = pw[2] / avg_level_usd if avg_level_usd > 0 else 0
        protect_1_dist = abs(pw[0] - sl_price) / entry_price * 100

    n_protect = len(protect_walls)
    n_thick_protect = sum(1 for _, _, usd in protect_walls if usd > avg_level_usd * 2)
    total_protect_usd = sum(usd for _, _, usd in protect_walls)

    return {
        # Spread
        "d_spread_pct": round(spread_pct, 5),

        # Band volumes (USD)
        "d_ask_vol_05": round(ask_vol_05, 0),
        "d_ask_vol_1": round(ask_vol_1, 0),
        "d_ask_vol_2": round(ask_vol_2, 0),
        "d_bid_vol_05": round(bid_vol_05, 0),
        "d_bid_vol_1": round(bid_vol_1, 0),
        "d_bid_vol_2": round(bid_vol_2, 0),

        # Imbalance (-1 to +1, positive = more bids)
        "d_imb_05": round(imb_05, 4),
        "d_imb_1": round(imb_1, 4),
        "d_imb_2": round(imb_2, 4),

        # Blocking walls (between entry and TP)
        "d_block_n": n_blocking,
        "d_block_n_thick": n_thick_blocking,
        "d_block_total_usd": round(total_blocking_usd, 0),
        "d_block_1_price": round(block_1_price, 8),
        "d_block_1_usd": round(block_1_usd, 0),
        "d_block_1_str": round(block_1_str, 2),
        "d_block_1_dist": round(block_1_dist, 3),
        "d_block_2_usd": round(block_2_usd, 0),
        "d_block_2_str": round(block_2_str, 2),
        "d_block_3_usd": round(block_3_usd, 0),
        "d_block_3_str": round(block_3_str, 2),

        # Protection walls (behind SL)
        "d_protect_n": n_protect,
        "d_protect_n_thick": n_thick_protect,
        "d_protect_total_usd": round(total_protect_usd, 0),
        "d_protect_1_price": round(protect_1_price, 8),
        "d_protect_1_usd": round(protect_1_usd, 0),
        "d_protect_1_str": round(protect_1_str, 2),
        "d_protect_1_dist": round(protect_1_dist, 3),

        # Reference
        "d_avg_level_usd": round(avg_level_usd, 0),
    }


def compute_close_snapshot(depth_data: dict, exit_price: float,
                            entry_snapshot: dict, side: str) -> Optional[dict]:
    """
    Light depth snapshot at trade close — check if blocking wall still exists.

    Args:
        depth_data: current orderbook
        exit_price: actual exit price
        entry_snapshot: the entry snapshot dict (to compare)
        side: "long" or "short"

    Returns dict with close-time depth metrics.
    """
    if not depth_data or not entry_snapshot:
        return None

    asks_raw = depth_data.get("asks", [])
    bids_raw = depth_data.get("bids", [])
    if not asks_raw or not bids_raw:
        return None

    asks = [(float(a[0]), float(a[1])) for a in asks_raw]
    bids = [(float(b[0]), float(b[1])) for b in bids_raw]

    best_ask = asks[0][0] if asks else exit_price
    best_bid = bids[0][0] if bids else exit_price
    spread_pct = (best_ask - best_bid) / exit_price * 100 if exit_price > 0 else 0

    # Band volumes
    ask_vol_1 = sum(p * q for p, q in asks if p <= exit_price * 1.01)
    bid_vol_1 = sum(p * q for p, q in bids if p >= exit_price * 0.99)
    imb_1 = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1) if (bid_vol_1 + ask_vol_1) > 0 else 0

    # Check if the biggest blocking wall from entry is still there
    entry_block_price = entry_snapshot.get("d_block_1_price", 0)
    entry_block_usd = entry_snapshot.get("d_block_1_usd", 0)
    wall_still_exists = False
    wall_current_usd = 0

    if entry_block_price > 0:
        search_levels = asks if side == "long" else bids
        for p, q in search_levels:
            if abs(p - entry_block_price) / entry_block_price < 0.002:  # within 0.2%
                wall_still_exists = True
                wall_current_usd = p * q
                break

    # Imbalance shift from entry
    entry_imb = entry_snapshot.get("d_imb_1", 0)
    imb_shift = imb_1 - entry_imb

    return {
        "dc_spread_pct": round(spread_pct, 5),
        "dc_imb_1": round(imb_1, 4),
        "dc_imb_shift": round(imb_shift, 4),
        "dc_wall_still_exists": wall_still_exists,
        "dc_wall_current_usd": round(wall_current_usd, 0),
        "dc_wall_entry_usd": round(entry_block_usd, 0),
        "dc_wall_absorbed": not wall_still_exists and entry_block_usd > 0,
    }
