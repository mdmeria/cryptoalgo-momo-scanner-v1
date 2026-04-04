#!/usr/bin/env python3
"""
ZCT Daily Trading Bias Calculator

Automates the 4-observation process from "How I create my daily trading bias":
  1. Velodata Return Buckets  → Direction (Y-axis: longs vs shorts)
  2. Velodata Spaghetti Chart → Strategy Type (X-axis: momentum vs MR)
  3. Orion Tick Count & Volume → Activity Level (strengthens/weakens bias)
  4. Magnitude + Structure    → Final confirmation

Output: 2D bias grid position + allowed strategies + minimum quality thresholds.
"""

import sys, json, time, statistics
from collections import defaultdict
from datetime import datetime, timezone

import requests
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── API endpoints ──────────────────────────────────────────────────────────
VELO_CHANGES_URL = "https://velo.xyz/api/m/changes"
VELO_PRICELINE_URL = "https://velo.xyz/api/m/priceLine"
ORION_SCREENER_URL = "https://screener.orionterminal.com/api/screener"

# ── Thresholds ─────────────────────────────────────────────────────────────
# Return bucket thresholds (ratio: 1.0 = flat, 1.05 = +5%, 0.95 = -5%)
BIG_MOVE_THRESHOLD = 0.05       # 5% move counts as "big mover"
MILD_SKEW_THRESHOLD = 0.55      # 55% of coins in one direction = mild
STRONG_SKEW_THRESHOLD = 0.70    # 70% = strong

# Spaghetti chart: R² thresholds for staircase quality
STAIRCASE_R2 = 0.80             # R² >= 0.80 = clean staircase
CHOP_R2 = 0.40                  # R² <= 0.40 = choppy range

# Orion activity: relative change thresholds
ACTIVITY_HIGH = 1.5             # 50% above baseline = elevated
ACTIVITY_LOW = 0.6              # 40% below baseline = depressed

# Magnitude thresholds for observation 4
LARGE_MOVE_PCT = 15.0           # top movers >= 15% = large
MEDIUM_MOVE_PCT = 5.0           # 5-15% = medium

# ── Excluded symbols (from our backtest config) ────────────────────────────
EXCLUDED = {"BTC", "ETH", "BNB", "XRP", "LTC", "BCH", "YFI", "PAXG", "XAU", "XAG", "XPT"}


def fetch_velo_changes(range_ms=14400000):
    """Obs 1: Fetch 4h return ratios for all coins."""
    try:
        resp = requests.get(VELO_CHANGES_URL, params={"range": range_ms}, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("d", [])
        # data is list of [symbol, ratio] pairs
        coins = {}
        for item in data:
            if isinstance(item, list) and len(item) == 2:
                sym = str(item[0]).upper()
                ratio = float(item[1])
                if sym not in EXCLUDED:
                    coins[sym] = ratio
        return coins
    except Exception as e:
        print(f"  [WARN] Velo changes fetch failed: {e}")
        return {}


def fetch_velo_priceline(range_ms=14400000, resolution="10 minutes", filter_type="Top Gainers"):
    """Obs 2: Fetch price paths for top movers."""
    try:
        # Velo requires %20 encoding, not + encoding for spaces
        url = (
            f"{VELO_PRICELINE_URL}?range={range_ms}"
            f"&resolution={resolution.replace(' ', '%20')}"
            f"&filter={filter_type.replace(' ', '%20')}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("d", [])
        # data is list of [timestamp, symbol, price] triples
        series = defaultdict(list)
        for item in data:
            if isinstance(item, list) and len(item) == 3:
                ts = int(item[0])
                sym = str(item[1]).upper()
                price = float(item[2])
                series[sym].append((ts, price))
        # Sort each series by timestamp
        for sym in series:
            series[sym].sort(key=lambda x: x[0])
        return dict(series)
    except Exception as e:
        print(f"  [WARN] Velo priceline fetch failed ({filter_type}): {e}")
        return {}


def fetch_orion_screener():
    """Obs 3: Fetch tick count and volume from Orion."""
    try:
        resp = requests.get(ORION_SCREENER_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        tickers = data if isinstance(data, list) else data.get("tickers", [])
        return tickers
    except Exception as e:
        print(f"  [WARN] Orion screener fetch failed: {e}")
        return []


def compute_r_squared(prices):
    """Compute R² of linear fit on price series (staircase quality)."""
    if len(prices) < 5:
        return 0.0
    y = np.array(prices, dtype=float)
    x = np.arange(len(y), dtype=float)
    # Normalize to percentage change from start
    if y[0] != 0:
        y = (y / y[0] - 1) * 100
    n = len(y)
    sx = x.sum()
    sy = y.sum()
    sxy = (x * y).sum()
    sxx = (x * x).sum()
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return max(0.0, 1 - ss_res / ss_tot)


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVATION 1: Return Buckets → Direction (Y-axis)
# ═══════════════════════════════════════════════════════════════════════════

def observation_1_direction(changes):
    """
    Determine directional bias from 4h return distribution.
    Returns: (score, label) where score is -2 to +2
      +2 = Strong long, +1 = Mild long, 0 = Neutral, -1 = Mild short, -2 = Strong short
    """
    if not changes:
        return 0, "UNCLEAR (no data)"

    up_coins = sum(1 for r in changes.values() if r > 1.001)   # up > 0.1%
    down_coins = sum(1 for r in changes.values() if r < 0.999)  # down > 0.1%
    total = len(changes)

    if total == 0:
        return 0, "UNCLEAR"

    up_pct = up_coins / total
    down_pct = down_coins / total

    # Count magnitude of moves
    big_up = sum(1 for r in changes.values() if r > 1 + BIG_MOVE_THRESHOLD)
    big_down = sum(1 for r in changes.values() if r < 1 - BIG_MOVE_THRESHOLD)

    # Compute median return
    returns = [(r - 1) * 100 for r in changes.values()]
    median_ret = statistics.median(returns)

    details = (
        f"  Coins up: {up_coins}/{total} ({up_pct:.0%}) | down: {down_coins}/{total} ({down_pct:.0%})\n"
        f"  Big movers (>{BIG_MOVE_THRESHOLD*100:.0f}%): {big_up} up, {big_down} down\n"
        f"  Median 4h return: {median_ret:+.2f}%"
    )

    # Determine bias
    if up_pct >= STRONG_SKEW_THRESHOLD and big_up >= 5:
        return 2, f"STRONG LONG bias (extreme green)\n{details}"
    elif up_pct >= MILD_SKEW_THRESHOLD or (big_up > big_down * 2 and big_up >= 3):
        return 1, f"MILD LONG bias\n{details}"
    elif down_pct >= STRONG_SKEW_THRESHOLD and big_down >= 5:
        return -2, f"STRONG SHORT bias (extreme red)\n{details}"
    elif down_pct >= MILD_SKEW_THRESHOLD or (big_down > big_up * 2 and big_down >= 3):
        return -1, f"MILD SHORT bias\n{details}"
    else:
        return 0, f"NEUTRAL (mixed)\n{details}"


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVATION 2: Spaghetti Chart → Strategy Type (X-axis)
# ═══════════════════════════════════════════════════════════════════════════

def observation_2_strategy_type(gainers_series, losers_series, changes):
    """
    Analyze price paths of top movers to determine momentum vs MR.
    Returns: (score, label) where score is -2 to +2
      +2 = Strong momentum, +1 = Mild momentum, 0 = Neutral,
      -1 = Mild MR, -2 = Strong MR
    """
    all_r2 = []
    staircase_count = 0
    chop_count = 0
    coin_details = []

    for label, series_dict in [("Gainers", gainers_series), ("Losers", losers_series)]:
        for sym, points in series_dict.items():
            prices = [p for _, p in points]
            if len(prices) < 5:
                continue
            # Only analyze coins that moved 5%+
            total_move = abs(prices[-1] / prices[0] - 1) * 100 if prices[0] else 0
            if total_move < 5:
                continue

            r2 = compute_r_squared(prices)
            all_r2.append(r2)
            quality = "staircase" if r2 >= STAIRCASE_R2 else ("chop" if r2 <= CHOP_R2 else "mixed")
            if r2 >= STAIRCASE_R2:
                staircase_count += 1
            elif r2 <= CHOP_R2:
                chop_count += 1
            coin_details.append((sym, total_move, r2, quality))

    total_analyzed = len(all_r2)
    if total_analyzed == 0:
        return 0, "UNCLEAR (no big movers to analyze)"

    avg_r2 = statistics.mean(all_r2)
    staircase_pct = staircase_count / total_analyzed
    chop_pct = chop_count / total_analyzed

    # Show top movers
    coin_details.sort(key=lambda x: x[1], reverse=True)
    top_lines = "\n".join(
        f"    {sym:<10} move={mv:+.1f}%  R²={r2:.2f}  [{q}]"
        for sym, mv, r2, q in coin_details[:10]
    )

    details = (
        f"  Analyzed: {total_analyzed} coins that moved 5%+\n"
        f"  Staircases: {staircase_count} ({staircase_pct:.0%}) | Chop: {chop_count} ({chop_pct:.0%})\n"
        f"  Avg R²: {avg_r2:.2f}\n"
        f"  Top movers:\n{top_lines}"
    )

    if staircase_pct >= 0.7:
        return 2, f"STRONGLY favours MOMENTUM (clean staircases)\n{details}"
    elif staircase_pct >= 0.5:
        return 1, f"MILDLY favours MOMENTUM (mostly staircases)\n{details}"
    elif chop_pct >= 0.7:
        return -2, f"STRONGLY favours MEAN REVERSION (pure chop)\n{details}"
    elif chop_pct >= 0.5:
        return -1, f"MILDLY favours MEAN REVERSION (mostly choppy)\n{details}"
    else:
        return 0, f"NEUTRAL (mixed structures)\n{details}"


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVATION 3: Orion Tick Count & Volume → Activity Level
# ═══════════════════════════════════════════════════════════════════════════

def observation_3_activity(tickers, baseline_file="orion_baseline.json"):
    """
    Compare current tick count and volume to recent baseline.
    Returns: (adjustment, label) where adjustment is -1, 0, or +1
      +1 = Strengthen momentum, 0 = No change, -1 = Strengthen MR
    """
    if not tickers:
        return 0, "UNCLEAR (no Orion data)"

    # Extract top coins by volume (USDT pairs)
    coin_data = []
    for t in tickers:
        sym = str(t.get("symbol", "")).upper()
        if not sym.endswith("USDT"):
            continue
        tf = t.get("tf5m") or {}
        trades = int(tf.get("trades", 0) or 0)
        volume = float(tf.get("volume", 0) or 0)
        if volume > 0:
            coin_data.append({"symbol": sym, "trades": trades, "volume": volume})

    coin_data.sort(key=lambda x: x["volume"], reverse=True)
    top_10 = coin_data[:10]

    if not top_10:
        return 0, "UNCLEAR (no USDT coins)"

    current_trades = statistics.mean([c["trades"] for c in top_10])
    current_volume = statistics.mean([c["volume"] for c in top_10])

    # Load or initialize baseline
    baseline = _load_baseline(baseline_file)
    has_baseline = len(baseline.get("history", [])) >= 5

    # Save current reading to baseline
    _update_baseline(baseline_file, baseline, current_trades, current_volume)

    if not has_baseline:
        top_lines = "\n".join(
            f"    {c['symbol']:<14} trades={c['trades']:>8,}  vol=${c['volume']:>14,.0f}"
            for c in top_10[:5]
        )
        return 0, (
            f"UNCLEAR (need 5+ sessions for baseline, have {len(baseline.get('history', []))})\n"
            f"  Current top 5:\n{top_lines}\n"
            f"  Avg trades (top10): {current_trades:,.0f} | Avg vol: ${current_volume:,.0f}\n"
            f"  Keep running daily to build baseline."
        )

    # Compare to baseline
    hist = baseline["history"][-7:]  # last 7 sessions
    avg_baseline_trades = statistics.mean([h["trades"] for h in hist])
    avg_baseline_volume = statistics.mean([h["volume"] for h in hist])

    trades_ratio = current_trades / avg_baseline_trades if avg_baseline_trades > 0 else 1.0
    volume_ratio = current_volume / avg_baseline_volume if avg_baseline_volume > 0 else 1.0
    combined_ratio = (trades_ratio + volume_ratio) / 2

    top_lines = "\n".join(
        f"    {c['symbol']:<14} trades={c['trades']:>8,}  vol=${c['volume']:>14,.0f}"
        for c in top_10[:5]
    )

    details = (
        f"  Current avg trades (top10): {current_trades:,.0f} (baseline: {avg_baseline_trades:,.0f})\n"
        f"  Current avg volume (top10): ${current_volume:,.0f} (baseline: ${avg_baseline_volume:,.0f})\n"
        f"  Trades ratio: {trades_ratio:.2f}x | Volume ratio: {volume_ratio:.2f}x\n"
        f"  Top 5 coins:\n{top_lines}"
    )

    if combined_ratio >= ACTIVITY_HIGH:
        return 1, f"ELEVATED activity → Strengthen MOMENTUM bias\n{details}"
    elif combined_ratio >= 1.2:
        return 0, f"SLIGHTLY above normal → Lean harder into existing bias\n{details}"
    elif combined_ratio <= ACTIVITY_LOW:
        return -1, f"DEPRESSED activity → Strengthen MEAN REVERSION bias\n{details}"
    elif combined_ratio <= 0.8:
        return 0, f"SLIGHTLY below normal → Lean harder into MR bias\n{details}"
    else:
        return 0, f"NORMAL activity → No adjustment\n{details}"


def _load_baseline(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"history": []}


def _update_baseline(path, baseline, trades, volume):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hist = baseline.get("history", [])
    # Only one entry per day
    if hist and hist[-1].get("date") == today:
        hist[-1] = {"date": today, "trades": trades, "volume": volume}
    else:
        hist.append({"date": today, "trades": trades, "volume": volume})
    # Keep last 30 days
    baseline["history"] = hist[-30:]
    try:
        with open(path, "w") as f:
            json.dump(baseline, f, indent=2)
    except Exception as e:
        print(f"  [WARN] Could not save baseline: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVATION 4: Magnitude + Structure → Final Confirmation
# ═══════════════════════════════════════════════════════════════════════════

def observation_4_magnitude(changes, gainers_series, losers_series):
    """
    Check magnitude of top movers + chart structure quality.
    Returns: (adjustment, label) where adjustment is -1, 0, or +1
      +1 = Confirms momentum, -1 = Confirms MR, 0 = No change
    """
    if not changes:
        return 0, "UNCLEAR (no data)"

    # Sort by absolute move size
    sorted_coins = sorted(changes.items(), key=lambda x: abs(x[1] - 1), reverse=True)
    top_gainers = [(s, (r - 1) * 100) for s, r in sorted_coins if r > 1][:5]
    top_losers = [(s, (r - 1) * 100) for s, r in sorted_coins if r < 1][:5]

    # Magnitude check
    large_moves = sum(1 for _, pct in top_gainers + top_losers if abs(pct) >= LARGE_MOVE_PCT)
    medium_moves = sum(1 for _, pct in top_gainers + top_losers if MEDIUM_MOVE_PCT <= abs(pct) < LARGE_MOVE_PCT)
    small_moves = sum(1 for _, pct in top_gainers + top_losers if abs(pct) < MEDIUM_MOVE_PCT)

    # Structure check: R² of top movers with price data
    all_series = {**gainers_series, **losers_series}
    clean_count = 0
    choppy_count = 0
    for sym, _ in top_gainers + top_losers:
        sym_upper = sym.upper()
        if sym_upper in all_series:
            prices = [p for _, p in all_series[sym_upper]]
            r2 = compute_r_squared(prices)
            if r2 >= STAIRCASE_R2:
                clean_count += 1
            elif r2 <= CHOP_R2:
                choppy_count += 1

    gainer_lines = "  Top gainers: " + ", ".join(f"{s} ({p:+.1f}%)" for s, p in top_gainers[:5])
    loser_lines = "  Top losers:  " + ", ".join(f"{s} ({p:+.1f}%)" for s, p in top_losers[:5])
    details = (
        f"{gainer_lines}\n{loser_lines}\n"
        f"  Large (>15%): {large_moves} | Medium (5-15%): {medium_moves} | Small (<5%): {small_moves}\n"
        f"  Clean structures: {clean_count} | Choppy: {choppy_count}"
    )

    if large_moves >= 3 and clean_count > choppy_count:
        return 1, f"Large moves + clean structures → Confirms MOMENTUM\n{details}"
    elif small_moves >= 7 and choppy_count > clean_count:
        return -1, f"Small moves + choppy charts → Confirms MEAN REVERSION\n{details}"
    else:
        return 0, f"Mixed magnitude/structure → No adjustment\n{details}"


# ═══════════════════════════════════════════════════════════════════════════
# FINAL BIAS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_final_bias(direction_score, strategy_score, activity_adj, magnitude_adj):
    """
    Combine all observations into final 2D bias.

    Y-axis (direction): direction_score (-2 to +2)
    X-axis (strategy):  strategy_score (-2 to +2) + activity_adj + magnitude_adj

    Returns dict with bias position and allowed strategies.
    """
    y = direction_score  # -2 to +2 (shorts to longs)
    x = strategy_score + activity_adj + magnitude_adj  # negative = MR, positive = Momo
    x = max(-3, min(3, x))  # clamp

    # Direction label
    if y >= 2:
        dir_label = "STRONG LONGS"
    elif y >= 1:
        dir_label = "MILD LONGS"
    elif y <= -2:
        dir_label = "STRONG SHORTS"
    elif y <= -1:
        dir_label = "MILD SHORTS"
    else:
        dir_label = "NEUTRAL DIRECTION"

    # Strategy label
    if x >= 2:
        strat_label = "STRONG MOMENTUM"
    elif x >= 1:
        strat_label = "MILD MOMENTUM"
    elif x <= -2:
        strat_label = "STRONG MEAN REVERSION"
    elif x <= -1:
        strat_label = "MILD MEAN REVERSION"
    else:
        strat_label = "NEUTRAL STRATEGY"

    # Determine allowed strategies and quality thresholds
    strategies = _compute_strategy_thresholds(y, x)

    return {
        "direction_score": y,
        "strategy_score": x,
        "direction_label": dir_label,
        "strategy_label": strat_label,
        "grid_position": f"({x:+d}, {y:+d})",
        "strategies": strategies,
    }


def _compute_strategy_thresholds(y, x):
    """
    Based on grid position, determine which strategies are allowed
    and their minimum quality (DPS) thresholds.

    DPS thresholds: higher = more selective
      - Aligned with bias: lower threshold (DPS 3+)
      - Neutral: medium threshold (DPS 4+)
      - Against bias: high threshold (DPS 5+) or SKIP
    """
    strategies = {}

    # ── Momo Long ──
    if y >= 1 and x >= 1:
        strategies["Momo Long"] = {"status": "ACTIVE", "min_dps": 3, "note": "Aligned — take B+ setups"}
    elif y >= 0 and x >= 0:
        strategies["Momo Long"] = {"status": "SELECTIVE", "min_dps": 4, "note": "Neutral — A setups only"}
    elif y >= -1 and x >= 0:
        strategies["Momo Long"] = {"status": "SELECTIVE", "min_dps": 5, "note": "Against direction — only A+ setups"}
    else:
        strategies["Momo Long"] = {"status": "SKIP", "min_dps": 6, "note": "Not aligned — skip"}

    # ── Momo Short ──
    if y <= -1 and x >= 1:
        strategies["Momo Short"] = {"status": "ACTIVE", "min_dps": 3, "note": "Aligned — take B+ setups"}
    elif y <= 0 and x >= 0:
        strategies["Momo Short"] = {"status": "SELECTIVE", "min_dps": 4, "note": "Neutral — A setups only"}
    elif y <= 1 and x >= 0:
        strategies["Momo Short"] = {"status": "SELECTIVE", "min_dps": 5, "note": "Against direction — only A+ setups"}
    else:
        strategies["Momo Short"] = {"status": "SKIP", "min_dps": 6, "note": "Not aligned — skip"}

    # ── MR Long ──
    if y >= 1 and x <= -1:
        strategies["MR Long"] = {"status": "ACTIVE", "min_dps": 3, "note": "Aligned — take B+ setups"}
    elif y >= 0 and x <= 0:
        strategies["MR Long"] = {"status": "SELECTIVE", "min_dps": 4, "note": "Neutral — A setups only"}
    elif y >= -1 and x <= 0:
        strategies["MR Long"] = {"status": "SELECTIVE", "min_dps": 5, "note": "Against direction — only A+ setups"}
    else:
        strategies["MR Long"] = {"status": "SKIP", "min_dps": 6, "note": "Not aligned — skip"}

    # ── MR Short ──
    if y <= -1 and x <= -1:
        strategies["MR Short"] = {"status": "ACTIVE", "min_dps": 3, "note": "Aligned — take B+ setups"}
    elif y <= 0 and x <= 0:
        strategies["MR Short"] = {"status": "SELECTIVE", "min_dps": 4, "note": "Neutral — A setups only"}
    elif y <= 1 and x <= 0:
        strategies["MR Short"] = {"status": "SELECTIVE", "min_dps": 5, "note": "Against direction — only A+ setups"}
    else:
        strategies["MR Short"] = {"status": "SKIP", "min_dps": 6, "note": "Not aligned — skip"}

    return strategies


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_bias_grid(bias):
    """Print a visual representation of the bias grid."""
    y = bias["direction_score"]
    x = bias["strategy_score"]

    print("\n" + "=" * 60)
    print("           DAILY TRADING BIAS")
    print("=" * 60)

    # Grid visualization
    # Simple text-based grid
    grid_y_labels = {2: "Strong Long", 1: "Mild Long", 0: "Neutral", -1: "Mild Short", -2: "Strong Short"}
    grid_x_labels = {-2: "Strong MR", -1: "Mild MR", 0: "Neutral", 1: "Mild Momo", 2: "Strong Momo"}

    print(f"                       LONGS (+2)")
    print(f"                         ^")
    for row_y in [2, 1, 0, -1, -2]:
        marker = " <*>" if row_y == y else "    "
        if row_y == 0:
            print(f"  MR (-2) <--------------+--------------> MOMO (+2){marker}")
        else:
            print(f"                         |{marker}")
    print(f"                         v")
    print(f"                      SHORTS (-2)")

    # Show position with labels
    y_lbl = grid_y_labels.get(max(-2, min(2, y)), "?")
    x_lbl = grid_x_labels.get(max(-2, min(2, x)), "?")

    print()
    print(f"  Position: {bias['grid_position']}")
    print(f"  Direction: {bias['direction_label']}")
    print(f"  Strategy:  {bias['strategy_label']}")


def print_strategy_table(strategies):
    """Print the strategy thresholds table."""
    print("\n" + "-" * 60)
    print("  STRATEGY THRESHOLDS")
    print("-" * 60)

    status_colors = {"ACTIVE": "+", "SELECTIVE": "~", "SKIP": "x"}

    for name, cfg in strategies.items():
        icon = status_colors.get(cfg["status"], "?")
        print(f"  [{icon}] {name:<12}  DPS >= {cfg['min_dps']}  ({cfg['status']})")
        print(f"      {cfg['note']}")

    print("-" * 60)
    print("  Legend: [+] ACTIVE  [~] SELECTIVE  [x] SKIP")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_bias_analysis():
    """Run the full 4-observation bias analysis."""
    now = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print(f"  ZCT DAILY TRADING BIAS — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    # ── Fetch all data ─────────────────────────────────────────────
    print("Fetching data...")
    t0 = time.time()

    changes = fetch_velo_changes(range_ms=14400000)  # 4h
    gainers = fetch_velo_priceline(range_ms=14400000, resolution="5 minutes", filter_type="Top Gainers")
    losers = fetch_velo_priceline(range_ms=14400000, resolution="5 minutes", filter_type="Top Losers")
    orion_tickers = fetch_orion_screener()

    print(f"  Fetched in {time.time()-t0:.1f}s: {len(changes)} coins, "
          f"{len(gainers)} gainers, {len(losers)} losers, {len(orion_tickers)} Orion tickers\n")

    # ── Observation 1: Return Buckets → Direction ──────────────────
    print("─" * 60)
    print("OBSERVATION 1: Velodata Return Buckets (Direction)")
    print("─" * 60)
    dir_score, dir_label = observation_1_direction(changes)
    print(f"  → {dir_label}\n")

    # ── Observation 2: Spaghetti Chart → Strategy Type ─────────────
    print("─" * 60)
    print("OBSERVATION 2: Velodata Spaghetti Chart (Strategy Type)")
    print("─" * 60)
    strat_score, strat_label = observation_2_strategy_type(gainers, losers, changes)
    print(f"  → {strat_label}\n")

    # ── Observation 3: Orion Tick Count & Volume ───────────────────
    print("─" * 60)
    print("OBSERVATION 3: Orion Tick Count & Volume (Activity Level)")
    print("─" * 60)
    activity_adj, activity_label = observation_3_activity(orion_tickers)
    print(f"  → {activity_label}\n")

    # ── Observation 4: Magnitude + Structure ───────────────────────
    print("─" * 60)
    print("OBSERVATION 4: Magnitude + Structure (Confirmation)")
    print("─" * 60)
    mag_adj, mag_label = observation_4_magnitude(changes, gainers, losers)
    print(f"  → {mag_label}\n")

    # ── Final Bias ─────────────────────────────────────────────────
    bias = compute_final_bias(dir_score, strat_score, activity_adj, mag_adj)
    print_bias_grid(bias)
    print_strategy_table(bias["strategies"])

    # Save result
    result = {
        "timestamp": now.isoformat(),
        "observations": {
            "1_direction": {"score": dir_score, "label": dir_label.split("\n")[0]},
            "2_strategy": {"score": strat_score, "label": strat_label.split("\n")[0]},
            "3_activity": {"adjustment": activity_adj, "label": activity_label.split("\n")[0]},
            "4_magnitude": {"adjustment": mag_adj, "label": mag_label.split("\n")[0]},
        },
        "bias": bias,
    }

    with open("daily_bias.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved to daily_bias.json\n")

    return result


if __name__ == "__main__":
    run_bias_analysis()
