#!/usr/bin/env python3
"""
ZCT Mean Reversion Scanner v2

Finds setups matching the ZCT MR criteria:
1. Fast spike into a PROVEN level (swing highs/lows + candle close clusters)
2. Volume flat or decreasing BEFORE the spike
3. Choppy/sideways price action on the left (4-8 hours)

Entry: confirmation candle closes back on correct side of level
SL: beyond spike extreme
TP: opposite range boundary
"""

import csv, os, sys
import numpy as np

# Data directories
DATA_DIRS = [
    'c:/Projects/CryptoAlgo/datasets/momo_1m_7d_top100_midcap_30d',
    'c:/Projects/CryptoAlgo/datasets/live/candles_1m',
]


def load_symbol(fpath):
    """Load OHLCV data from CSV."""
    with open(fpath, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 500:
        return None
    return {
        'closes': np.array([float(r['close']) for r in rows]),
        'highs': np.array([float(r['high']) for r in rows]),
        'lows': np.array([float(r['low']) for r in rows]),
        'volumes': np.array([float(r['volume']) for r in rows]),
        'timestamps': [r['timestamp'][:19] for r in rows],
    }


def compute_atr(highs, lows, closes, period=30):
    """Compute ATR array."""
    n = len(closes)
    atr = np.zeros(n)
    for i in range(period + 1, n):
        tr = np.maximum(
            highs[i - period:i] - lows[i - period:i],
            np.maximum(
                np.abs(highs[i - period:i] - closes[i - period - 1:i - 1]),
                np.abs(lows[i - period:i] - closes[i - period - 1:i - 1])
            )
        )
        atr[i] = np.mean(tr)
    return atr


def find_levels(highs, lows, closes, tol_pct=0.3):
    """
    Find proven price levels from swing highs/lows AND candle close clusters.
    Returns list of {'price', 'swing_touches', 'close_touches', 'total'}.
    """
    n = len(closes)
    w = 10  # swing window

    # Collect all candidate prices: swing highs, swing lows, and closes
    level_candidates = []

    # Swing highs
    for j in range(w, n - w):
        if highs[j] >= np.max(highs[max(0, j - w):j]) and highs[j] >= np.max(highs[j + 1:j + w + 1]):
            level_candidates.append(('swing_high', highs[j]))

    # Swing lows
    for j in range(w, n - w):
        if lows[j] <= np.min(lows[max(0, j - w):j]) and lows[j] <= np.min(lows[j + 1:j + w + 1]):
            level_candidates.append(('swing_low', lows[j]))

    if not level_candidates:
        return [], []

    # Cluster all swing prices
    all_prices = sorted(set(p for _, p in level_candidates))

    clusters = []
    used = set()
    for i, p in enumerate(all_prices):
        if i in used:
            continue
        cluster_prices = [p]
        used.add(i)
        for j, p2 in enumerate(all_prices):
            if j in used:
                continue
            if abs(p2 - p) / p * 100 < tol_pct:
                cluster_prices.append(p2)
                used.add(j)

        level_price = np.mean(cluster_prices)

        # Count swing touches at this level
        swing_touches = 0
        for typ, sp in level_candidates:
            if abs(sp - level_price) / level_price * 100 < tol_pct:
                swing_touches += 1

        # Count candle close touches (within tol, min 5 bars apart)
        close_touches = 0
        last_ct = -5
        for j in range(len(closes)):
            if abs(closes[j] - level_price) / level_price * 100 < tol_pct and j - last_ct >= 5:
                close_touches += 1
                last_ct = j

        # Count wick touches (high or low near level, not just swing)
        wick_touches = 0
        last_wt = -5
        for j in range(len(highs)):
            touched = (abs(highs[j] - level_price) / level_price * 100 < tol_pct or
                       abs(lows[j] - level_price) / level_price * 100 < tol_pct)
            if touched and j - last_wt >= 5:
                wick_touches += 1
                last_wt = j

        total = swing_touches + close_touches
        clusters.append({
            'price': level_price,
            'swing_touches': swing_touches,
            'close_touches': close_touches,
            'wick_touches': wick_touches,
            'total': total,
        })

    # Separate into resistance (above median) and support (below median)
    if not clusters:
        return [], []

    median_price = np.median(closes)
    res_levels = sorted([c for c in clusters if c['price'] > median_price and c['total'] >= 2],
                        key=lambda x: x['total'], reverse=True)
    sup_levels = sorted([c for c in clusters if c['price'] <= median_price and c['total'] >= 2],
                        key=lambda x: x['total'], reverse=True)

    return res_levels, sup_levels


def detect_spike(closes, highs, lows, atr, i, max_bars=5):
    """Detect fast spike at bar i. Returns (direction, size_pct, n_bars) or None."""
    for nb in [1, 2, 3, max_bars]:
        if i - nb < 0:
            continue
        move = closes[i] - closes[i - nb]
        move_pct = abs(move) / closes[i - nb] * 100
        atr_mult = abs(move) / atr[i] if atr[i] > 0 else 0

        # Steeper = fewer bars needed for same threshold
        if nb <= 2:
            min_atr = 1.8
            min_pct = 0.8
        else:
            min_atr = 2.5
            min_pct = 1.2

        if atr_mult >= min_atr and move_pct >= min_pct:
            direction = "up" if move > 0 else "down"
            return direction, move_pct, nb

    return None


def check_volume_flat(volumes, i, lookback=60, spike_bars=3):
    """Check if volume was flat/decreasing BEFORE the spike."""
    start = max(0, i - lookback)
    end = i - spike_bars  # exclude spike bars
    vol_w = volumes[start:end]
    if len(vol_w) < 20:
        return False, 0

    x = np.arange(len(vol_w))
    slope = np.polyfit(x, vol_w, 1)[0]
    avg_vol = np.mean(vol_w)
    if avg_vol <= 0:
        return False, 0

    norm_slope = slope / avg_vol
    # Flat = slope near 0, decreasing = negative
    return norm_slope <= 0.01, norm_slope


def check_choppy(highs, lows, closes, start, end):
    """
    Check if price action is choppy/sideways.
    Returns (is_choppy, range_pct, details).
    """
    h = highs[start:end]
    l = lows[start:end]
    c = closes[start:end]
    if len(h) < 100:
        return False, 0, {}

    range_total = np.max(h) - np.min(l)
    range_pct = range_total / np.mean(c) * 100

    if range_pct < 0.8 or range_pct > 8.0:
        return False, range_pct, {}

    # Choppiness: measure how much price oscillates vs trends
    # Method: count direction changes in smoothed close
    smooth = np.convolve(c, np.ones(10) / 10, mode='valid')
    if len(smooth) < 20:
        return False, range_pct, {}

    diffs = np.diff(smooth)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    chop_ratio = sign_changes / len(diffs)

    # Also check: how many times does price cross the midline?
    mid = (np.max(h) + np.min(l)) / 2
    crosses = np.sum(np.diff(np.sign(c - mid)) != 0)

    is_choppy = chop_ratio > 0.3 or crosses > 8

    return is_choppy, range_pct, {
        'chop_ratio': round(chop_ratio, 3),
        'midline_crosses': crosses,
        'sign_changes': sign_changes,
    }


def scan_symbol(sym, data):
    """Scan a single symbol for MR setups."""
    closes = data['closes']
    highs = data['highs']
    lows = data['lows']
    volumes = data['volumes']
    timestamps = data['timestamps']
    n = len(closes)

    atr = compute_atr(highs, lows, closes)
    setups = []
    last_setup = -30  # min bars between setups

    for i in range(360, n - 30):
        if i - last_setup < 30:
            continue
        if atr[i] <= 0:
            continue

        # --- Step 1: Detect spike ---
        spike = detect_spike(closes, highs, lows, atr, i)
        if spike is None:
            continue
        direction, spike_pct, spike_bars = spike

        # --- Step 2: Find levels in lookback window ---
        lb_start = max(0, i - 480)  # 8 hours
        lb_end = i - spike_bars  # before spike

        chunk_h = highs[lb_start:lb_end]
        chunk_l = lows[lb_start:lb_end]
        chunk_c = closes[lb_start:lb_end]

        if len(chunk_h) < 100:
            continue

        res_levels, sup_levels = find_levels(chunk_h, chunk_l, chunk_c)

        # --- Step 3: Did spike hit a proven level? ---
        level_hit = None
        side = None

        if direction == "up":  # spike up → MR short at resistance
            for lv in res_levels:
                # Spike high should reach the level
                dist = abs(highs[i] - lv['price']) / lv['price'] * 100
                if dist < 0.5 or highs[i] >= lv['price']:
                    if lv['total'] >= 3:
                        level_hit = lv
                        side = "short"
                        break
        else:  # spike down → MR long at support
            for lv in sup_levels:
                dist = abs(lows[i] - lv['price']) / lv['price'] * 100
                if dist < 0.5 or lows[i] <= lv['price']:
                    if lv['total'] >= 3:
                        level_hit = lv
                        side = "long"
                        break

        if level_hit is None:
            continue

        # --- Step 4: Volume check ---
        vol_ok, vol_slope = check_volume_flat(volumes, i, spike_bars=spike_bars)

        # --- Step 5: Choppiness check ---
        chop_start = max(0, i - 360)
        is_choppy, range_pct, chop_details = check_choppy(highs, lows, closes, chop_start, i - spike_bars)

        # --- DPS Scoring ---
        # Approach: spike quality
        if spike_bars <= 2 and spike_pct >= 1.5:
            approach_score = 2  # clear fast spike
        elif spike_pct >= 1.0:
            approach_score = 1  # decent spike
        else:
            approach_score = 0

        # Volume
        if vol_ok and vol_slope <= -0.005:
            vol_score = 2  # clearly decreasing
        elif vol_ok:
            vol_score = 2  # flat
        else:
            vol_score = 1  # unclear but not terrible

        # Choppy/range
        if is_choppy and range_pct >= 1.0:
            chop_score = 2
        elif range_pct >= 0.8:
            chop_score = 1
        else:
            chop_score = 0

        # Level strength
        if level_hit['total'] >= 8:
            level_score = 2
        elif level_hit['total'] >= 4:
            level_score = 1
        else:
            level_score = 0

        dps_total = approach_score + vol_score + chop_score + level_score

        # Need at least 5/8 to be worth taking
        if dps_total < 4:
            continue

        # --- Step 6: Confirmation — next candle ---
        cb = i + 1
        if cb >= n:
            continue

        if side == "short" and closes[cb] >= level_hit['price'] * 1.002:
            continue  # didn't confirm
        if side == "long" and closes[cb] <= level_hit['price'] * 0.998:
            continue

        # --- Entry, TP, SL ---
        entry = closes[cb]
        entry_ts = timestamps[cb]

        if side == "short":
            sl = max(highs[i], level_hit['price']) * 1.005
            # TP at best support level
            if sup_levels:
                tp = sup_levels[0]['price'] * 1.003
            else:
                tp = entry * (1 - spike_pct / 100 * 0.5)
            sl_pct = (sl - entry) / entry * 100
            tp_pct = (entry - tp) / entry * 100
        else:
            sl = min(lows[i], level_hit['price']) * 0.995
            if res_levels:
                tp = res_levels[0]['price'] * 0.997
            else:
                tp = entry * (1 + spike_pct / 100 * 0.5)
            sl_pct = (entry - sl) / entry * 100
            tp_pct = (tp - entry) / entry * 100

        if sl_pct <= 0 or tp_pct <= 0:
            continue
        rr = tp_pct / sl_pct
        if rr < 0.8:
            continue

        # --- Outcome (2 hour window) ---
        fb = min(120, n - cb - 1)
        if fb < 10:
            continue

        outcome = "OPEN"
        for k in range(cb + 1, cb + fb + 1):
            if side == "short":
                if lows[k] <= tp:
                    outcome = "TP"
                    break
                if highs[k] >= sl:
                    outcome = "SL"
                    break
            else:
                if highs[k] >= tp:
                    outcome = "TP"
                    break
                if lows[k] <= sl:
                    outcome = "SL"
                    break

        setups.append({
            'symbol': sym,
            'side': side,
            'spike_ts': timestamps[i],
            'entry_ts': entry_ts,
            'level': round(level_hit['price'], 8),
            'swing_t': level_hit['swing_touches'],
            'close_t': level_hit['close_touches'],
            'total_t': level_hit['total'],
            'entry': round(entry, 8),
            'tp': round(tp, 8),
            'sl': round(sl, 8),
            'sl_pct': round(sl_pct, 2),
            'tp_pct': round(tp_pct, 2),
            'rr': round(rr, 2),
            'spike_pct': round(spike_pct, 2),
            'spike_bars': spike_bars,
            'range_pct': round(range_pct, 2),
            'dps': dps_total,
            'approach': approach_score,
            'vol': vol_score,
            'chop': chop_score,
            'lvl_str': level_score,
            'outcome': outcome,
        })
        last_setup = i

    return setups


def main():
    # Collect all files
    files = []
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            continue
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.csv'):
                files.append((data_dir, f))

    all_setups = []
    seen_syms = set()

    for data_dir, fname in files:
        sym = fname.replace('_1m_7d.csv', '').replace('_1m.csv', '')
        if sym in seen_syms:
            continue
        seen_syms.add(sym)

        fpath = os.path.join(data_dir, fname)
        try:
            data = load_symbol(fpath)
        except Exception:
            continue
        if data is None:
            continue

        sym_setups = scan_symbol(sym, data)
        all_setups.extend(sym_setups)
        if sym_setups:
            print(f"  {sym}: {len(sym_setups)} setups", file=sys.stderr)

    # Results
    tp_c = sum(1 for s in all_setups if s['outcome'] == 'TP')
    sl_c = sum(1 for s in all_setups if s['outcome'] == 'SL')
    op_c = sum(1 for s in all_setups if s['outcome'] == 'OPEN')
    print(f"\nTotal: {len(all_setups)} setups  TP: {tp_c}  SL: {sl_c}  OPEN: {op_c}")
    if tp_c + sl_c > 0:
        print(f"WR: {tp_c}/{tp_c + sl_c} = {tp_c / (tp_c + sl_c) * 100:.1f}%")

    # By DPS score
    print("\nBy DPS Score:")
    for dps in sorted(set(s['dps'] for s in all_setups)):
        sub = [s for s in all_setups if s['dps'] == dps]
        tp_s = sum(1 for s in sub if s['outcome'] == 'TP')
        sl_s = sum(1 for s in sub if s['outcome'] == 'SL')
        wr = tp_s / (tp_s + sl_s) * 100 if tp_s + sl_s > 0 else 0
        print(f"  DPS={dps}: {len(sub)} setups, TP={tp_s} SL={sl_s} WR={wr:.1f}%")

    print()
    all_setups.sort(key=lambda x: (x['dps'], x['total_t'], x['spike_pct']), reverse=True)

    # Header
    hdr = (f"{'Symbol':>14} {'Side':>5} {'DPS':>3} {'A':>1}{'V':>1}{'C':>1}{'L':>1} "
           f"{'Spk%':>5} {'SwT':>3} {'ClT':>3} {'Tot':>3} "
           f"{'Level':>10} {'Entry':>10} {'TP':>10} {'SL':>10} "
           f"{'SL%':>5} {'TP%':>5} {'RR':>4} {'Rng%':>5} {'Out':>4} {'Entry Time':>19}")
    print(hdr)
    print("-" * len(hdr))
    for s in all_setups[:50]:
        print(f"{s['symbol']:>14} {s['side']:>5} {s['dps']:>3} "
              f"{s['approach']}{s['vol']}{s['chop']}{s['lvl_str']} "
              f"{s['spike_pct']:>4.1f}% {s['swing_t']:>3} {s['close_t']:>3} {s['total_t']:>3} "
              f"{s['level']:>10.6g} {s['entry']:>10.6g} {s['tp']:>10.6g} {s['sl']:>10.6g} "
              f"{s['sl_pct']:>5.2f} {s['tp_pct']:>5.2f} {s['rr']:>4.1f} {s['range_pct']:>5.2f} {s['outcome']:>4} {s['entry_ts']:>19}")


if __name__ == "__main__":
    main()
