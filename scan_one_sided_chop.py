#!/usr/bin/env python3
"""
One-Sided Chop Scanner

Pattern: Price moves into a choppy zone. The chop is messy but ONE boundary
(support or resistance) is consistently respected. Enter at that boundary.

Key differences from bouncy ball:
- Don't require both sides to be clean
- Only the entry side needs consistent touches
- The chop can be messy/varied in depth
- Pre-trend = how price entered the chop (up → long at support, down → short at resistance)
- TP from shallowest recent swing depth, capped at 1.25R
"""
import csv, os, sys
import numpy as np

DATA_DIRS = [
    'c:/Projects/CryptoAlgo/datasets/live/candles_1m',
]


def load_sym(fpath):
    with open(fpath, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 500:
        return None
    return {
        'c': np.array([float(r['close']) for r in rows]),
        'h': np.array([float(r['high']) for r in rows]),
        'l': np.array([float(r['low']) for r in rows]),
        'o': np.array([float(r['open']) for r in rows]),
        'v': np.array([float(r['volume']) for r in rows]),
        'ts': [r['timestamp'][:19] for r in rows],
    }


def detect_choppy_regime(closes, highs, lows, end_bar, min_bars=180, max_bars=480):
    """
    Detect if price has been choppy. Doesn't require a clean range —
    just lots of direction changes (oscillation).

    Returns (start_bar, is_choppy) or None.
    """
    for duration in [360, 480, 240, 180]:
        start = end_bar - duration
        if start < 0:
            continue

        c = closes[start:end_bar]
        if len(c) < 100:
            continue

        # Smooth and count direction changes
        smooth = np.convolve(c, np.ones(10) / 10, mode='valid')
        if len(smooth) < 30:
            continue
        diffs = np.diff(smooth)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        chop_ratio = sign_changes / len(diffs)

        # Also check: price crosses its own mean many times
        mean_price = np.mean(c)
        crosses = np.sum(np.diff(np.sign(c - mean_price)) != 0)

        if chop_ratio > 0.25 or crosses > 8:
            return start, duration

    return None


def find_respected_level(highs, lows, closes, opens, start, end, side):
    """
    Find a consistently respected level on ONE side using candle BODY prices.
    For longs (support): use max(open, close) of swing low candles — the body bottom
    For shorts (resistance): use min(open, close) of swing high candles — the body top

    Returns {'price', 'touches', 'avg_overshoot'} or None.
    """
    h = highs[start:end]
    l = lows[start:end]
    c = closes[start:end]
    o = opens[start:end]
    n = len(h)
    if n < 60:
        return None

    # Find swing points (local min/max with window 10)
    w = 10
    swing_bodies = []

    if side == "long":
        # Find swing lows — use the candle body (higher of open/close) at low points
        for j in range(w, n - w):
            if l[j] <= np.min(l[max(0, j - w):j]) and l[j] <= np.min(l[j + 1:j + w + 1]):
                body_low = min(o[j], c[j])  # bottom of candle body
                swing_bodies.append(body_low)
    else:
        # Find swing highs — use the candle body (lower of open/close) at high points
        for j in range(w, n - w):
            if h[j] >= np.max(h[max(0, j - w):j]) and h[j] >= np.max(h[j + 1:j + w + 1]):
                body_high = max(o[j], c[j])  # top of candle body
                swing_bodies.append(body_high)

    if len(swing_bodies) < 3:
        return None

    # Cluster swing body prices (within 0.2%)
    sorted_prices = np.sort(swing_bodies)
    candidates = []
    used = set()

    for idx, p in enumerate(sorted_prices):
        if idx in used:
            continue
        cluster = [p]
        used.add(idx)
        for idx2, p2 in enumerate(sorted_prices):
            if idx2 in used:
                continue
            if abs(p2 - p) / p * 100 < 0.2:
                cluster.append(p2)
                used.add(idx2)

        if len(cluster) >= 3:
            level = np.mean(cluster)

            # Count all bars that touch this level (body or wick within 0.2%)
            touches = 0
            last_t = -10
            for j in range(n):
                if j - last_t < 10:
                    continue
                body = max(o[j], c[j]) if side == "long" else min(o[j], c[j])
                # Check candle body near level (within 0.2%)
                body_near = abs(body - level) / level * 100 < 0.2
                # Also count close near level
                close_near = abs(c[j] - level) / level * 100 < 0.2
                if body_near or close_near:
                    touches += 1
                    last_t = j

            if touches >= 3:
                candidates.append({'price': level, 'touches': touches, 'n_swings': len(cluster)})

    if not candidates:
        return None

    candidates.sort(key=lambda x: x['touches'], reverse=True)
    return candidates[0]


def find_real_swings(highs, lows, closes, opens, level, start, end, side, min_travel_pct=0.5):
    """
    Find real swing touches using candle body proximity (0.2%).
    min_travel is based on absolute % from level.
    """
    min_travel = level * min_travel_pct / 100
    swings = []
    near = False
    farthest = 0

    for j in range(start, end):
        body_top = max(opens[j], closes[j])
        body_bot = min(opens[j], closes[j])

        # Near level = candle body within 0.2% of level
        if side == "long":
            at_level = abs(body_bot - level) / level * 100 < 0.2 or abs(closes[j] - level) / level * 100 < 0.2
            dist = highs[j] - level  # how far UP from support
        else:
            at_level = abs(body_top - level) / level * 100 < 0.2 or abs(closes[j] - level) / level * 100 < 0.2
            dist = level - lows[j]  # how far DOWN from resistance

        if at_level:
            if not near and farthest >= min_travel:
                swings.append({'bar': j, 'depth': farthest})
            near = True
            farthest = 0
        else:
            near = False
            if dist > farthest:
                farthest = dist

    swings.reverse()
    return swings


def detect_pre_trend(closes, highs, lows, chop_start, lookback=120):
    """How did price enter the chop? Compare price before vs at chop start."""
    pre_start = max(0, chop_start - lookback)
    if chop_start - pre_start < 30:
        return "flat", 0.0

    # Check wider window for peak/trough
    wide_start = max(0, chop_start - 240)
    peak = np.max(highs[wide_start:chop_start + 60])
    trough = np.min(lows[wide_start:chop_start + 60])
    current = np.mean(closes[chop_start:min(chop_start + 60, len(closes))])

    drop = (peak - current) / peak * 100
    rise = (current - trough) / trough * 100

    if drop > 1.0 and drop > rise:
        return "down", -drop
    elif rise > 1.0 and rise > drop:
        return "up", rise

    # Fallback: simple close-to-close
    move = (closes[chop_start] - closes[pre_start]) / closes[pre_start] * 100
    if move > 0.5:
        return "up", move
    elif move < -0.5:
        return "down", move
    return "flat", move


def score_volume(volumes, bar, side, lookback=60, skip=3):
    """ZCT volume scoring. MR Long: flat=2,dec=1,inc=0. MR Short: dec=2,flat=1,inc=0."""
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


def sim_outcome(h, l, entry, sl, tp, side, start, n):
    fb = min(120, n - start - 1)
    if fb < 10:
        return "OPEN"
    rd = abs(entry - sl)
    t09 = entry - rd * 0.9 if side == "short" else entry + rd * 0.9
    tsl = entry + rd * 0.1 if side == "short" else entry - rd * 0.1
    trailed = False
    csl = sl
    for k in range(start + 1, start + fb + 1):
        if not trailed:
            if side == "short" and l[k] <= t09:
                trailed = True
                csl = tsl
            elif side == "long" and h[k] >= t09:
                trailed = True
                csl = tsl
        if side == "short":
            if l[k] <= tp:
                return "TP"
            if h[k] >= csl:
                return "TRAIL_SL" if trailed else "SL"
        else:
            if h[k] >= tp:
                return "TP"
            if l[k] <= csl:
                return "TRAIL_SL" if trailed else "SL"
    return "OPEN"


# --- Main scan ---
files = []
for d in DATA_DIRS:
    if os.path.exists(d):
        for f in sorted(os.listdir(d)):
            if f.endswith('.csv'):
                files.append((d, f))

results = []
seen = set()

for dd, fn in files:
    sym = fn.replace('_1m_7d.csv', '').replace('_1m.csv', '')
    if sym in seen:
        continue
    seen.add(sym)
    try:
        data = load_sym(os.path.join(dd, fn))
    except Exception:
        continue
    if data is None:
        continue

    c, h, l, o, v, ts = data['c'], data['h'], data['l'], data['o'], data['v'], data['ts']
    n = len(c)
    last_s = -60

    for i in range(360, n - 30, 10):
        if i - last_s < 60:
            continue

        # Step 1: Is it choppy?
        chop = detect_choppy_regime(c, h, l, i)
        if chop is None:
            continue
        chop_start, chop_dur = chop

        # Step 2: Pre-trend — how did price enter the chop?
        pre_trend, pre_pct = detect_pre_trend(c, h, l, chop_start)

        # Step 3: Based on pre-trend, look for respected level on ONE side
        if pre_trend == "up":
            side = "long"
        elif pre_trend == "down":
            side = "short"
        else:
            continue  # flat = skip, no directional bias

        level_info = find_respected_level(h, l, c, o, chop_start, i, side)
        if level_info is None:
            continue

        level = level_info['price']

        # Step 4: Is price at the level NOW? (within 0.2%)
        # For shorts: candle body top near level, or wick up to 0.2% above
        # For longs: candle body bottom near level, or wick up to 0.2% below
        body_top = max(o[i], c[i])
        body_bot = min(o[i], c[i])

        if side == "long":
            # Price should be near or just below the support level
            if body_bot > level * 1.002:
                continue  # too far above
            if l[i] > level * 1.002:
                continue  # wick didn't even get close
        else:
            # Price should be near or just above the resistance level
            if body_top < level * 0.998:
                continue  # too far below
            if h[i] < level * 0.998:
                continue  # wick didn't even get close

        # Step 5: Confirmation — close back on correct side
        if side == "long" and c[i] <= level * 0.998:
            continue
        if side == "short" and c[i] >= level * 1.002:
            continue

        # Step 6: Find real swing touches + last swing within 30 min
        swings = find_real_swings(h, l, c, o, level, chop_start, i, side)
        if len(swings) < 1:
            continue

        last_swing_min = i - swings[0]['bar']
        if last_swing_min > 30:
            continue

        # Step 7: TP from swing depth
        recent = swings[:3]
        depths = [s['depth'] for s in recent]
        tp_depth = min(depths) * 0.95

        entry = level
        sl_buffer = 0.01  # 1.0%
        if side == "long":
            sl = level * (1 - sl_buffer)
            tp = entry + tp_depth
        else:
            sl = level * (1 + sl_buffer)
            tp = entry - tp_depth

        sl_dist = abs(entry - sl)
        sl_pct = sl_dist / entry * 100
        tp_pct = tp_depth * 0.95 / entry * 100

        if sl_pct <= 0 or tp_pct < 0.5:
            continue

        rr = tp_pct / sl_pct

        # Cap RR
        if rr > 1.5:
            capped_dist = sl_dist * 1.25
            if side == "long":
                tp = entry + capped_dist
            else:
                tp = entry - capped_dist
            tp_pct = capped_dist / entry * 100
            rr = 1.25

        if rr < 1.1:
            continue

        # Volume scoring
        vol_score, vol_type = score_volume(v, i, side)

        # DPS: duration + approach + volume
        dur_score = 2 if chop_dur >= 240 else (1 if chop_dur >= 120 else 0)
        app_score = 2 if last_swing_min <= 5 else (1 if last_swing_min <= 15 else 0)
        dps = dur_score + app_score + vol_score

        # Outcome
        out = sim_outcome(h, l, entry, sl, tp, side, i, n)

        chop_hrs = chop_dur / 60
        depths_str = "/".join(f"{d / entry * 100:.1f}%" for d in depths)

        results.append({
            'sym': sym, 'side': side, 'ts': ts[i],
            'entry': round(entry, 8), 'tp': round(tp, 8), 'sl': round(sl, 8),
            'tp_p': round(tp_pct, 2), 'rr': round(rr, 2),
            'sw': len(swings), 'last_min': last_swing_min,
            'touches': level_info['touches'],
            'depths': depths_str,
            'dps': dps, 'dur_s': dur_score, 'app_s': app_score, 'vol_s': vol_score,
            'vol_type': vol_type,
            'pre': pre_trend, 'pre_pct': round(pre_pct, 2),
            'chop_hrs': round(chop_hrs, 1),
            'out': out,
        })
        last_s = i

# Results
tp_c = sum(1 for r in results if r['out'] == 'TP')
sl_c = sum(1 for r in results if r['out'] in ('SL', 'TRAIL_SL'))
tr_c = sum(1 for r in results if r['out'] == 'TRAIL_SL')
op_c = sum(1 for r in results if r['out'] == 'OPEN')
print(f"Total: {len(results)}  TP: {tp_c}  SL: {sl_c - tr_c}  TRAIL: {tr_c}  OPEN: {op_c}")
if tp_c + sl_c > 0:
    print(f"WR: {tp_c}/{tp_c + sl_c} = {tp_c / (tp_c + sl_c) * 100:.1f}%")

print("\nBy DPS:")
for d in sorted(set(r['dps'] for r in results)):
    sub = [r for r in results if r['dps'] == d]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL', 'TRAIL_SL'))
    if t + s > 0:
        print(f"  DPS={d}: {len(sub)}, TP={t} SL={s} WR={t / (t + s) * 100:.1f}%")

print("\nBy Side:")
for sd in ['long', 'short']:
    sub = [r for r in results if r['side'] == sd]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL', 'TRAIL_SL'))
    if t + s > 0:
        print(f"  {sd}: {len(sub)}, TP={t} SL={s} WR={t / (t + s) * 100:.1f}%")

print("\nBy RR:")
for a, b, lb in [(1, 1.25, '1-1.25'), (1.25, 1.26, '1.25 (capped)'), (1.26, 1.5, '1.26-1.5')]:
    sub = [r for r in results if a <= r['rr'] <= b]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL', 'TRAIL_SL'))
    if sub and t + s > 0:
        print(f"  RR {lb}: {len(sub)}, TP={t} SL={s} WR={t / (t + s) * 100:.1f}%")

print("\nBy Swings:")
for ns in [1, 2, 3]:
    lb = f"{ns}" if ns < 3 else "3+"
    sub = [r for r in results if (r['sw'] == ns if ns < 3 else r['sw'] >= ns)]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL', 'TRAIL_SL'))
    if sub and t + s > 0:
        print(f"  {lb} sw: {len(sub)}, TP={t} SL={s} WR={t / (t + s) * 100:.1f}%")

print()
results.sort(key=lambda x: (x['dps'], x['sw'], x['touches']), reverse=True)

print(f"{'Symbol':>14} {'Side':>5} {'DPS':>3} {'D':>1}{'A':>1}{'V':>1} {'Vol':>4} {'Pre':>4} "
      f"{'Entry':>10} {'TP':>10} {'SL':>10} {'TP%':>5} {'RR':>4} "
      f"{'Sw':>2} {'Tch':>3} {'Lst':>3} {'Depths':>15} {'Chp':>4} {'Out':>6} {'Time'}")
print("-" * 140)
for r in results[:50]:
    print(f"{r['sym']:>14} {r['side']:>5} {r['dps']:>3} {r['dur_s']}{r['app_s']}{r['vol_s']} "
          f"{r['vol_type'][:4]:>4} {r['pre']:>4} "
          f"{r['entry']:>10.6g} {r['tp']:>10.6g} {r['sl']:>10.6g} {r['tp_p']:>5.2f} {r['rr']:>4.1f} "
          f"{r['sw']:>2} {r['touches']:>3} {r['last_min']:>2}m "
          f"{r['depths']:>15} {r['chop_hrs']:>3.0f}h {r['out']:>6} {r['ts']}")
