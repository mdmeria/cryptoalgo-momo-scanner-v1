#!/usr/bin/env python3
"""
Range Shift v2 — Corrected swing logic.
Swings are any touch-leave-return cycle at the level.
TP = shallowest depth of last 3 swings (not a filter).
Level from candle bodies, not percentile wicks.
"""
import csv, os
import numpy as np

DATA_DIRS = [
    'c:/Projects/CryptoAlgo/datasets/momo_1m_7d_top100_midcap_30d',
    'c:/Projects/CryptoAlgo/datasets/live/candles_1m',
]


def load(fpath):
    with open(fpath, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 600:
        return None
    return {
        'o': np.array([float(r['open']) for r in rows]),
        'h': np.array([float(r['high']) for r in rows]),
        'l': np.array([float(r['low']) for r in rows]),
        'c': np.array([float(r['close']) for r in rows]),
        'v': np.array([float(r['volume']) for r in rows]),
        'ts': [r['timestamp'][:19] for r in rows],
    }


def find_tf_levels(highs, lows, closes, opens, end_bar):
    """
    Find levels from 15-min and 1-hour candle highs/lows.
    Returns list of candidate levels with source info.
    """
    levels = []

    # 15-min high/low (last 15 bars)
    s15 = max(0, end_bar - 15)
    h15 = np.max(highs[s15:end_bar])
    l15 = np.min(lows[s15:end_bar])
    # Use body for the level
    h15_bodies = [max(opens[j], closes[j]) for j in range(s15, end_bar) if highs[j] >= h15 * 0.999]
    l15_bodies = [min(opens[j], closes[j]) for j in range(s15, end_bar) if lows[j] <= l15 * 1.001]
    if h15_bodies:
        levels.append({'price': np.mean(h15_bodies), 'source': '15m_high', 'side': 'short'})
    if l15_bodies:
        levels.append({'price': np.mean(l15_bodies), 'source': '15m_low', 'side': 'long'})

    # 1-hour high/low (last 60 bars)
    s60 = max(0, end_bar - 60)
    h60 = np.max(highs[s60:end_bar])
    l60 = np.min(lows[s60:end_bar])
    h60_bodies = [max(opens[j], closes[j]) for j in range(s60, end_bar) if highs[j] >= h60 * 0.999]
    l60_bodies = [min(opens[j], closes[j]) for j in range(s60, end_bar) if lows[j] <= l60 * 1.001]
    if h60_bodies:
        levels.append({'price': np.mean(h60_bodies), 'source': '1h_high', 'side': 'short'})
    if l60_bodies:
        levels.append({'price': np.mean(l60_bodies), 'source': '1h_low', 'side': 'long'})

    return levels


def find_respected_swings(highs, lows, closes, opens, level, start, end, side, max_breach_bars=5):
    """
    Find swings where price RESPECTED the level.

    For longs (support):
      - Price swings down toward level
      - Max 3 consecutive candle closes below level
      - Then price recovers back above level
      = one respected swing

    For shorts (resistance):
      - Price swings up toward level
      - Max 3 consecutive candle closes above level
      - Then price drops back below level
      = one respected swing

    Returns list of {bar, depth_pct, touch_type}.
    bar = the bar where price returned to correct side after the touch.
    depth_pct = how far price traveled away from level before this swing.
    touch_type = 'body' (close at/beyond level) or 'wick' (wick touched, close on correct side)
    """
    swings = []
    farthest = 0
    breach_count = 0  # consecutive closes beyond level
    in_swing = False  # price has traveled away from level
    swing_start = start

    for j in range(start, end):
        c_val = closes[j]
        h_val = highs[j]
        l_val = lows[j]
        body_top = max(opens[j], c_val)
        body_bot = min(opens[j], c_val)

        if side == "long":
            # Track how far price went above level (away from support)
            dist_away = h_val - level
            if dist_away > farthest:
                farthest = dist_away

            # Is price at or below the level?
            close_below = c_val < level
            wick_touched = l_val <= level  # wick reached down to level
            close_at_level = abs(c_val - level) / level * 100 < 0.1  # close very near level

            if close_below:
                breach_count += 1
                if breach_count > max_breach_bars:
                    # Level broken — reset
                    farthest = 0
                    breach_count = 0
                    in_swing = False
            elif wick_touched or close_at_level:
                # Price touched level and closed above — potential respected swing
                if farthest > 0:
                    depth_pct = farthest / level * 100
                    if close_at_level or close_below is False:
                        touch = 'wick' if (not close_at_level and wick_touched) else 'body'
                        swings.append({
                            'bar': j,
                            'depth_pct': depth_pct,
                            'depth': farthest,
                            'touch_type': touch,
                        })
                # Reset for next swing
                farthest = 0
                breach_count = 0
            else:
                # Price above level, not touching — reset breach count
                breach_count = 0

        else:  # short — resistance
            # Track how far price went below level (away from resistance)
            dist_away = level - l_val
            if dist_away > farthest:
                farthest = dist_away

            # Is price at or above the level?
            close_above = c_val > level
            wick_touched = h_val >= level  # wick reached up to level
            close_at_level = abs(c_val - level) / level * 100 < 0.1

            if close_above:
                breach_count += 1
                if breach_count > max_breach_bars:
                    # Level broken — reset
                    farthest = 0
                    breach_count = 0
                    in_swing = False
            elif wick_touched or close_at_level:
                # Price touched level and closed below — potential respected swing
                if farthest > 0:
                    depth_pct = farthest / level * 100
                    if close_at_level or close_above is False:
                        touch = 'wick' if (not close_at_level and wick_touched) else 'body'
                        swings.append({
                            'bar': j,
                            'depth_pct': depth_pct,
                            'depth': farthest,
                            'touch_type': touch,
                        })
                # Reset for next swing
                farthest = 0
                breach_count = 0
            else:
                # Price below level, not touching — reset breach count
                breach_count = 0

    return swings


def detect_pre_trend(closes, highs, lows, chop_start, lookback=240):
    """How did price enter the chop zone?"""
    wide_start = max(0, chop_start - lookback)
    wide_end = min(len(closes), chop_start + 60)
    if wide_end - wide_start < 30:
        return "flat", 0
    peak = np.max(highs[wide_start:wide_end])
    trough = np.min(lows[wide_start:wide_end])
    current = np.mean(closes[chop_start:min(chop_start + 60, len(closes))])
    drop = (peak - current) / peak * 100
    rise = (current - trough) / trough * 100
    if drop > 1.0 and drop > rise:
        return "down", -drop
    if rise > 1.0 and rise > drop:
        return "up", rise
    return "flat", 0


def score_volume(volumes, bar, side, lookback=60, skip=3):
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


def detect_choppy(closes, end):
    """Detect if recent price action is choppy."""
    for dur in [360, 180]:
        s = end - dur
        if s < 0:
            continue
        cc = closes[s:end]
        if len(cc) < 100:
            continue
        smooth = np.convolve(cc, np.ones(10) / 10, mode='valid')
        if len(smooth) < 30:
            continue
        diffs = np.diff(smooth)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        ratio = sign_changes / len(diffs)
        if ratio > 0.25:
            return s, dur
    return None


# Main scan
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
        data = load(os.path.join(dd, fn))
    except Exception:
        continue
    if data is None:
        continue

    o, h, l, c, v, ts = data['o'], data['h'], data['l'], data['c'], data['v'], data['ts']
    n = len(c)
    last_s = -60

    for i in range(600, n - 10):
        if i - last_s < 60:
            continue

        chop = detect_choppy(c, i)
        if chop is None:
            continue
        chop_start, chop_dur = chop

        pre, pre_pct = detect_pre_trend(c, h, l, chop_start)

        # Get candidate levels from 15m and 1h timeframes
        tf_levels = find_tf_levels(h, l, c, o, i)
        if not tf_levels:
            continue

        # Filter by pre-trend direction
        if pre == "up":
            candidates = [lv for lv in tf_levels if lv['side'] == 'long']
        elif pre == "down":
            candidates = [lv for lv in tf_levels if lv['side'] == 'short']
        else:
            # Flat pre-trend: try both sides
            candidates = tf_levels

        found_setup = False
        for cand in candidates:
            level = cand['price']
            side = cand['side']
            source = cand['source']

            # Find respected swings at this level
            swings = find_respected_swings(h, l, c, o, level, chop_start, i, side)

            # Need at least 3 respected swings
            if len(swings) < 3:
                continue

            # Last swing must be within 30 minutes
            last_sw_mins = i - swings[-1]['bar']
            if last_sw_mins > 30:
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

            # TP = shallowest depth of last 3 swings
            recent = swings[-3:]
            depths = [s['depth_pct'] for s in recent]
            tp_depth_pct = min(depths) * 0.95

            entry = level
            sl_pct = 1.0
            if side == "short":
                sl_v = level * 1.01
                tp_v = level * (1 - tp_depth_pct / 100)
            else:
                sl_v = level * 0.99
                tp_v = level * (1 + tp_depth_pct / 100)

            tp_p = tp_depth_pct * 0.95
            rr = tp_p / sl_pct if sl_pct > 0 else 0

            if rr > 1.5:
                rr = 1.25
                cap_dist = abs(entry - sl_v) * 1.25
                tp_v = entry - cap_dist if side == "short" else entry + cap_dist
                tp_p = cap_dist / entry * 100

            if rr < 0.8 or tp_p < 0.5:
                continue

            # DPS
            dur_score = 2 if chop_dur / 60 >= 4 else (1 if chop_dur / 60 >= 2 else 0)
            app_score = 2 if last_sw_mins <= 5 else (1 if last_sw_mins <= 15 else 0)
            vol_score, vol_type = score_volume(v, i, side)
            dps = dur_score + app_score + vol_score

            # Outcome with trail SL
            fb = min(120, n - i - 1)
            if fb < 10:
                continue
            rd = abs(entry - sl_v)
            t09 = entry - rd * 0.9 if side == "short" else entry + rd * 0.9
            tsl = entry + rd * 0.1 if side == "short" else entry - rd * 0.1
            trailed = False
            cur_sl = sl_v
            out = "OPEN"
            for k in range(i + 1, i + fb + 1):
                if not trailed:
                    if side == "short" and l[k] <= t09:
                        trailed = True
                        cur_sl = tsl
                    elif side == "long" and h[k] >= t09:
                        trailed = True
                        cur_sl = tsl
                if side == "short":
                    if l[k] <= tp_v:
                        out = "TP"
                        break
                    if h[k] >= cur_sl:
                        out = "TRAIL_SL" if trailed else "SL"
                        break
                else:
                    if h[k] >= tp_v:
                        out = "TP"
                        break
                    if l[k] <= cur_sl:
                        out = "TRAIL_SL" if trailed else "SL"
                        break

            depth_str = "/".join(f"{d:.1f}%" for d in depths)
            touch_types = "/".join(s['touch_type'] for s in swings[-3:])
            results.append({
                'sym': sym, 'side': side, 'ts': ts[i],
                'entry': round(entry, 8), 'tp': round(tp_v, 8), 'sl': round(sl_v, 8),
                'tp_p': round(tp_p, 2), 'rr': round(rr, 2),
                'sw': len(swings), 'src': source, 'lst': last_sw_mins,
                'depths': depth_str, 'touches': touch_types,
                'dps': dps, 'dur': dur_score, 'app': app_score, 'vol': vol_score,
                'vol_type': vol_type, 'pre': pre,
                'chop': round(chop_dur / 60, 1),
                'out': out,
            })
            found_setup = True
            break  # one setup per bar per symbol

        if found_setup:
            last_s = i

tp = sum(1 for r in results if r['out'] == 'TP')
sl = sum(1 for r in results if r['out'] in ('SL', 'TRAIL_SL'))
tr = sum(1 for r in results if r['out'] == 'TRAIL_SL')
op = sum(1 for r in results if r['out'] == 'OPEN')
print(f"Total: {len(results)}  TP: {tp}  SL: {sl-tr}  TRAIL: {tr}  OPEN: {op}")
if tp + sl > 0:
    print(f"WR: {tp}/{tp+sl} = {tp/(tp+sl)*100:.1f}%")

print("\nBy DPS:")
for d in sorted(set(r['dps'] for r in results)):
    sub = [r for r in results if r['dps'] == d]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL', 'TRAIL_SL'))
    if t + s > 0:
        print(f"  DPS={d}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%")

print("\nBy Side:")
for sd in ['long', 'short']:
    sub = [r for r in results if r['side'] == sd]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL', 'TRAIL_SL'))
    if t + s > 0:
        print(f"  {sd}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%")

print("\nBy Swings:")
for a, b, lb in [(1, 2, '1'), (2, 3, '2'), (3, 5, '3-4'), (5, 99, '5+')]:
    sub = [r for r in results if a <= r['sw'] < b]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL', 'TRAIL_SL'))
    if t + s > 0:
        print(f"  {lb} sw: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%")

print("\nBy TP%:")
for a, b, lb in [(0, 0.8, '<0.8%'), (0.8, 1.0, '0.8-1.0%'), (1.0, 1.5, '1.0-1.5%'), (1.5, 99, '1.5%+')]:
    sub = [r for r in results if a <= r['tp_p'] < b]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL', 'TRAIL_SL'))
    if sub:
        wr = t / (t + s) * 100 if t + s > 0 else 0
        print(f"  {lb}: {len(sub)}, TP={t} SL={s} WR={wr:.1f}%")

print()
results.sort(key=lambda x: (x['dps'], x['sw'], x['rr']), reverse=True)
print(f"{'Symbol':>14} {'Side':>5} {'DPS':>3} {'D':>1}{'A':>1}{'V':>1} {'Vol':>4} {'Pre':>4} {'Src':>8} {'Entry':>10} {'TP':>10} {'SL':>10} {'TP%':>5} {'RR':>4} {'Sw':>2} {'Lst':>3} {'Depths':>20} {'TouchTypes':>15} {'Chp':>4} {'Out':>6} {'Time'}")
print("-" * 170)
for r in results:
    print(f"{r['sym']:>14} {r['side']:>5} {r['dps']:>3} {r['dur']}{r['app']}{r['vol']} {r['vol_type'][:4]:>4} {r['pre']:>4} {r['src']:>8} {r['entry']:>10.6g} {r['tp']:>10.6g} {r['sl']:>10.6g} {r['tp_p']:>5.2f} {r['rr']:>4.1f} {r['sw']:>2} {r['lst']:>2}m {r['depths']:>20} {r['touches']:>15} {r['chop']:>3.0f}h {r['out']:>6} {r['ts']}")
