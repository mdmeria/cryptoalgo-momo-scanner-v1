#!/usr/bin/env python3
"""Range Shift v2 + ZCT DPS scoring (duration, approach, volume with side-specific rules)."""
import csv, os, sys
import numpy as np

DATA_DIRS = [
    'c:/Projects/CryptoAlgo/datasets/momo_1m_7d_top100_midcap_30d',
    'c:/Projects/CryptoAlgo/datasets/live/candles_1m',
]

def load_sym(fpath):
    with open(fpath, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 500: return None
    return {
        'c': np.array([float(r['close']) for r in rows]),
        'h': np.array([float(r['high']) for r in rows]),
        'l': np.array([float(r['low']) for r in rows]),
        'v': np.array([float(r['volume']) for r in rows]),
        'ts': [r['timestamp'][:19] for r in rows],
    }

def find_range(highs, lows, closes, start, end):
    h, l, c = highs[start:end], lows[start:end], closes[start:end]
    if len(h) < 60: return None
    upper = np.percentile(h, 92)
    lower = np.percentile(l, 8)
    rng = (upper - lower) / lower * 100
    if rng < 0.8 or rng > 5.0: return None
    inside = np.mean((c >= lower * 0.998) & (c <= upper * 1.002)) * 100
    if inside < 65: return None
    return upper, lower, inside, rng

def find_real_swings(highs, lows, closes, level, start, end, range_width, side):
    min_travel = range_width * 0.4
    swings = []
    near_level = False
    farthest = 0
    for j in range(start, end):
        at_level = (abs(highs[j] - level) / level * 100 < 0.4 or
                    abs(lows[j] - level) / level * 100 < 0.4 or
                    abs(closes[j] - level) / level * 100 < 0.3)
        if side == "short":
            dist = level - lows[j]
        else:
            dist = highs[j] - level
        if at_level:
            if not near_level and farthest >= min_travel:
                swings.append({'bar': j, 'depth': farthest})
            near_level = True
            farthest = 0
        else:
            near_level = False
            if dist > farthest: farthest = dist
    swings.reverse()
    return swings

def score_volume(volumes, bar, side, lookback=60, spike_bars=3):
    """Score volume per ZCT rules. MR Long: flat=2, dec=1, inc=0. MR Short: dec=2, flat=1, inc=0."""
    start = max(0, bar - lookback)
    end = bar - spike_bars
    vol = volumes[start:end]
    if len(vol) < 20: return 1, "unclear", 0
    x = np.arange(len(vol))
    slope = np.polyfit(x, vol, 1)[0]
    avg = np.mean(vol)
    if avg <= 0: return 1, "unclear", 0
    norm = slope / avg

    if norm > 0.005:
        vol_type = "increasing"
    elif norm < -0.005:
        vol_type = "decreasing"
    else:
        vol_type = "flat"

    if side == "long":
        if vol_type == "flat": return 2, vol_type, norm
        if vol_type == "decreasing": return 1, vol_type, norm
        return 0, vol_type, norm
    else:  # short
        if vol_type == "decreasing": return 2, vol_type, norm
        if vol_type == "flat": return 1, vol_type, norm
        return 0, vol_type, norm

def score_approach(last_swing_mins):
    """Score approach: recent spike to level is better."""
    if last_swing_mins <= 5: return 2, "spike"
    if last_swing_mins <= 15: return 1, "unclear"
    return 0, "grind"

def score_duration(chop_hrs):
    """Score choppy range duration."""
    if chop_hrs >= 4: return 2
    if chop_hrs >= 2: return 1
    return 0

files = []
for d in DATA_DIRS:
    if os.path.exists(d):
        for f in sorted(os.listdir(d)):
            if f.endswith('.csv'): files.append((d, f))

results = []
seen = set()
for dd, fn in files:
    sym = fn.replace('_1m_7d.csv', '').replace('_1m.csv', '')
    if sym in seen: continue
    seen.add(sym)
    try: data = load_sym(os.path.join(dd, fn))
    except: continue
    if data is None: continue
    c, h, l, v, ts = data['c'], data['h'], data['l'], data['v'], data['ts']
    n = len(c)
    last_s = -60
    for i in range(600, n - 30, 10):
        if i - last_s < 60: continue
        curr = None
        for dur in [180, 360]:
            r = find_range(h, l, c, i - dur, i)
            if r: curr = {'upper': r[0], 'lower': r[1], 'start': i - dur, 'dur': dur, 'rng': r[3]}; break
        if not curr: continue
        rw = curr['upper'] - curr['lower']
        prev = None
        for dur in [180, 360]:
            ps = curr['start'] - dur
            if ps < 0: continue
            r = find_range(h, l, c, ps, curr['start'])
            if r: prev = {'upper': r[0], 'lower': r[1], 'start': ps}; break
        if not prev: continue
        cm = (curr['upper'] + curr['lower']) / 2
        pm = (prev['upper'] + prev['lower']) / 2
        sp = (cm - pm) / pm * 100
        if abs(sp) < 0.5: continue
        if sp < 0:
            side, el, ol = "short", curr['upper'], curr['lower']
        else:
            side, el, ol = "long", curr['lower'], curr['upper']
        if side == "short" and h[i] < el * 0.997: continue
        if side == "long" and l[i] > el * 1.003: continue
        if side == "short" and c[i] >= el: continue
        if side == "long" and c[i] <= el: continue
        swings = find_real_swings(h, l, c, el, curr['start'], i, rw, side)
        if len(swings) < 1: continue
        mins_since = i - swings[0]['bar']
        if mins_since > 30: continue
        recent = swings[:3]
        depths = [s['depth'] for s in recent]
        tp_depth = min(depths)
        entry = el
        if side == "short":
            tp = entry - tp_depth * 0.95
            sl = el * 1.005
            sl_p = (sl - entry) / entry * 100
            tp_p = (entry - tp) / entry * 100
        else:
            tp = entry + tp_depth * 0.95
            sl = el * 0.995
            sl_p = (entry - sl) / entry * 100
            tp_p = (tp - entry) / entry * 100
        if sl_p <= 0 or tp_p <= 0 or tp_p < 0.5: continue
        rr = tp_p / sl_p
        if rr < 1.0: continue

        # DPS Scoring
        chop_hrs = (i - prev['start']) / 60
        dur_score = score_duration(chop_hrs)
        app_score, app_label = score_approach(mins_since)
        vol_score, vol_label, vol_slope = score_volume(v, i, side)
        dps = dur_score + app_score + vol_score

        # Outcome
        fb = min(120, n - i - 1)
        if fb < 10: continue
        rd = abs(entry - sl)
        t09 = entry - rd * 0.9 if side == "short" else entry + rd * 0.9
        tsl = entry + rd * 0.1 if side == "short" else entry - rd * 0.1
        trailed = False; cur_sl = sl; out = "OPEN"
        for k in range(i+1, i+fb+1):
            if not trailed:
                if side == "short" and l[k] <= t09: trailed = True; cur_sl = tsl
                elif side == "long" and h[k] >= t09: trailed = True; cur_sl = tsl
            if side == "short":
                if l[k] <= tp: out = "TP"; break
                if h[k] >= cur_sl: out = "TRAIL_SL" if trailed else "SL"; break
            else:
                if h[k] >= tp: out = "TP"; break
                if l[k] <= cur_sl: out = "TRAIL_SL" if trailed else "SL"; break

        results.append({
            'sym': sym, 'side': side, 'ts': ts[i],
            'entry': round(entry, 8), 'tp': round(tp, 8), 'sl': round(sl, 8),
            'tp_p': round(tp_p, 2), 'rr': round(rr, 2),
            'sw': len(swings), 'mins': mins_since,
            'dps': dps, 'dur_s': dur_score, 'app_s': app_score, 'vol_s': vol_score,
            'vol_type': vol_label,
            'chop': round(chop_hrs, 1),
            'shift': "down" if sp < 0 else "up",
            'out': out,
        })
        last_s = i

tp = sum(1 for r in results if r['out'] == 'TP')
sl = sum(1 for r in results if r['out'] in ('SL','TRAIL_SL'))
tr = sum(1 for r in results if r['out'] == 'TRAIL_SL')
op = sum(1 for r in results if r['out'] == 'OPEN')
print(f"Total: {len(results)}  TP: {tp}  SL: {sl-tr}  TRAIL: {tr}  OPEN: {op}")
if tp+sl > 0: print(f"WR: {tp}/{tp+sl} = {tp/(tp+sl)*100:.1f}%")

print("\nBy DPS Score:")
for d in sorted(set(r['dps'] for r in results)):
    sub = [r for r in results if r['dps'] == d]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if t+s > 0: print(f"  DPS={d}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%")

print("\nBy Volume Type:")
for vt in ['decreasing', 'flat', 'increasing']:
    sub = [r for r in results if r['vol_type'] == vt]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if t+s > 0: print(f"  {vt}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%")

print("\nBy Volume Score (side-adjusted):")
for vs in [0, 1, 2]:
    sub = [r for r in results if r['vol_s'] == vs]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if t+s > 0: print(f"  Vol={vs}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%")

print("\nBy Approach Score:")
for a in [0, 1, 2]:
    sub = [r for r in results if r['app_s'] == a]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if t+s > 0: print(f"  App={a}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%")

print("\nBy Duration Score:")
for d in [0, 1, 2]:
    sub = [r for r in results if r['dur_s'] == d]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if t+s > 0: print(f"  Dur={d}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%")

print("\nBy RR + DPS>=4:")
for a,b,lb in [(1,1.5,'1-1.5'),(1.5,2,'1.5-2'),(2,99,'2+')]:
    sub = [r for r in results if a <= r['rr'] < b and r['dps'] >= 4]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if t+s > 0: print(f"  RR {lb} DPS>=4: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%")

print("\nBy Swings + DPS>=4:")
for ns in [1,2,3]:
    lb = f"{ns}" if ns < 3 else "3+"
    sub = [r for r in results if (r['sw'] == ns if ns < 3 else r['sw'] >= ns) and r['dps'] >= 4]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if t+s > 0: print(f"  {lb} sw DPS>=4: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%")

print()
results.sort(key=lambda x: (x['dps'], x['sw'], x['rr']), reverse=True)
print(f"{'Symbol':>14} {'Side':>5} {'DPS':>3} {'D':>1}{'A':>1}{'V':>1} {'Vol':>4} {'Entry':>10} {'TP':>10} {'SL':>10} {'TP%':>5} {'RR':>4} {'Sw':>2} {'Lst':>3} {'Chp':>5} {'Out':>6} {'Time'}")
print("-" * 125)
for r in results[:40]:
    print(f"{r['sym']:>14} {r['side']:>5} {r['dps']:>3} {r['dur_s']}{r['app_s']}{r['vol_s']} {r['vol_type'][:4]:>4} {r['entry']:>10.6g} {r['tp']:>10.6g} {r['sl']:>10.6g} {r['tp_p']:>5.2f} {r['rr']:>4.1f} {r['sw']:>2} {r['mins']:>2}m {r['chop']:>4.1f}h {r['out']:>6} {r['ts']}")
