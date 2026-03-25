#!/usr/bin/env python3
"""Range Shift Scanner with last-touch recency filter."""
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

def count_touches(highs, lows, closes, level, min_gap=10):
    count = 0
    last = -min_gap
    for j in range(len(highs)):
        if j - last < min_gap: continue
        near = (abs(highs[j] - level) / level * 100 < 0.4 or
                abs(lows[j] - level) / level * 100 < 0.4 or
                abs(closes[j] - level) / level * 100 < 0.3)
        if near:
            count += 1
            last = j
    return count

def last_touch_before(highs, lows, closes, level, end_bar, start_bar):
    for j in range(end_bar - 1, max(start_bar - 1, 0), -1):
        near = (abs(highs[j] - level) / level * 100 < 0.4 or
                abs(lows[j] - level) / level * 100 < 0.4 or
                abs(closes[j] - level) / level * 100 < 0.3)
        if near:
            return j
    return None

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
    c, h, l, ts = data['c'], data['h'], data['l'], data['ts']
    n = len(c)
    last_s = -60
    for i in range(600, n - 30, 10):
        if i - last_s < 60: continue
        curr = None
        for dur in [180, 360]:
            r = find_range(h, l, c, i - dur, i)
            if r: curr = {'upper': r[0], 'lower': r[1], 'start': i - dur, 'dur': dur, 'rng': r[3]}; break
        if not curr: continue
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
            side, el, tl = "short", curr['upper'], curr['lower']
        else:
            side, el, tl = "long", curr['lower'], curr['upper']
        if side == "short" and h[i] < el * 0.997: continue
        if side == "long" and l[i] > el * 1.003: continue
        if side == "short" and c[i] >= el: continue
        if side == "long" and c[i] <= el: continue
        # Last touch recency
        lt = last_touch_before(h, l, c, el, i, curr['start'])
        if lt is None: continue
        mins_since = i - lt
        # Touches
        et = count_touches(h[curr['start']:i], l[curr['start']:i], c[curr['start']:i], el)
        if et < 1: continue
        ph, pl, pc = h[prev['start']:curr['start']], l[prev['start']:curr['start']], c[prev['start']:curr['start']]
        if count_touches(ph, pl, pc, prev['upper']) < 2: continue
        if count_touches(ph, pl, pc, prev['lower']) < 2: continue
        entry = el
        if side == "short":
            sl_v = el * 1.005; tp_v = tl * 1.003
            sl_p = (sl_v - entry) / entry * 100; tp_p = (entry - tp_v) / entry * 100
        else:
            sl_v = el * 0.995; tp_v = tl * 0.997
            sl_p = (entry - sl_v) / entry * 100; tp_p = (tp_v - entry) / entry * 100
        if sl_p <= 0 or tp_p <= 0 or tp_p < 0.5: continue
        rr = tp_p / sl_p
        if rr < 1.0: continue
        fb = min(120, n - i - 1)
        if fb < 10: continue
        rd = abs(entry - sl_v)
        t09 = entry - rd * 0.9 if side == "short" else entry + rd * 0.9
        tsl = entry + rd * 0.1 if side == "short" else entry - rd * 0.1
        trailed = False; cur_sl = sl_v; out = "OPEN"
        for k in range(i+1, i+fb+1):
            if not trailed:
                if side == "short" and l[k] <= t09: trailed = True; cur_sl = tsl
                elif side == "long" and h[k] >= t09: trailed = True; cur_sl = tsl
            if side == "short":
                if l[k] <= tp_v: out = "TP"; break
                if h[k] >= cur_sl: out = "TRAIL_SL" if trailed else "SL"; break
            else:
                if h[k] >= tp_v: out = "TP"; break
                if l[k] <= cur_sl: out = "TRAIL_SL" if trailed else "SL"; break
        chop_hrs = (i - prev['start']) / 60
        results.append({
            'sym': sym, 'side': side, 'ts': ts[i],
            'entry': round(entry, 8), 'tp': round(tp_v, 8), 'sl': round(sl_v, 8),
            'tp_p': round(tp_p, 2), 'rr': round(rr, 2),
            'et': et, 'mins': mins_since, 'chop': round(chop_hrs, 1),
            'rng': round(curr['rng'], 2), 'shift': "down" if sp < 0 else "up", 'out': out,
        })
        last_s = i

tp = sum(1 for r in results if r['out'] == 'TP')
sl = sum(1 for r in results if r['out'] in ('SL','TRAIL_SL'))
print(f"Total: {len(results)}  TP: {tp}  SL: {sl}")
if tp+sl > 0: print(f"WR: {tp}/{tp+sl} = {tp/(tp+sl)*100:.1f}%")

print("\nBy Last Touch Recency:")
for a, b, lb in [(0, 10, '<10min'), (10, 30, '10-30min'), (30, 60, '30-60min'), (60, 999, '60min+')]:
    sub = [r for r in results if a <= r['mins'] < b]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if sub:
        wr = t/(t+s)*100 if t+s>0 else 0
        print(f"  {lb}: {len(sub)}, TP={t} SL={s} WR={wr:.1f}%")

fresh = [r for r in results if r['mins'] <= 30]
tp_f = sum(1 for r in fresh if r['out'] == 'TP')
sl_f = sum(1 for r in fresh if r['out'] in ('SL','TRAIL_SL'))
if tp_f + sl_f > 0:
    print(f"\nFresh (<=30min): {len(fresh)}, TP={tp_f} SL={sl_f} WR={tp_f/(tp_f+sl_f)*100:.1f}%")

print("\nBy RR (fresh only):")
for a,b,lb in [(1,1.5,'1-1.5'),(1.5,2,'1.5-2'),(2,99,'2+')]:
    sub = [r for r in fresh if a <= r['rr'] < b]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if sub:
        wr = t/(t+s)*100 if t+s>0 else 0
        print(f"  RR {lb}: {len(sub)}, TP={t} SL={s} WR={wr:.1f}%")

print()
fresh.sort(key=lambda x: (x['chop'], x['rr']), reverse=True)
print(f"{'Symbol':>14} {'Side':>5} {'Entry':>10} {'TP':>10} {'SL':>10} {'TP%':>5} {'RR':>4} {'ET':>2} {'Last':>4} {'Chp':>5} {'Out':>6} {'Time'}")
print("-" * 120)
for r in fresh[:40]:
    print(f"{r['sym']:>14} {r['side']:>5} {r['entry']:>10.6g} {r['tp']:>10.6g} {r['sl']:>10.6g} {r['tp_p']:>5.2f} {r['rr']:>4.1f} {r['et']:>2} {r['mins']:>3}m {r['chop']:>4.1f}h {r['out']:>6} {r['ts']}")
