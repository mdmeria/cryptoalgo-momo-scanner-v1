#!/usr/bin/env python3
"""
Range Shift Scanner — Find coins that were chopping at one level,
then shifted to chop at a new level.

Entry: at the boundary where price shifted FROM (e.g. shifted down → short at upper boundary)
Only needs 1 touch to confirm new level, enter on 2nd touch.
"""
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

def find_range(highs, lows, closes, start, end, pct_upper=92, pct_lower=8):
    """Find range boundaries in a window. Returns (upper, lower, inside_pct) or None."""
    h, l, c = highs[start:end], lows[start:end], closes[start:end]
    if len(h) < 60: return None
    upper = np.percentile(h, pct_upper)
    lower = np.percentile(l, pct_lower)
    rng = (upper - lower) / lower * 100
    if rng < 0.8 or rng > 5.0: return None
    inside = np.mean((c >= lower * 0.998) & (c <= upper * 1.002)) * 100
    if inside < 65: return None
    return upper, lower, inside, rng

def count_touches(highs, lows, closes, level, min_gap=10):
    """Count touches near a level."""
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
    last_setup = -60

    for i in range(600, n - 30, 10):
        if i - last_setup < 60: continue

        # --- Step 1: Find CURRENT range (last 180-360 bars) ---
        curr_range = None
        for dur in [180, 360]:
            r = find_range(h, l, c, i - dur, i)
            if r is not None:
                curr_range = {'upper': r[0], 'lower': r[1], 'inside': r[2],
                              'rng': r[3], 'start': i - dur, 'dur': dur}
                break
        if curr_range is None: continue

        # --- Step 2: Find PREVIOUS range (before current range) ---
        prev_range = None
        curr_start = curr_range['start']
        for dur in [180, 360]:
            prev_start = curr_start - dur
            if prev_start < 0: continue
            r = find_range(h, l, c, prev_start, curr_start)
            if r is not None:
                prev_range = {'upper': r[0], 'lower': r[1], 'inside': r[2],
                              'rng': r[3], 'start': prev_start, 'dur': dur}
                break
        if prev_range is None: continue

        # --- Step 3: Did the range SHIFT? ---
        # Ranges must overlap somewhat (same coin, nearby prices) but be at different levels
        curr_mid = (curr_range['upper'] + curr_range['lower']) / 2
        prev_mid = (prev_range['upper'] + prev_range['lower']) / 2
        shift_pct = (curr_mid - prev_mid) / prev_mid * 100

        # Need meaningful shift (at least 0.5%)
        if abs(shift_pct) < 0.5: continue

        # Determine direction: shifted down → short bias, shifted up → long bias
        if shift_pct < 0:
            shift_dir = "down"
            # Short at the UPPER boundary of the new (lower) range
            side = "short"
            entry_level = curr_range['upper']
            tp_level = curr_range['lower']
        else:
            shift_dir = "up"
            # Long at the LOWER boundary of the new (higher) range
            side = "long"
            entry_level = curr_range['lower']
            tp_level = curr_range['upper']

        # --- Step 4: Is price at the entry boundary NOW? ---
        if side == "short":
            at_boundary = h[i] >= entry_level * 0.997
        else:
            at_boundary = l[i] <= entry_level * 1.003
        if not at_boundary: continue

        # --- Step 5: Confirmation — close back inside range ---
        if side == "short" and c[i] >= entry_level: continue
        if side == "long" and c[i] <= entry_level: continue

        # --- Step 6: Count touches at entry level in current range ---
        curr_h = h[curr_range['start']:i]
        curr_l = l[curr_range['start']:i]
        curr_c = c[curr_range['start']:i]
        entry_touches = count_touches(curr_h, curr_l, curr_c, entry_level)

        # Need at least 1 previous touch (this is the 2nd touch = entry)
        if entry_touches < 1: continue

        # --- Step 7: Count touches at previous range (quality check) ---
        prev_h = h[prev_range['start']:curr_start]
        prev_l = l[prev_range['start']:curr_start]
        prev_c = c[prev_range['start']:curr_start]
        prev_upper_t = count_touches(prev_h, prev_l, prev_c, prev_range['upper'])
        prev_lower_t = count_touches(prev_h, prev_l, prev_c, prev_range['lower'])

        # Previous range should be real (at least 3 touches on each side)
        if prev_upper_t < 2 or prev_lower_t < 2: continue

        # --- Entry, TP, SL ---
        entry = entry_level
        if side == "short":
            sl = entry_level * 1.005
            tp = tp_level * 1.003
            sl_pct = (sl - entry) / entry * 100
            tp_pct = (entry - tp) / entry * 100
        else:
            sl = entry_level * 0.995
            tp = tp_level * 0.997
            sl_pct = (entry - sl) / entry * 100
            tp_pct = (tp - entry) / entry * 100

        if sl_pct <= 0 or tp_pct <= 0: continue
        if tp_pct < 0.5: continue  # need some room
        rr = tp_pct / sl_pct
        if rr < 1.0: continue

        # --- Outcome (2 hour window) with trail SL ---
        fb = min(120, n - i - 1)
        if fb < 10: continue

        r_dist = abs(entry - sl)
        target_09r = entry - r_dist * 0.9 if side == "short" else entry + r_dist * 0.9
        trail_sl = entry + r_dist * 0.1 if side == "short" else entry - r_dist * 0.1
        trailed = False
        current_sl = sl
        out = "OPEN"

        for k in range(i + 1, i + fb + 1):
            if not trailed:
                if side == "short" and l[k] <= target_09r:
                    trailed = True; current_sl = trail_sl
                elif side == "long" and h[k] >= target_09r:
                    trailed = True; current_sl = trail_sl
            if side == "short":
                if l[k] <= tp: out = "TP"; break
                if h[k] >= current_sl:
                    out = "TRAIL_SL" if trailed else "SL"; break
            else:
                if h[k] >= tp: out = "TP"; break
                if l[k] <= current_sl:
                    out = "TRAIL_SL" if trailed else "SL"; break

        # Total chop duration (prev range + current range)
        total_chop_bars = i - prev_range['start']
        total_chop_hrs = total_chop_bars / 60

        results.append({
            'sym': sym, 'side': side, 'ts': ts[i],
            'shift': shift_dir, 'shift_pct': round(shift_pct, 2),
            'prev_upper': round(prev_range['upper'], 8),
            'prev_lower': round(prev_range['lower'], 8),
            'curr_upper': round(curr_range['upper'], 8),
            'curr_lower': round(curr_range['lower'], 8),
            'entry': round(entry, 8), 'tp': round(tp, 8), 'sl': round(sl, 8),
            'sl_pct': round(sl_pct, 2), 'tp_pct': round(tp_pct, 2), 'rr': round(rr, 2),
            'entry_touches': entry_touches,
            'prev_ut': prev_upper_t, 'prev_lt': prev_lower_t,
            'chop_hrs': round(total_chop_hrs, 1),
            'curr_rng': round(curr_range['rng'], 2),
            'out': out,
        })
        last_setup = i

# Results
tp = sum(1 for r in results if r['out'] == 'TP')
sl = sum(1 for r in results if r['out'] in ('SL', 'TRAIL_SL'))
tr = sum(1 for r in results if r['out'] == 'TRAIL_SL')
op = sum(1 for r in results if r['out'] == 'OPEN')
print(f"Total: {len(results)}  TP: {tp}  SL: {sl-tr}  TRAIL: {tr}  OPEN: {op}")
if tp + sl > 0:
    print(f"WR: {tp}/{tp+sl} = {tp/(tp+sl)*100:.1f}%")

print("\nBy Shift Direction:")
for sd in ['down', 'up']:
    sub = [r for r in results if r['shift'] == sd]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if t+s > 0: print(f"  {sd}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%")

print("\nBy RR:")
for a,b,lb in [(1,1.5,'1-1.5'),(1.5,2,'1.5-2'),(2,3,'2-3'),(3,99,'3+')]:
    sub = [r for r in results if a <= r['rr'] < b]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if sub: print(f"  RR {lb}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%" if t+s>0 else f"  RR {lb}: {len(sub)}")

print("\nBy Chop Duration:")
for a,b,lb in [(0,6,'<6h'),(6,10,'6-10h'),(10,99,'10h+')]:
    sub = [r for r in results if a <= r['chop_hrs'] < b]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if sub: print(f"  {lb}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%" if t+s>0 else f"  {lb}: {len(sub)}")

print()
results.sort(key=lambda x: (x['chop_hrs'], x['rr']), reverse=True)

print(f"{'Symbol':>14} {'Side':>5} {'Shft':>4} {'Sh%':>5} {'PrevR':>12} {'CurrR':>12} {'Entry':>10} {'TP':>10} {'SL':>10} {'TP%':>5} {'RR':>4} {'ET':>2} {'Chp':>5} {'Out':>6} {'Time'}")
print("-" * 150)
for r in results[:50]:
    prev_r = f"{r['prev_lower']:.5g}-{r['prev_upper']:.5g}"
    curr_r = f"{r['curr_lower']:.5g}-{r['curr_upper']:.5g}"
    print(f"{r['sym']:>14} {r['side']:>5} {r['shift']:>4} {r['shift_pct']:>+4.1f}% {prev_r:>12} {curr_r:>12} {r['entry']:>10.6g} {r['tp']:>10.6g} {r['sl']:>10.6g} {r['tp_pct']:>5.2f} {r['rr']:>4.1f} {r['entry_touches']:>2} {r['chop_hrs']:>4.1f}h {r['out']:>6} {r['ts']}")
