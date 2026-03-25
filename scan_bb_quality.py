#!/usr/bin/env python3
"""Bouncy Ball v3 — with level respect quality + range cleanliness filters."""
import csv, os, sys
import numpy as np

DATA_DIRS = [
    'c:/Projects/CryptoAlgo/datasets/momo_1m_7d_top100_midcap_30d',
    'c:/Projects/CryptoAlgo/datasets/live/candles_1m',
]

def load_sym(fpath):
    with open(fpath, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 300: return None
    return {
        'c': np.array([float(r['close']) for r in rows]),
        'h': np.array([float(r['high']) for r in rows]),
        'l': np.array([float(r['low']) for r in rows]),
        'ts': [r['timestamp'][:19] for r in rows],
    }

def touch_quality(highs, lows, closes, level, min_gap=10):
    """Count touches + measure overshoot (how cleanly level is respected)."""
    n = len(highs)
    touches = 0
    overshoots = []
    last = -min_gap
    for j in range(n):
        if j - last < min_gap: continue
        # Wick near level (within 0.5%)
        near_high = abs(highs[j] - level) / level * 100 < 0.5
        near_low = abs(lows[j] - level) / level * 100 < 0.5
        close_at = abs(closes[j] - level) / level * 100 < 0.3
        if near_high or near_low or close_at:
            touches += 1
            os_h = max(0, (highs[j] - level) / level * 100) if highs[j] > level else 0
            os_l = max(0, (level - lows[j]) / level * 100) if lows[j] < level else 0
            overshoots.append(max(os_h, os_l))
            last = j
    avg_os = np.mean(overshoots) if overshoots else 999
    return touches, avg_os

def range_clean(highs, lows, closes, upper, lower):
    """Score how clean the range is (0-10)."""
    n = len(closes)
    if n < 50: return 0
    inside = np.mean((closes >= lower * 0.998) & (closes <= upper * 1.002)) * 100
    blasts = np.sum(highs > upper * 1.005) + np.sum(lows < lower * 0.995)
    blast_pct = blasts / n * 100
    ret = np.diff(closes) / closes[:-1]
    ac = np.corrcoef(ret[:-1], ret[1:])[0, 1] if len(ret) > 20 else 0
    s = 0
    if inside > 90: s += 3
    elif inside > 80: s += 2
    elif inside > 70: s += 1
    if blast_pct < 2: s += 3
    elif blast_pct < 5: s += 2
    elif blast_pct < 10: s += 1
    if ac < -0.1: s += 2
    elif ac < 0: s += 1
    mid = (upper + lower) / 2
    end_pos = abs(closes[-1] - mid) / (upper - lower) if upper != lower else 1
    if end_pos < 0.3: s += 2
    elif end_pos < 0.5: s += 1
    return s

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
    for i in range(360, n - 30, 10):
        if i - last_s < 60: continue
        s0 = i - 360
        ch, cl, cc = h[s0:i], l[s0:i], c[s0:i]
        upper = np.percentile(ch, 92)
        lower = np.percentile(cl, 8)
        rng = (upper - lower) / lower * 100
        if rng < 1.0 or rng > 5.0: continue
        ut, uo = touch_quality(ch, cl, cc, upper)
        lt_, lo = touch_quality(ch, cl, cc, lower)
        if ut < 3 or lt_ < 3: continue
        avg_os = (uo + lo) / 2
        if avg_os > 0.25: continue
        cs = range_clean(ch, cl, cc, upper, lower)
        if cs < 5: continue
        at_u = h[i] >= upper * 0.997
        at_l = l[i] <= lower * 1.003
        if not at_u and not at_l: continue

        # --- Pre-chop trend: what was price doing BEFORE the range? ---
        # Look 120-360 bars before the range started
        pre_start = max(0, s0 - 240)
        pre_end = s0
        if pre_end - pre_start >= 60:
            pre_close_start = c[pre_start]
            pre_close_end = c[pre_end]
            pre_trend_pct = (pre_close_end - pre_close_start) / pre_close_start * 100
            if pre_trend_pct > 0.5:
                pre_trend = "up"
            elif pre_trend_pct < -0.5:
                pre_trend = "down"
            else:
                pre_trend = "flat"
        else:
            pre_trend = "flat"
            pre_trend_pct = 0

        # Direction filter: only trade WITH the pre-chop trend
        if at_u:
            side = "short"
            if pre_trend == "up": continue  # was trending up, don't short
        else:
            side = "long"
            if pre_trend == "down": continue  # was trending down, don't long

        # --- Confirmation: next candle closes back inside range ---
        cb = i + 1
        if cb >= n: continue
        if side == "short" and c[cb] >= upper: continue
        if side == "long" and c[cb] <= lower: continue

        # --- Entry at the level (limit order style) ---
        if side == "short":
            entry = upper  # limit sell at resistance
            sl = upper * 1.005; tp = lower * 1.003
            sl_p = (sl - entry) / entry * 100; tp_p = (entry - tp) / entry * 100
        else:
            entry = lower  # limit buy at support
            sl = lower * 0.995; tp = upper * 0.997
            sl_p = (entry - sl) / entry * 100; tp_p = (tp - entry) / entry * 100
        if sl_p <= 0 or tp_p <= 0: continue
        rr = tp_p / sl_p
        if rr < 0.8: continue

        # --- Outcome with Trail SL: 0.9R → move SL to 0.1R ---
        fb = min(120, n - cb - 1)
        out = "OPEN"
        r_dist = abs(entry - sl)  # 1R distance
        target_09r = entry - r_dist * 0.9 if side == "short" else entry + r_dist * 0.9
        trail_sl = entry + r_dist * 0.1 if side == "short" else entry - r_dist * 0.1
        trailed = False
        current_sl = sl

        for k in range(cb+1, cb+fb+1):
            # Check trail trigger first
            if not trailed:
                if side == "short" and l[k] <= target_09r:
                    trailed = True; current_sl = trail_sl
                elif side == "long" and h[k] >= target_09r:
                    trailed = True; current_sl = trail_sl

            # Check TP/SL
            if side == "short":
                if l[k] <= tp: out = "TP"; break
                if h[k] >= current_sl:
                    out = "TRAIL_SL" if trailed else "SL"; break
            else:
                if h[k] >= tp: out = "TP"; break
                if l[k] <= current_sl:
                    out = "TRAIL_SL" if trailed else "SL"; break

        results.append({
            'sym': sym, 'side': side, 'ts': ts[cb],
            'upper': round(upper, 8), 'lower': round(lower, 8),
            'entry': round(entry, 8), 'tp': round(tp, 8), 'sl': round(sl, 8),
            'sl_p': round(sl_p, 2), 'tp_p': round(tp_p, 2), 'rr': round(rr, 2),
            'rng': round(rng, 2), 'ut': ut, 'lt': lt_, 'uo': round(uo, 3),
            'lo': round(lo, 3), 'cs': cs, 'out': out,
            'pre_trend': pre_trend, 'pre_pct': round(pre_trend_pct, 2),
        })
        last_s = i

tp = sum(1 for r in results if r['out'] == 'TP')
sl = sum(1 for r in results if r['out'] in ('SL', 'TRAIL_SL'))
tr = sum(1 for r in results if r['out'] == 'TRAIL_SL')
op = sum(1 for r in results if r['out'] == 'OPEN')
print(f"Total: {len(results)}  TP: {tp}  SL: {sl-tr}  TRAIL_SL: {tr}  OPEN: {op}")
if tp+sl > 0: print(f"WR: {tp}/{tp+sl} = {tp/(tp+sl)*100:.1f}%")

print("\nBy Clean Score:")
for cs in sorted(set(r['cs'] for r in results)):
    sub = [r for r in results if r['cs'] == cs]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    print(f"  Clean={cs}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%" if t+s>0 else f"  Clean={cs}: {len(sub)}")

print("\nBy RR:")
for a,b,lb in [(0,1,'<1'),(1,1.5,'1-1.5'),(1.5,2,'1.5-2'),(2,99,'2+')]:
    sub = [r for r in results if a <= r['rr'] < b]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    print(f"  RR {lb}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%" if t+s>0 else f"  RR {lb}: {len(sub)}")

print("\nBy Pre-Trend:")
for pt in ['up', 'down', 'flat']:
    sub = [r for r in results if r['pre_trend'] == pt]
    t = sum(1 for r in sub if r['out'] == 'TP')
    s = sum(1 for r in sub if r['out'] in ('SL','TRAIL_SL'))
    if sub: print(f"  {pt}: {len(sub)}, TP={t} SL={s} WR={t/(t+s)*100:.1f}%" if t+s>0 else f"  {pt}: {len(sub)}")

print()
results.sort(key=lambda x: (x['cs'], min(x['ut'], x['lt'])), reverse=True)
print(f"{'Symbol':>14} {'Side':>5} {'UpT':>3} {'LoT':>3} {'Cln':>3} {'Pre':>4} {'Entry':>10} {'TP':>10} {'SL':>10} {'SL%':>5} {'TP%':>5} {'RR':>4} {'Rng':>4} {'Out':>6} {'Entry Time'}")
print("-" * 140)
for r in results[:50]:
    print(f"{r['sym']:>14} {r['side']:>5} {r['ut']:>3} {r['lt']:>3} {r['cs']:>3} {r['pre_trend']:>4} {r['entry']:>10.6g} {r['tp']:>10.6g} {r['sl']:>10.6g} {r['sl_p']:>5.2f} {r['tp_p']:>5.2f} {r['rr']:>4.1f} {r['rng']:>4.1f} {r['out']:>6} {r['ts']}")
