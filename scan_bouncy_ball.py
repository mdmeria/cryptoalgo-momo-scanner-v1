#!/usr/bin/env python3
"""
Bouncy Ball Scanner — Find coins oscillating between two levels.

Pattern: Price trapped in a range, bouncing between support and resistance.
- Both boundaries must have 3+ touches/closes WITHIN the current regime
- Range width > 1%
- Entry when price spikes to a boundary and rejects
- TP at opposite boundary
- SL just beyond the touched boundary
"""

import csv, os, sys
import numpy as np

DATA_DIRS = [
    'c:/Projects/CryptoAlgo/datasets/momo_1m_7d_top100_midcap_30d',
    'c:/Projects/CryptoAlgo/datasets/live/candles_1m',
]


def load_symbol(fpath):
    with open(fpath, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 300:
        return None
    return {
        'closes': np.array([float(r['close']) for r in rows]),
        'highs': np.array([float(r['high']) for r in rows]),
        'lows': np.array([float(r['low']) for r in rows]),
        'volumes': np.array([float(r['volume']) for r in rows]),
        'timestamps': [r['timestamp'][:19] for r in rows],
    }


def count_touches_fast(arr, level, tol_pct, min_gap, above=True):
    """Count how many times arr touches level (with min gap between touches)."""
    if above:
        mask = arr >= level * (1 - tol_pct / 100)
    else:
        mask = arr <= level * (1 + tol_pct / 100)
    count = 0
    last = -min_gap
    for j in range(len(arr)):
        if mask[j] and j - last >= min_gap:
            count += 1
            last = j
    return count


def count_closes_near(closes, level, tol_pct=0.3, min_gap=5):
    """Count candle closes near a level."""
    mask = np.abs(closes - level) / level * 100 < tol_pct
    count = 0
    last = -min_gap
    for j in range(len(closes)):
        if mask[j] and j - last >= min_gap:
            count += 1
            last = j
    return count


def find_range_regime(highs, lows, closes, end_bar):
    """Find choppy range regime ending at end_bar."""
    best = None

    for duration in [180, 360]:
        start = end_bar - duration
        if start < 0:
            continue

        h = highs[start:end_bar]
        l = lows[start:end_bar]
        c = closes[start:end_bar]

        upper = np.percentile(h, 92)
        lower = np.percentile(l, 8)

        range_pct = (upper - lower) / lower * 100
        if range_pct < 1.0 or range_pct > 6.0:
            continue

        # Vectorized inside check
        inside_pct = np.mean((c >= lower * 0.995) & (c <= upper * 1.005)) * 100
        if inside_pct < 70:
            continue

        ut = count_touches_fast(h, upper, 0.3, 10, above=True)
        lt = count_touches_fast(l, lower, 0.3, 10, above=False)
        uc = count_closes_near(c, upper)
        lc = count_closes_near(c, lower)

        total_upper = ut + uc
        total_lower = lt + lc

        if total_upper < 3 or total_lower < 3:
            continue

        score = total_upper + total_lower
        if best is None or score > best['score']:
            best = {
                'start': start, 'duration': duration,
                'upper': upper, 'lower': lower,
                'total_upper': total_upper, 'total_lower': total_lower,
                'range_pct': range_pct, 'inside_pct': inside_pct,
                'score': score,
            }

    return best


def scan_symbol(sym, data):
    closes = data['closes']
    highs = data['highs']
    lows = data['lows']
    volumes = data['volumes']
    timestamps = data['timestamps']
    n = len(closes)

    # ATR
    atr = np.zeros(n)
    for i in range(31, n):
        tr = np.maximum(
            highs[i-30:i] - lows[i-30:i],
            np.maximum(np.abs(highs[i-30:i] - closes[i-31:i-1]),
                       np.abs(lows[i-30:i] - closes[i-31:i-1]))
        )
        atr[i] = np.mean(tr)

    setups = []
    last_setup = -30

    for i in range(180, n - 30, 5):  # step by 5 bars for speed
        if i - last_setup < 30:
            continue
        if atr[i] <= 0:
            continue

        # --- Step 1: Is there a choppy range regime ending at bar i? ---
        regime = find_range_regime(highs, lows, closes, i)
        if regime is None:
            continue

        upper = regime['upper']
        lower = regime['lower']

        # --- Step 2: Did price just spike to a boundary? ---
        # Check last 1-5 bars for a move toward a boundary
        spike_to_upper = False
        spike_to_lower = False
        spike_bars = 0
        spike_size = 0

        for nb in [1, 2, 3, 5]:
            if i - nb < 0:
                continue
            move = closes[i] - closes[i - nb]
            move_pct = abs(move) / closes[i - nb] * 100

            # Spike up toward resistance
            if move > 0 and highs[i] >= upper * 0.997:
                if move_pct >= 0.5 or (atr[i] > 0 and abs(move) / atr[i] >= 1.5):
                    spike_to_upper = True
                    spike_bars = nb
                    spike_size = move_pct
                    break

            # Spike down toward support
            if move < 0 and lows[i] <= lower * 1.003:
                if move_pct >= 0.5 or (atr[i] > 0 and abs(move) / atr[i] >= 1.5):
                    spike_to_lower = True
                    spike_bars = nb
                    spike_size = move_pct
                    break

        if not spike_to_upper and not spike_to_lower:
            continue

        # --- Step 3: Volume flat/decreasing before spike ---
        vol_start = max(0, i - 60)
        vol_end = max(vol_start + 10, i - spike_bars)
        vol_w = volumes[vol_start:vol_end]
        if len(vol_w) < 15:
            continue
        slope = np.polyfit(np.arange(len(vol_w)), vol_w, 1)[0]
        avg_vol = np.mean(vol_w)
        vol_slope = slope / avg_vol if avg_vol > 0 else 0
        vol_ok = vol_slope <= 0.01

        # --- Step 4: Confirmation — next candle closes back inside range ---
        cb = i + 1
        if cb >= n:
            continue

        if spike_to_upper:
            side = "short"
            # Confirmation: close back below resistance
            if closes[cb] >= upper:
                continue
        else:
            side = "long"
            # Confirmation: close back above support
            if closes[cb] <= lower:
                continue

        # --- Entry, TP, SL ---
        entry = closes[cb]
        entry_ts = timestamps[cb]

        if side == "short":
            sl = max(highs[i], upper) * 1.003  # above resistance
            tp = lower * 1.003  # just above support (opposite boundary)
            sl_pct = (sl - entry) / entry * 100
            tp_pct = (entry - tp) / entry * 100
        else:
            sl = min(lows[i], lower) * 0.997  # below support
            tp = upper * 0.997  # just below resistance (opposite boundary)
            sl_pct = (entry - sl) / entry * 100
            tp_pct = (tp - entry) / entry * 100

        if sl_pct <= 0 or tp_pct <= 0:
            continue
        rr = tp_pct / sl_pct

        # --- DPS scoring ---
        # Spike quality
        a_score = 2 if spike_size >= 1.0 else (1 if spike_size >= 0.5 else 0)
        # Volume
        v_score = 2 if vol_ok and vol_slope <= 0 else (1 if vol_ok else 0)
        # Range quality (touches on both sides)
        min_touches = min(regime['total_upper'], regime['total_lower'])
        r_score = 2 if min_touches >= 5 else (1 if min_touches >= 3 else 0)
        # Range duration
        d_score = 2 if regime['duration'] >= 360 else (1 if regime['duration'] >= 180 else 0)

        dps = a_score + v_score + r_score + d_score
        if dps < 4:
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
            'symbol': sym, 'side': side,
            'spike_ts': timestamps[i], 'entry_ts': entry_ts,
            'upper': round(upper, 8), 'lower': round(lower, 8),
            'up_t': regime['total_upper'], 'lo_t': regime['total_lower'],
            'entry': round(entry, 8), 'tp': round(tp, 8), 'sl': round(sl, 8),
            'sl_pct': round(sl_pct, 2), 'tp_pct': round(tp_pct, 2),
            'rr': round(rr, 2),
            'spike_pct': round(spike_size, 2),
            'range_pct': round(regime['range_pct'], 2),
            'duration': regime['duration'],
            'inside_pct': round(regime['inside_pct'], 1),
            'dps': dps,
            'a': a_score, 'v': v_score, 'r': r_score, 'd': d_score,
            'outcome': outcome,
        })
        last_setup = i

    return setups


def main():
    files = []
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            continue
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.csv'):
                files.append((data_dir, f))

    all_setups = []
    seen = set()

    for data_dir, fname in files:
        sym = fname.replace('_1m_7d.csv', '').replace('_1m.csv', '')
        if sym in seen:
            continue
        seen.add(sym)

        try:
            data = load_symbol(os.path.join(data_dir, fname))
        except Exception:
            continue
        if data is None:
            continue

        s = scan_symbol(sym, data)
        all_setups.extend(s)
        if s:
            print(f"  {sym}: {len(s)} setups", file=sys.stderr)

    # Results
    tp_c = sum(1 for s in all_setups if s['outcome'] == 'TP')
    sl_c = sum(1 for s in all_setups if s['outcome'] == 'SL')
    op_c = sum(1 for s in all_setups if s['outcome'] == 'OPEN')
    print(f"\nTotal: {len(all_setups)}  TP: {tp_c}  SL: {sl_c}  OPEN: {op_c}")
    if tp_c + sl_c > 0:
        print(f"WR: {tp_c}/{tp_c + sl_c} = {tp_c / (tp_c + sl_c) * 100:.1f}%")

    # By DPS
    print("\nBy DPS:")
    for d in sorted(set(s['dps'] for s in all_setups)):
        sub = [s for s in all_setups if s['dps'] == d]
        t = sum(1 for s in sub if s['outcome'] == 'TP')
        sl = sum(1 for s in sub if s['outcome'] == 'SL')
        wr = t / (t + sl) * 100 if t + sl > 0 else 0
        print(f"  DPS={d}: {len(sub)} setups, TP={t} SL={sl} WR={wr:.1f}%")

    # By RR
    print("\nBy RR:")
    for rr_min, rr_max, label in [(0, 1, '<1'), (1, 1.5, '1-1.5'), (1.5, 2, '1.5-2'), (2, 99, '2+')]:
        sub = [s for s in all_setups if rr_min <= s['rr'] < rr_max]
        t = sum(1 for s in sub if s['outcome'] == 'TP')
        sl = sum(1 for s in sub if s['outcome'] == 'SL')
        wr = t / (t + sl) * 100 if t + sl > 0 else 0
        print(f"  RR {label}: {len(sub)} setups, TP={t} SL={sl} WR={wr:.1f}%")

    print()
    all_setups.sort(key=lambda x: (x['dps'], min(x['up_t'], x['lo_t']), x['rr']), reverse=True)

    hdr = (f"{'Symbol':>14} {'Side':>5} {'DPS':>3} "
           f"{'Spk%':>5} {'UpT':>3} {'LoT':>3} "
           f"{'Upper':>10} {'Lower':>10} {'Entry':>10} {'TP':>10} {'SL':>10} "
           f"{'SL%':>5} {'TP%':>5} {'RR':>4} {'Rng%':>5} {'Dur':>4} {'In%':>4} "
           f"{'Out':>4} {'Entry Time':>19}")
    print(hdr)
    print("-" * len(hdr))
    for s in all_setups[:50]:
        print(f"{s['symbol']:>14} {s['side']:>5} {s['dps']:>3} "
              f"{s['spike_pct']:>4.1f}% {s['up_t']:>3} {s['lo_t']:>3} "
              f"{s['upper']:>10.6g} {s['lower']:>10.6g} {s['entry']:>10.6g} {s['tp']:>10.6g} {s['sl']:>10.6g} "
              f"{s['sl_pct']:>5.2f} {s['tp_pct']:>5.2f} {s['rr']:>4.1f} {s['range_pct']:>5.2f} {s['duration']:>4} "
              f"{s['inside_pct']:>4.0f} "
              f"{s['outcome']:>4} {s['entry_ts']:>19}")


if __name__ == "__main__":
    main()
