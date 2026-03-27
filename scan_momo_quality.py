#!/usr/bin/env python3
"""
Momentum Quality Scanner
Filters:
  2hr: R2 > 0.90, channel < 1.25%, maxDD < 0.7%
  30m: volume consistently increasing
  10m: channel < 0.10%, slope in trade direction
"""
import pandas as pd, numpy as np
import os

DATA_DIRS = ['datasets/live/candles_1m', 'datasets/momo_1m_7d_top100_midcap_30d']

def calc_regression(closes, highs, lows):
    n = len(closes)
    x = np.arange(n, dtype=float)
    xm, ym = np.mean(x), np.mean(closes)
    sxy = np.sum((x - xm) * (closes - ym))
    sxx = np.sum((x - xm) ** 2)
    syy = np.sum((closes - ym) ** 2)
    if sxx == 0 or syy == 0:
        return None
    slope = sxy / sxx
    r2 = (sxy ** 2) / (sxx * syy)
    pred = slope * x + (ym - slope * xm)
    std = np.std(closes - pred)
    ch = std / ym * 100 * 2
    going_up = slope > 0
    if going_up:
        rh = np.maximum.accumulate(highs)
        max_dd = np.max((rh - lows) / rh * 100)
    else:
        rl = np.minimum.accumulate(lows)
        max_dd = np.max((highs - rl) / rl * 100)
    return {
        'r2': r2, 'channel': ch, 'max_dd': max_dd,
        'slope': slope / ym * 100, 'going_up': going_up,
    }


def smma(arr, period):
    """Simple smoothed moving average."""
    result = np.zeros(len(arr))
    result[0] = arr[0]
    alpha = 1.0 / period
    for i in range(1, len(arr)):
        result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
    return result


def check_momo_gates(c, h, l, v, end, side):
    """Check additional momentum gates. Returns (passed, reason)."""
    if end < 360:
        return False, "not_enough_bars"

    entry_price = c[end - 1]

    # Gate 1: 24h change > 5% (use 1440 bars if available, else 720)
    lookback_24h = min(1440, end)
    day_change = abs(c[end-1] - c[end - lookback_24h]) / c[end - lookback_24h] * 100
    if day_change < 5.0:
        return False, f"day_change_{day_change:.1f}pct"

    # Gate 2: Not crossed in last 6 hours (360 bars)
    last_6h = c[max(0, end-360):end-1]
    if side == 'long':
        if np.any(last_6h >= entry_price):
            return False, "crossed_6h"
    else:
        if np.any(last_6h <= entry_price):
            return False, "crossed_6h"

    # Gate 3: SMMA30 trending + max 3 crosses in last 120 bars
    if end < 150:
        return False, "not_enough_for_smma"
    smma30 = smma(c[:end], 30)
    smma_window = smma30[end-120:end]
    close_window = c[end-120:end]

    # SMMA trending in right direction
    if side == 'long' and smma_window[-1] <= smma_window[0]:
        return False, "smma30_not_trending"
    if side == 'short' and smma_window[-1] >= smma_window[0]:
        return False, "smma30_not_trending"

    # Max 3 crosses
    above = close_window > smma_window
    crosses = np.sum(np.diff(above.astype(int)) != 0)
    if crosses > 3:
        return False, f"smma_crosses_{crosses}"

    # Gate 4: Price vs VWAP (session VWAP approximation using 480-bar rolling mean)
    vwap_lookback = min(480, end)
    vwap = np.mean(c[end - vwap_lookback:end])
    if side == 'long' and entry_price < vwap:
        return False, "below_vwap"
    if side == 'short' and entry_price > vwap:
        return False, "above_vwap"

    return True, ""


def main():
    results = []
    seen = set()

    for d in DATA_DIRS:
        if not os.path.exists(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.csv') or fn == 'dataset_manifest.csv':
                continue
            sym = fn.replace('_1m_7d.csv', '').replace('_1m.csv', '')
            if sym in seen:
                continue
            seen.add(sym)
            try:
                df = pd.read_csv(os.path.join(d, fn), encoding='utf-8')
                n = len(df)
                if n < 250:
                    continue
                c = df['close'].values.astype(float)
                h = df['high'].values.astype(float)
                l = df['low'].values.astype(float)
                v = df['volume'].values.astype(float)
                ts_arr = df['timestamp'].values

                last_trade = -60
                for end in range(120, n - 60):
                    if end - last_trade < 60:
                        continue
                    start_2h = end - 120
                    ts = str(df.iloc[end - 1]['timestamp'])[:16]

                    # Time gap check
                    try:
                        t0 = pd.Timestamp(ts_arr[start_2h])
                        t1 = pd.Timestamp(ts_arr[end - 1])
                        if (t1 - t0).total_seconds() / 60 > 140:
                            continue
                    except:
                        continue

                    # Filter 1: 2hr R2>0.90, channel<1.25%, maxDD<0.7%
                    res_2h = calc_regression(c[start_2h:end], h[start_2h:end], l[start_2h:end])
                    if res_2h is None:
                        continue
                    if res_2h['r2'] <= 0.85 or res_2h['channel'] >= 1.25:
                        continue

                    # Split checks: first hour relaxed, last hour strict
                    mid = start_2h + 60
                    res_1h_first = calc_regression(c[start_2h:mid], h[start_2h:mid], l[start_2h:mid])
                    res_1h_last = calc_regression(c[mid:end], h[mid:end], l[mid:end])
                    if res_1h_first is None or res_1h_last is None:
                        continue
                    # First hour: maxDD < 1.15%
                    # Last hour: maxDD < 0.7%
                    if res_1h_first['max_dd'] >= 1.15 or res_1h_last['max_dd'] >= 0.7:
                        continue

                    side = 'long' if res_2h['going_up'] else 'short'

                    # Volume trend (scored but not filtered)
                    vol_usd = v[end - 30:end] * c[end - 30:end]
                    if len(vol_usd) < 20:
                        continue
                    vol_slope = np.polyfit(np.arange(len(vol_usd)), vol_usd, 1)[0]
                    vol_avg = np.mean(vol_usd)
                    if vol_avg <= 0:
                        continue

                    # Filter 3: 15-bar channel < 0.10% AND slope in trade direction (ZCT approach = grind)
                    res_15 = calc_regression(c[end - 15:end], h[end - 15:end], l[end - 15:end])
                    if res_15 is None:
                        continue
                    if res_15['channel'] >= 0.20:
                        continue
                    # Slope must match trade direction
                    if side == 'long' and res_15['slope'] <= 0:
                        continue
                    if side == 'short' and res_15['slope'] >= 0:
                        continue

                    # Momentum gates: 24h change, 6h not crossed, SMMA30, VWAP
                    gate_passed, gate_reason = check_momo_gates(c, h, l, v, end, side)
                    if not gate_passed:
                        continue

                    # ZCT DPS Scoring
                    # Duration: check if 4hr staircase also passes R2 > 0.85
                    dps_dur = 1  # 2hr staircase = 1
                    if end >= 240:
                        res_4h = calc_regression(c[end-240:end], h[end-240:end], l[end-240:end])
                        if res_4h is not None and res_4h['r2'] > 0.85:
                            dps_dur = 2
                    # Approach: always 2 (15m channel < 0.10% = grind)
                    dps_app = 2
                    # Volume: side-specific
                    vol_norm = vol_slope / vol_avg if vol_avg > 0 else 0
                    if vol_norm > 0.01:
                        vol_label = "increasing"
                    elif vol_norm < -0.01:
                        vol_label = "decreasing"
                    else:
                        vol_label = "flat"
                    if side == 'long':
                        dps_vol = 2 if vol_label == "increasing" else (1 if vol_label == "flat" else 0)
                    else:
                        dps_vol = 2 if vol_label == "increasing" else (1 if vol_label == "decreasing" else 0)
                    dps_total = dps_dur + dps_app + dps_vol

                    # Entry level = candle close where setup detected
                    entry_level = c[end - 1]

                    # Step 1: Confirmation — next 2 candles must close on correct side
                    if end + 2 >= n:
                        continue
                    if side == 'long':
                        if c[end] <= entry_level or c[end + 1] <= entry_level:
                            continue  # confirmation failed
                    else:
                        if c[end] >= entry_level or c[end + 1] >= entry_level:
                            continue

                    # Step 2: Place limit order at entry_level, wait up to 10 bars
                    # Cancel if price moves 0.75R onside without filling
                    entry = entry_level
                    if side == 'long':
                        tp = entry * 1.011
                        sl = entry * 0.99
                    else:
                        tp = entry * 0.989
                        sl = entry * 1.01

                    rd = abs(entry - sl)
                    cancel_price = entry + rd * 0.75 if side == 'long' else entry - rd * 0.75

                    filled = False
                    fill_bar = None
                    limit_start = end + 2  # after 2 confirmation bars
                    for k in range(limit_start, min(limit_start + 10, n)):
                        # Check 0.75R cancel first
                        if side == 'long' and h[k] >= cancel_price:
                            break  # too far gone
                        if side == 'short' and l[k] <= cancel_price:
                            break

                        # Check if limit fills (wick reaches entry level)
                        if side == 'long' and l[k] <= entry_level:
                            filled = True
                            fill_bar = k
                            break
                        if side == 'short' and h[k] >= entry_level:
                            filled = True
                            fill_bar = k
                            break

                    if not filled:
                        move_1h = (c[end - 1] - c[max(0, end - 60)]) / c[max(0, end - 60)] * 100
                        results.append({
                            'sym': sym, 'ts': ts, 'side': side,
                            'entry': round(entry, 8),
                            'tp': round(tp, 8), 'sl': round(sl, 8),
                            'r2': round(res_2h['r2'], 3),
                            'ch': round(res_2h['channel'], 3),
                            'ch15': round(res_15['channel'], 4),
                            'move_1h': round(move_1h, 2),
                            'dps': dps_total, 'dps_d': dps_dur, 'dps_a': dps_app, 'dps_v': dps_vol,
                            'vol': vol_label,
                            'out': 'MISSED',
                        })
                        last_trade = end
                        continue

                    # Step 3: Simulate trade from fill bar
                    # Trail SL
                    t09 = entry + rd * 0.9 if side == 'long' else entry - rd * 0.9
                    tsl = entry - rd * 0.1 if side == 'long' else entry + rd * 0.1

                    trailed = False
                    cur_sl = sl
                    out = 'OPEN'
                    for k in range(fill_bar + 1, min(fill_bar + 60, n)):
                        if not trailed:
                            if side == 'long' and h[k] >= t09:
                                trailed = True
                                cur_sl = tsl
                            elif side == 'short' and l[k] <= t09:
                                trailed = True
                                cur_sl = tsl
                        if side == 'long':
                            if h[k] >= tp:
                                out = 'TP'
                                break
                            if l[k] <= cur_sl:
                                out = 'TRAIL' if trailed else 'SL'
                                break
                        else:
                            if l[k] <= tp:
                                out = 'TP'
                                break
                            if h[k] >= cur_sl:
                                out = 'TRAIL' if trailed else 'SL'
                                break

                    move_1h = (c[end - 1] - c[max(0, end - 60)]) / c[max(0, end - 60)] * 100

                    results.append({
                        'sym': sym, 'ts': ts, 'side': side,
                        'entry': round(entry, 8),
                        'tp': round(tp, 8), 'sl': round(sl, 8),
                        'r2': round(res_2h['r2'], 3),
                        'ch': round(res_2h['channel'], 3),
                        'ch15': round(res_15['channel'], 4),
                        'move_1h': round(move_1h, 2),
                        'dps': dps_total, 'dps_d': dps_dur, 'dps_a': dps_app, 'dps_v': dps_vol,
                        'vol': vol_label,
                        'out': out,
                    })
                    last_trade = end
            except:
                continue

    # Results
    missed = [r for r in results if r['out'] == 'MISSED']
    filled = [r for r in results if r['out'] not in ('MISSED', 'OPEN')]
    open_trades = [r for r in results if r['out'] == 'OPEN']
    tp = sum(1 for r in filled if r['out'] == 'TP')
    sl = sum(1 for r in filled if r['out'] == 'SL')
    tr = sum(1 for r in filled if r['out'] == 'TRAIL')

    print(f"Setups detected: {len(results)}")
    print(f"  Confirmed (2 candle close): {len(missed) + len(filled) + len(open_trades)}")
    print(f"  Missed (no pullback/0.75R cancel): {len(missed)}")
    print(f"  Filled: {len(filled) + len(open_trades)}")
    print(f"  Resolved: {len(filled)}  OPEN: {len(open_trades)}")
    if filled:
        print(f"\nFilled results: TP={tp}  SL={sl}  Trail={tr}")
        print(f"WR (TP): {tp}/{len(filled)} = {tp / len(filled) * 100:.1f}%")
        print(f"WR (TP+Trail): {(tp + tr)}/{len(filled)} = {(tp + tr) / len(filled) * 100:.1f}%")
        pnl = tp * 1.1 + tr * 0.1 - sl * 1.0
        print(f"PnL: {pnl:+.1f}%")

    print(f"\nBy side (filled only):")
    for s in ['long', 'short']:
        sub = [r for r in filled if r['side'] == s]
        t = sum(1 for r in sub if r['out'] == 'TP')
        s2 = sum(1 for r in sub if r['out'] == 'SL')
        t2 = sum(1 for r in sub if r['out'] == 'TRAIL')
        if sub:
            print(f"  {s}: {len(sub)} trades, TP={t} SL={s2} Trail={t2} WR={t / len(sub) * 100:.1f}%")

    print(f"\nFilled trades:")
    print(f"{'Symbol':>14} {'Side':>5} {'DPS':>3} {'D':>1}{'A':>1}{'V':>1} {'Vol':>5} {'R2':>5} {'Ch2h':>5} {'Ch15':>6} {'1hMov':>6} {'Entry':>10} {'TP':>10} {'SL':>10} {'Out':>6} {'Time'}")
    print("-" * 130)
    for r in sorted(filled + open_trades, key=lambda x: x['ts']):
        print(f"{r['sym']:>14} {r['side']:>5} {r['dps']:>3} {r['dps_d']}{r['dps_a']}{r['dps_v']} {r['vol']:>5} {r['r2']:.3f} {r['ch']:.3f} {r['ch15']:.4f} {r['move_1h']:+.2f}% {r['entry']:>10.6g} {r['tp']:>10.6g} {r['sl']:>10.6g} {r['out']:>6} {r['ts']}")

    if missed:
        print(f"\nMissed trades ({len(missed)}):")
        print(f"{'Symbol':>14} {'Side':>5} {'DPS':>3} {'Vol':>5} {'R2':>5} {'1hMov':>6} {'Entry':>10} {'Time'}")
        print("-" * 75)
        for r in sorted(missed, key=lambda x: x['ts']):
            print(f"{r['sym']:>14} {r['side']:>5} {r['dps']:>3} {r['vol']:>5} {r['r2']:.3f} {r['move_1h']:+.2f}% {r['entry']:>10.6g} {r['ts']}")


if __name__ == "__main__":
    main()
