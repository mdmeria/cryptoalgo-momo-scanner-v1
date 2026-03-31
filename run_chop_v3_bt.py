#!/usr/bin/env python3
"""Quick backtest for MR chop v3 with full ZCT DPS breakdown."""
import sys, pandas as pd, numpy as np
from strategy_mr_chop import MRChopSettings, check_range_shift_setup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

syms = ['1000PEPEUSDT','AAVEUSDT','BCHUSDT','BERAUSDT','DOGEUSDT','DYDXUSDT',
        'ENSOUSDT','HYPEUSDT','LINKUSDT','PENDLEUSDT','PIXELUSDT','SIRENUSDT',
        'SUIUSDT','XLMUSDT','XPTUSDT','ZROUSDT']

cfg = MRChopSettings()
setups = []
cutoff = pd.Timestamp('2026-03-20', tz='UTC')

for sym in syms:
    try:
        df = pd.read_csv(f'datasets/live/candles_1m/{sym}_1m.csv')
    except Exception:
        continue
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df[df['timestamp'] >= cutoff].reset_index(drop=True)
    if len(df) < 620:
        continue
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    n = len(df)
    last_bar = -60
    for i in range(600, n - 5):
        if i - last_bar < 60:
            continue
        result = check_range_shift_setup(df, i, cfg)
        if not result['passed']:
            continue
        entry = result['entry']; sl = result['sl']; tp = result['tp']; side = result['side']
        filled = False; fill_bar = None
        for k in range(i + 1, min(i + 4, n)):
            if side == 'short' and h[k] >= entry:
                filled = True; fill_bar = k; break
            elif side == 'long' and l[k] <= entry:
                filled = True; fill_bar = k; break
        outcome = 'MISSED'; pnl = 0.0
        if filled:
            r_dist = abs(entry - sl)
            t09 = entry + r_dist * 0.9 if side == 'long' else entry - r_dist * 0.9
            tsl = entry - r_dist * 0.1 if side == 'long' else entry + r_dist * 0.1
            trailed = False; cur_sl = sl; outcome = 'OPEN'
            for k2 in range(fill_bar + 1, min(fill_bar + 120, n)):
                if not trailed:
                    if side == 'long' and h[k2] >= t09:
                        trailed = True; cur_sl = tsl
                    elif side == 'short' and l[k2] <= t09:
                        trailed = True; cur_sl = tsl
                if side == 'long':
                    if l[k2] <= cur_sl:
                        outcome = 'TRAIL' if trailed else 'SL'; break
                    if h[k2] >= tp:
                        outcome = 'TP'; break
                else:
                    if h[k2] >= cur_sl:
                        outcome = 'TRAIL' if trailed else 'SL'; break
                    if l[k2] <= tp:
                        outcome = 'TP'; break
            if outcome == 'TP': pnl = result['tp_pct']
            elif outcome == 'SL': pnl = -result['sl_pct']
            elif outcome == 'TRAIL': pnl = 0.1 * result['sl_pct']
            elif outcome == 'OPEN':
                lc = c[min(fill_bar + 120, n) - 1]
                pnl = (lc - entry) / entry * 100 if side == 'long' else (entry - lc) / entry * 100

        dps_t = result['dps_total']
        conf = 'max' if dps_t >= 6 else ('high' if dps_t >= 4 else ('low' if dps_t >= 3 else 'avoid'))

        setups.append({
            'symbol': sym, 'ts': str(df.iloc[i]['timestamp']),
            'side': side, 'entry': round(entry, 8), 'tp': round(tp, 8), 'sl': round(sl, 8),
            'tp_pct': result['tp_pct'], 'sl_pct': result['sl_pct'], 'rr': result['rr'],
            'dps': dps_t, 'dps_dur': result['dps_dur'], 'dps_app': result['dps_app'], 'dps_vol': result['dps_vol'],
            'confidence': conf,
            'chop_hrs': result['chop_hrs'], 'containment': result.get('containment_pct', 0),
            'src': result.get('level_source', '?'),
            'swings': result['n_swings'], 'pre': result.get('pre_trend', '?'),
            'approach': result.get('approach_type', '?'), 'vol': result.get('vol_type', '?'),
            'r2': result.get('r2', 0), 'ch_pct': result.get('channel_pct', 0),
            'outcome': outcome, 'pnl': round(pnl, 3),
        })
        last_bar = i

sdf = pd.DataFrame(setups)
sdf.to_csv('mr_chop_v3_recent_backtest.csv', index=False)
filled = sdf[sdf['outcome'] != 'MISSED']

print(f"Total: {len(sdf)} setups, {len(filled)} filled\n")

def show_breakdown(label, col, filled_df):
    print(f"--- {label} ---")
    for val in sorted(filled_df[col].unique()):
        sub = filled_df[filled_df[col] == val]
        tp_c = (sub['outcome'] == 'TP').sum()
        sl_c = (sub['outcome'] == 'SL').sum()
        tr_c = (sub['outcome'] == 'TRAIL').sum()
        wr = tp_c / (tp_c + sl_c) * 100 if (tp_c + sl_c) > 0 else 0
        print(f"  {str(val):8s}: {len(sub):3d} trades | TP={tp_c:2d} SL={sl_c:2d} Trail={tr_c:2d} | WR={wr:5.1f}% | PnL={sub['pnl'].sum():+.2f}%")
    print()

show_breakdown("By DPS Total", "dps", filled)
show_breakdown("By Confidence", "confidence", filled)
show_breakdown("By DPS Duration", "dps_dur", filled)
show_breakdown("By DPS Approach", "dps_app", filled)
show_breakdown("By DPS Volume", "dps_vol", filled)
show_breakdown("By Approach Type", "approach", filled)
show_breakdown("By Side", "side", filled)
show_breakdown("By Pre-Trend", "pre", filled)
show_breakdown("By Vol Type", "vol", filled)

# Chop duration buckets
print("--- By Chop Duration ---")
for lo, hi, lbl in [(2, 3, '2-3h'), (3, 5, '3-5h'), (5, 8, '5-8h'), (8, 13, '8h+')]:
    sub = filled[(filled['chop_hrs'] >= lo) & (filled['chop_hrs'] < hi)]
    if len(sub) == 0:
        continue
    tp_c = (sub['outcome'] == 'TP').sum()
    sl_c = (sub['outcome'] == 'SL').sum()
    wr = tp_c / (tp_c + sl_c) * 100 if (tp_c + sl_c) > 0 else 0
    print(f"  {lbl:5s}: {len(sub):3d} trades | WR={wr:5.1f}% | PnL={sub['pnl'].sum():+.2f}%")

# Swing count buckets
print("\n--- By Swing Count ---")
for lo, hi, lbl in [(3, 10, '3-9'), (10, 20, '10-19'), (20, 40, '20-39'), (40, 100, '40+')]:
    sub = filled[(filled['swings'] >= lo) & (filled['swings'] < hi)]
    if len(sub) == 0:
        continue
    tp_c = (sub['outcome'] == 'TP').sum()
    sl_c = (sub['outcome'] == 'SL').sum()
    wr = tp_c / (tp_c + sl_c) * 100 if (tp_c + sl_c) > 0 else 0
    print(f"  {lbl:5s}: {len(sub):3d} trades | WR={wr:5.1f}% | PnL={sub['pnl'].sum():+.2f}%")

# Channel width buckets
print("\n--- By Channel Width ---")
for lo, hi, lbl in [(0.8, 1.0, '<1%'), (1.0, 1.5, '1-1.5%'), (1.5, 2.5, '1.5-2.5%'), (2.5, 4.0, '2.5%+')]:
    sub = filled[(filled['ch_pct'] >= lo) & (filled['ch_pct'] < hi)]
    if len(sub) == 0:
        continue
    tp_c = (sub['outcome'] == 'TP').sum()
    sl_c = (sub['outcome'] == 'SL').sum()
    wr = tp_c / (tp_c + sl_c) * 100 if (tp_c + sl_c) > 0 else 0
    print(f"  {lbl:6s}: {len(sub):3d} trades | WR={wr:5.1f}% | PnL={sub['pnl'].sum():+.2f}%")
