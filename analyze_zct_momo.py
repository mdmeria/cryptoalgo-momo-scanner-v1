#!/usr/bin/env python3
"""Independent analysis of ZCT Momo v3 results."""
import pandas as pd, numpy as np

df = pd.read_csv('zct_momo_results.csv')
filled = df[df['outcome']!='MISSED'].copy()
filled['ts_dt'] = pd.to_datetime(filled['ts'])
filled['hour'] = filled['ts_dt'].dt.hour
filled['dow'] = filled['ts_dt'].dt.dayofweek
filled['month'] = filled['ts_dt'].dt.month
filled['date'] = filled['ts_dt'].dt.date
filled['year'] = filled['ts_dt'].dt.year
closed = filled[filled['outcome'].isin(['TP','SL'])]

def wr_pnl(sub_closed, sub_all):
    tp = (sub_closed['outcome']=='TP').sum()
    sl = (sub_closed['outcome']=='SL').sum()
    wr = tp/(tp+sl)*100 if (tp+sl)>0 else 0
    return len(sub_closed), tp, sl, wr, sub_all['pnl'].sum()

print('='*70)
print('INDEPENDENT ANALYSIS — ZCT MOMO v3')
print('='*70)

# 1. DAY OF WEEK
print('\n=== 1. DAY OF WEEK ===')
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
for dow in range(7):
    sc = closed[closed['dow']==dow]
    sa = filled[filled['dow']==dow]
    if len(sc) < 10: continue
    n,tp,sl,wr,pnl = wr_pnl(sc, sa)
    print(f'  {days[dow]:3s}: {n:4d} closed | WR={wr:.1f}% | PnL={pnl:+.2f}%')

# 2. MONTH
print('\n=== 2. MONTH OF YEAR ===')
for m in range(1,13):
    sc = closed[closed['month']==m]
    sa = filled[filled['month']==m]
    if len(sc) < 20: continue
    n,tp,sl,wr,pnl = wr_pnl(sc, sa)
    print(f'  Month {m:2d}: {n:4d} | WR={wr:.1f}% | PnL={pnl:+.2f}%')

# 3. DURATION
print('\n=== 3. STAIRCASE DURATION ===')
for lo, hi, lbl in [(2.0,2.5,'2.0-2.5h'),(2.5,3.0,'2.5-3.0h'),(3.0,4.0,'3.0-4.0h'),(4.0,20,'4.0h+')]:
    sc = closed[(closed['dur_hrs']>=lo) & (closed['dur_hrs']<hi)]
    sa = filled[(filled['dur_hrs']>=lo) & (filled['dur_hrs']<hi)]
    if len(sc) < 10: continue
    n,tp,sl,wr,pnl = wr_pnl(sc, sa)
    print(f'  {lbl:10s}: {n:4d} | WR={wr:.1f}% | PnL={pnl:+.2f}%')

# 4. FIRST SIGNAL vs SUBSEQUENT
print('\n=== 4. FIRST SIGNAL OF DAY vs SUBSEQUENT ===')
filled_s = filled.sort_values('ts')
filled_s['is_first'] = filled_s.groupby(['symbol','date']).cumcount() == 0
first_idx = filled_s[filled_s['is_first']].index
first_c = closed[closed.index.isin(first_idx)]
sub_c = closed[~closed.index.isin(first_idx)]
first_a = filled[filled.index.isin(first_idx)]
sub_a = filled[~filled.index.isin(first_idx)]
n1,t1,s1,w1,p1 = wr_pnl(first_c, first_a)
n2,t2,s2,w2,p2 = wr_pnl(sub_c, sub_a)
print(f'  First:      {n1:4d} | WR={w1:.1f}% | PnL={p1:+.2f}%')
print(f'  Subsequent: {n2:4d} | WR={w2:.1f}% | PnL={p2:+.2f}%')

# 5. TP DISTANCE
print('\n=== 5. TP DISTANCE ===')
for lo, hi, lbl in [(1.0,1.15,'1.0-1.15%'),(1.15,1.5,'1.15-1.5%'),(1.5,2.5,'1.5-2.5%'),(2.5,5.0,'2.5%+')]:
    sc = closed[(closed['tp_pct']>=lo) & (closed['tp_pct']<hi)]
    sa = filled[(filled['tp_pct']>=lo) & (filled['tp_pct']<hi)]
    if len(sc) < 10: continue
    n,tp,sl,wr,pnl = wr_pnl(sc, sa)
    print(f'  TP {lbl:10s}: {n:4d} | WR={wr:.1f}% | PnL={pnl:+.2f}%')

# 6. VOLUME TREND
print('\n=== 6. VOLUME TREND ===')
for vt in sorted(closed['vol_trend'].dropna().unique()):
    sc = closed[closed['vol_trend']==vt]
    sa = filled[filled['vol_trend']==vt]
    if len(sc) < 10: continue
    n,tp,sl,wr,pnl = wr_pnl(sc, sa)
    print(f'  {str(vt):12s}: {n:4d} | WR={wr:.1f}% | PnL={pnl:+.2f}%')

# 7. SIGNALS PER HOUR
print('\n=== 7. SIGNALS PER HOUR (crowded vs isolated) ===')
filled_s['hour_key'] = filled_s['ts_dt'].dt.strftime('%Y-%m-%d-%H')
hcounts = filled_s.groupby('hour_key').size()
filled_s['sigs_hr'] = filled_s['hour_key'].map(hcounts)
for lo, hi, lbl in [(1,1,'Isolated(1)'),(2,3,'2-3 sigs'),(4,50,'4+ sigs')]:
    mask = (filled_s['sigs_hr']>=lo) & (filled_s['sigs_hr']<=hi)
    sc = closed[closed.index.isin(filled_s[mask].index)]
    sa = filled[filled.index.isin(filled_s[mask].index)]
    if len(sc) < 10: continue
    n,tp,sl,wr,pnl = wr_pnl(sc, sa)
    print(f'  {lbl:15s}: {n:4d} | WR={wr:.1f}% | PnL={pnl:+.2f}%')

# 8. BEST COMBOS
print('\n=== 8. BEST COMBOS ===')
combos = [
    ('US open shorts DPS4+', (closed['hour'].between(13,15)) & (closed['side']=='short') & (closed['dps']>=4)),
    ('US open longs DPS4+', (closed['hour'].between(13,15)) & (closed['side']=='long') & (closed['dps']>=4)),
    ('Asian shorts DPS4+', (closed['hour'].between(1,5)) & (closed['side']=='short') & (closed['dps']>=4)),
    ('Asian longs DPS4+', (closed['hour'].between(1,5)) & (closed['side']=='long') & (closed['dps']>=4)),
    ('Mon-Wed', closed['dow'].between(0,2)),
    ('Thu-Sun', closed['dow'].between(3,6)),
    ('Swing SL + US open', (closed['sl_pct']>1.0) & (closed['hour'].between(13,15))),
    ('Swing SL + Asian', (closed['sl_pct']>1.0) & (closed['hour'].between(1,5))),
    ('Default SL + US open', (closed['sl_pct']==1.0) & (closed['hour'].between(13,15))),
    ('Isolated + DPS5', (closed.index.isin(filled_s[filled_s['sigs_hr']==1].index)) & (closed['dps']==5)),
    ('First sig + swing SL', (closed.index.isin(first_idx)) & (closed['sl_pct']>1.0)),
    ('Dur 2-2.5h + DPS5', (closed['dur_hrs'].between(2.0,2.5)) & (closed['dps']==5)),
    ('Dur 2.5-3h + DPS4+', (closed['dur_hrs'].between(2.5,3.0)) & (closed['dps']>=4)),
    ('Short + score-3 + swing SL', (closed['side']=='short') & (closed['mkt_score']==-3) & (closed['sl_pct']>1.0)),
    ('Long + score+3 + swing SL', (closed['side']=='long') & (closed['mkt_score']==3) & (closed['sl_pct']>1.0)),
    ('US open + first sig', (closed['hour'].between(13,15)) & (closed.index.isin(first_idx))),
    ('Asian + isolated + DPS4+', (closed['hour'].between(1,5)) & (closed.index.isin(filled_s[filled_s['sigs_hr']==1].index)) & (closed['dps']>=4)),
]

for label, mask in combos:
    sc = closed[mask]
    if len(sc) < 5: continue
    tp=(sc['outcome']=='TP').sum(); sl=(sc['outcome']=='SL').sum()
    wr=tp/(tp+sl)*100 if (tp+sl)>0 else 0
    pnl = filled[mask & filled['outcome'].isin(['TP','SL','TRAIL_SL','OPEN'])]['pnl'].sum() if len(filled[mask])>0 else 0
    print(f'  {label:35s}: {len(sc):4d} | WR={wr:.1f}% | PnL={pnl:+.2f}%')

# 9. BY YEAR
print('\n=== 9. BY YEAR ===')
for yr in sorted(filled['year'].unique()):
    sc = closed[closed['ts_dt'].dt.year==yr]
    sa = filled[filled['year']==yr]
    if len(sc) < 10: continue
    n,tp,sl,wr,pnl = wr_pnl(sc, sa)
    print(f'  {yr}: {n:4d} | WR={wr:.1f}% | PnL={pnl:+.2f}%')

# 10. WORST/BEST SYMBOLS
print('\n=== 10. TOP 10 WORST SYMBOLS ===')
sym_pnl = filled.groupby('symbol')['pnl'].sum().sort_values()
for sym, pnl in sym_pnl.head(10).items():
    sc = closed[closed['symbol']==sym]
    n,tp,sl,wr,_ = wr_pnl(sc, sc)
    print(f'  {sym:18s}: {n:3d} | WR={wr:.1f}% | PnL={pnl:+.2f}%')

print('\n=== 10b. TOP 10 BEST SYMBOLS ===')
for sym, pnl in sym_pnl.tail(10).items():
    sc = closed[closed['symbol']==sym]
    n,tp,sl,wr,_ = wr_pnl(sc, sc)
    print(f'  {sym:18s}: {n:3d} | WR={wr:.1f}% | PnL={pnl:+.2f}%')
