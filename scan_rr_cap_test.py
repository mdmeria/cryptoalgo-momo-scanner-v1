#!/usr/bin/env python3
"""Test: what if we capped all RR > 1.5 trades to 1.25 RR?"""
import csv, os
import numpy as np

DATA_DIRS = ['c:/Projects/CryptoAlgo/datasets/live/candles_1m']

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

def find_range(h, l, c, s, e):
    hh, ll, cc = h[s:e], l[s:e], c[s:e]
    if len(hh) < 60: return None
    u = np.percentile(hh, 92); lo = np.percentile(ll, 8)
    rng = (u-lo)/lo*100
    if rng < 0.8 or rng > 5.0: return None
    ins = np.mean((cc >= lo*0.998) & (cc <= u*1.002))*100
    if ins < 65: return None
    return u, lo

def find_swings(h, l, c, level, s, e, rw, side):
    mt = rw * 0.4; swings = []; near = False; far = 0
    for j in range(s, e):
        at = abs(h[j]-level)/level*100<0.4 or abs(l[j]-level)/level*100<0.4 or abs(c[j]-level)/level*100<0.3
        d = (level-l[j]) if side=='short' else (h[j]-level)
        if at:
            if not near and far >= mt: swings.append({'bar':j,'depth':far})
            near=True; far=0
        else:
            near=False
            if d>far: far=d
    swings.reverse(); return swings

def sim_outcome(h, l, entry, sl, tp, side, start, n):
    fb = min(120, n - start - 1)
    if fb < 10: return "OPEN"
    rd = abs(entry-sl)
    t09 = entry-rd*0.9 if side=='short' else entry+rd*0.9
    tsl = entry+rd*0.1 if side=='short' else entry-rd*0.1
    trailed=False; csl=sl
    for k in range(start+1, start+fb+1):
        if not trailed:
            if side=='short' and l[k]<=t09: trailed=True; csl=tsl
            elif side=='long' and h[k]>=t09: trailed=True; csl=tsl
        if side=='short':
            if l[k]<=tp: return "TP"
            if h[k]>=csl: return "TRAIL_SL" if trailed else "SL"
        else:
            if h[k]>=tp: return "TP"
            if l[k]<=csl: return "TRAIL_SL" if trailed else "SL"
    return "OPEN"

files = []
for d in DATA_DIRS:
    if os.path.exists(d):
        for f in sorted(os.listdir(d)):
            if f.endswith('.csv'): files.append((d,f))

trades = []
seen = set()
for dd, fn in files:
    sym = fn.replace('_1m.csv','')
    if sym in seen: continue
    seen.add(sym)
    try: data = load_sym(os.path.join(dd,fn))
    except: continue
    if data is None: continue
    c,h,l,ts = data['c'],data['h'],data['l'],data['ts']
    n = len(c); last_s = -60
    for i in range(600, n-30, 10):
        if i-last_s<60: continue
        curr = None
        for dur in [180,360]:
            r = find_range(h,l,c,i-dur,i)
            if r: curr={'u':r[0],'l':r[1],'s':i-dur}; break
        if not curr: continue
        rw = curr['u']-curr['l']
        prev = None
        for dur in [180,360]:
            ps=curr['s']-dur
            if ps<0: continue
            r = find_range(h,l,c,ps,curr['s'])
            if r: prev={'s':ps,'u':r[0],'l':r[1]}; break
        if not prev: continue
        cm=(curr['u']+curr['l'])/2; pm=(prev['u']+prev['l'])/2
        sp=(cm-pm)/pm*100
        if abs(sp)<0.5: continue
        if sp<0: side,el='short',curr['u']
        else: side,el='long',curr['l']
        if side=='short' and h[i]<el*0.997: continue
        if side=='long' and l[i]>el*1.003: continue
        if side=='short' and c[i]>=el: continue
        if side=='long' and c[i]<=el: continue
        sw = find_swings(h,l,c,el,curr['s'],i,rw,side)
        if len(sw)<3: continue
        if i-sw[0]['bar']>30: continue
        depths = [s['depth'] for s in sw[:3]]
        td = min(depths)
        entry = el
        sl_v = el*1.005 if side=='short' else el*0.995
        sl_d = abs(entry-sl_v)
        if side=='short': orig_tp = entry-td*0.95; orig_tp_p=(entry-orig_tp)/entry*100
        else: orig_tp = entry+td*0.95; orig_tp_p=(orig_tp-entry)/entry*100
        sl_p = sl_d/entry*100
        orig_rr = orig_tp_p/sl_p if sl_p>0 else 0
        if orig_rr<1.0 or orig_tp_p<0.5: continue

        # Capped TP at 1.25 RR
        cap_d = sl_d * 1.25
        cap_tp = entry-cap_d if side=='short' else entry+cap_d

        orig_out = sim_outcome(h,l,entry,sl_v,orig_tp,side,i,n)
        cap_out = sim_outcome(h,l,entry,sl_v,cap_tp,side,i,n)

        trades.append({
            'sym':sym,'side':side,'ts':ts[i],
            'orig_rr':round(orig_rr,2),'orig_out':orig_out,
            'cap_out':cap_out,
        })
        last_s = i

def wr(lst, key='orig_out'):
    t=sum(1 for r in lst if r[key]=='TP')
    s=sum(1 for r in lst if r[key] in ('SL','TRAIL_SL'))
    return t,s,t/(t+s)*100 if t+s>0 else 0

print(f"Total trades (3+ swings, <=30min): {len(trades)}")
t1,s1,w1=wr(trades,'orig_out')
t2,s2,w2=wr(trades,'cap_out')
print(f"  Original TP:  TP={t1} SL={s1} WR={w1:.1f}%")
print(f"  Capped 1.25R: TP={t2} SL={s2} WR={w2:.1f}%")

print("\nBy original RR bucket:")
for a,b,lb in [(1.0,1.5,'1.0-1.5'),(1.5,2.0,'1.5-2.0'),(2.0,3.0,'2.0-3.0'),(3.0,99,'3.0+')]:
    sub = [r for r in trades if a<=r['orig_rr']<b]
    if not sub: continue
    t1,s1,w1=wr(sub,'orig_out')
    t2,s2,w2=wr(sub,'cap_out')
    print(f"  RR {lb} ({len(sub)} trades):")
    print(f"    Original: TP={t1} SL={s1} WR={w1:.1f}%")
    print(f"    Capped:   TP={t2} SL={s2} WR={w2:.1f}%")
    # How many flipped from SL to TP?
    flipped = sum(1 for r in sub if r['orig_out'] in ('SL','TRAIL_SL') and r['cap_out']=='TP')
    lost = sum(1 for r in sub if r['orig_out']=='TP' and r['cap_out'] in ('SL','TRAIL_SL'))
    print(f"    Flipped SL->TP: {flipped}, Lost TP->SL: {lost}")
