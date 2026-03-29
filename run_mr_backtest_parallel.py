#!/usr/bin/env python3
"""Parallel backtest for MR strategy with runtime logging."""
import sys, os, time, pandas as pd, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scan_mean_reversion import MRSettings, check_mr_gates_at_bar

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_DIR = Path("datasets/futures_jan17_mar29")
MAX_BARS = 120
LIMIT_EXPIRY = 3
OUTPUT_CSV = "mr_v2_aligned3_futures_results.csv"
LOG_FILE = "mr_v2_aligned3_futures_log.txt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def process_symbol(sym, dataset_dir, warmup):
    """Process one symbol, return list of trade dicts."""
    cfg = MRSettings()
    fpath = dataset_dir / f"{sym}_1m.csv"
    df = pd.read_csv(str(fpath), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) < warmup + 100:
        return []

    scan_start = warmup
    scan_end = len(df) - MAX_BARS - LIMIT_EXPIRY
    if scan_start >= scan_end:
        return []

    trades = []
    cooldown = -1
    for i in range(scan_start, scan_end):
        if i > cooldown:
            r = check_mr_gates_at_bar(df, i, cfg)
            if r["passed"]:
                side, level, tp, sl = r["side"], r["entry"], r["tp"], r["sl"]
                fill_bar = None
                for fb in range(i + 1, min(i + 1 + LIMIT_EXPIRY, len(df))):
                    if side == "long" and df.iloc[fb]["low"] <= level:
                        fill_bar = fb; break
                    elif side == "short" and df.iloc[fb]["high"] >= level:
                        fill_bar = fb; break

                if fill_bar is None:
                    trades.append({
                        "symbol": sym, "ts": str(df.iloc[i + 1]["timestamp"]),
                        "side": side, "entry": round(level, 8),
                        "tp": round(tp, 8), "sl": round(sl, 8),
                        "tp_pct": round(r["tp_pct"], 3), "sl_pct": round(r["sl_pct"], 3),
                        "dps": r["dps_total"], "conf": r["dps_confidence"],
                        "touches": r["touches"], "outcome": "MISSED", "pnl": 0,
                    })
                    cooldown = i + cfg.cooldown_bars
                    continue

                bars = df.iloc[fill_bar:fill_bar + MAX_BARS]
                outcome = "OPEN"
                for _, bar in bars.iterrows():
                    if side == "long":
                        if bar["low"] <= sl: outcome = "SL"; break
                        if bar["high"] >= tp: outcome = "TP"; break
                    else:
                        if bar["high"] >= sl: outcome = "SL"; break
                        if bar["low"] <= tp: outcome = "TP"; break
                if outcome == "TP":
                    pnl = r["tp_pct"]
                elif outcome == "SL":
                    pnl = -r["sl_pct"]
                else:
                    lc = float(bars.iloc[-1]["close"])
                    pnl = (lc - level) / level * 100 if side == "long" else (level - lc) / level * 100

                trades.append({
                    "symbol": sym, "ts": str(df.iloc[fill_bar]["timestamp"]),
                    "side": side, "entry": round(level, 8),
                    "tp": round(tp, 8), "sl": round(sl, 8),
                    "tp_pct": round(r["tp_pct"], 3), "sl_pct": round(r["sl_pct"], 3),
                    "dps": r["dps_total"], "conf": r["dps_confidence"],
                    "touches": r["touches"], "outcome": outcome, "pnl": round(pnl, 3),
                })
                cooldown = i + cfg.cooldown_bars
    return trades


def main():
    # Clear log
    with open(LOG_FILE, "w") as f:
        f.write("")

    cfg = MRSettings()
    warmup = max(cfg.range_max_bars, cfg.noise_lookback_bars, 720)

    syms = sorted([f.replace("_1m.csv", "") for f in os.listdir(DATASET_DIR) if f.endswith("_1m.csv")])
    log(f"Starting: {len(syms)} symbols, dataset={DATASET_DIR}")

    all_trades = []
    done = 0
    t0 = time.time()
    workers = min(8, os.cpu_count() or 4)
    log(f"Using {workers} parallel workers")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_symbol, sym, DATASET_DIR, warmup): sym for sym in syms}
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            try:
                trades = future.result()
                if trades:
                    all_trades.extend(trades)
                    # Write trades to CSV incrementally
                    tdf = pd.DataFrame(all_trades)
                    tdf.to_csv(OUTPUT_CSV, index=False)
                    for t in trades:
                        log(f"  TRADE {sym} {t['ts']} {t['side']} entry={t['entry']} outcome={t['outcome']} pnl={t['pnl']}")
            except Exception as e:
                log(f"  ERROR {sym}: {e}")

            if done % 25 == 0 or done == len(syms):
                elapsed = time.time() - t0
                log(f"Progress: {done}/{len(syms)} symbols, {len(all_trades)} trades, {elapsed:.0f}s elapsed")

    # Final save
    tdf = pd.DataFrame(all_trades)
    tdf.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.0f}s")

    if len(tdf) == 0:
        log("No trades found!")
        return

    filled = tdf[tdf["outcome"] != "MISSED"]
    tp_n = (filled["outcome"] == "TP").sum()
    sl_n = (filled["outcome"] == "SL").sum()
    op_n = (filled["outcome"] == "OPEN").sum()
    wr = tp_n / (tp_n + sl_n) * 100 if (tp_n + sl_n) > 0 else 0

    log(f"Signals: {len(tdf)} | Filled: {len(filled)} | Missed: {len(tdf) - len(filled)}")
    log(f"Filled: {len(filled)} | TP={tp_n} SL={sl_n} OPEN={op_n} | WR={wr:.1f}% | PnL={filled['pnl'].sum():+.2f}%")

    for side in ["long", "short"]:
        sub = filled[filled["side"] == side]
        if len(sub) == 0: continue
        tp = (sub["outcome"] == "TP").sum()
        sl = (sub["outcome"] == "SL").sum()
        wr_s = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        log(f"  {side:5s}: {len(sub)} | WR={wr_s:.1f}% | PnL={sub['pnl'].sum():+.2f}%")

    log(f"\nBy DPS:")
    for dps in sorted(filled["dps"].unique()):
        sub = filled[filled["dps"] == dps]
        tp = (sub["outcome"] == "TP").sum()
        sl = (sub["outcome"] == "SL").sum()
        wr_d = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        log(f"  DPS {dps}: {len(sub):3d} | WR={wr_d:.1f}% | PnL={sub['pnl'].sum():+.2f}%")

    log(f"\nBy Confidence:")
    for conf in ["avoid", "low", "high", "max"]:
        sub = filled[filled["conf"] == conf]
        if len(sub) == 0: continue
        tp = (sub["outcome"] == "TP").sum()
        sl = (sub["outcome"] == "SL").sum()
        wr_c = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
        log(f"  {conf:6s}: {len(sub):3d} | WR={wr_c:.1f}% | PnL={sub['pnl'].sum():+.2f}%")

    log(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
