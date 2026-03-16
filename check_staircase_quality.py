"""
Check ZCT staircase grind quality for 120 bars before trade entry.
"""

import pandas as pd
import numpy as np

TRADES = [
    # Winners
    {"symbol": "BCHUSDT",    "entry": "2026-03-13 00:26", "side": "long",  "result": "TP"},
    {"symbol": "XPLUSDT",    "entry": "2026-03-03 05:13", "side": "long",  "result": "TP"},
    {"symbol": "XLMUSDT",    "entry": "2026-03-10 14:49", "side": "long",  "result": "TP"},
    {"symbol": "DASHUSDT",   "entry": "2026-03-04 08:52", "side": "long",  "result": "TP"},
    {"symbol": "RENDERUSDT", "entry": "2026-03-11 12:38", "side": "long",  "result": "TP"},
    {"symbol": "SEIUSDT",    "entry": "2026-03-02 15:00", "side": "long",  "result": "TP"},
    # Losers
    {"symbol": "KITEUSDT",   "entry": "2026-03-09 16:09", "side": "short", "result": "SL"},
    {"symbol": "ICPUSDT",    "entry": "2026-03-12 14:10", "side": "long",  "result": "SL"},
    {"symbol": "RESOLVUSDT", "entry": "2026-03-10 00:57", "side": "short", "result": "SL"},
    {"symbol": "OPUSDT",     "entry": "2026-03-03 05:28", "side": "long",  "result": "SL"},
]

DATA_DIR = "datasets/momo_1m_mar2_mar14"
LOOKBACK = 120  # 2 hours of 1-min bars


def compute_smma30(close_series):
    """SMMA30 approximated as EWM with alpha=1/30."""
    return close_series.ewm(alpha=1/30, adjust=False).mean()


def analyse_staircase(df_window, side):
    """Compute staircase metrics on a 120-bar window."""
    o = df_window["open"].values
    h = df_window["high"].values
    l = df_window["low"].values
    c = df_window["close"].values

    # 1. Directional consistency
    if side == "long":
        same_color = np.sum(c > o) / len(c)
    else:
        same_color = np.sum(c < o) / len(c)

    # 2. Average wick ratio  (total_wick / candle_range)
    candle_range = h - l
    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)
    upper_wick = h - body_top
    lower_wick = body_bot - l
    total_wick = upper_wick + lower_wick
    # avoid div-by-zero for zero-range candles
    mask = candle_range > 0
    safe_range = np.where(mask, candle_range, 1.0)
    wick_ratios = np.where(mask, total_wick / safe_range, 0.0)
    avg_wick_ratio = np.mean(wick_ratios)

    # 3. Net move %
    net_move_pct = (c[-1] - c[0]) / c[0] * 100.0
    if side == "short":
        net_move_pct = -net_move_pct  # positive = favourable direction

    # 4. SMMA30 crosses — need a longer series for warm-up
    # We already receive only the window; compute SMMA on it and count crosses
    smma = compute_smma30(pd.Series(c)).values
    above = c > smma
    crosses = int(np.sum(np.diff(above.astype(int)) != 0))

    # 5. Smoothness — std of per-bar returns
    returns = np.diff(c) / c[:-1]
    ret_std = np.std(returns) * 100.0  # in pct-points

    return {
        "dir_consist": same_color,
        "avg_wick_ratio": avg_wick_ratio,
        "net_move_pct": net_move_pct,
        "smma_crosses": crosses,
        "ret_std": ret_std,
    }


def staircase_score(m):
    """Combine metrics into a 0-100 score."""
    # Directional consistency: 50% neutral, 80%+ great  (weight 30)
    s_dir = np.clip((m["dir_consist"] - 0.40) / 0.40, 0, 1) * 30

    # Low wick: lower is better; 0.3 is decent, 0.0 is perfect  (weight 20)
    s_wick = np.clip(1.0 - m["avg_wick_ratio"], 0, 1) * 20

    # Net move: 0-3% mapped to 0-1  (weight 20)
    s_move = np.clip(m["net_move_pct"] / 3.0, 0, 1) * 20

    # Low crosses: 0 = perfect, 10+ = 0  (weight 20)
    s_cross = np.clip(1.0 - m["smma_crosses"] / 10.0, 0, 1) * 20

    # Smoothness: lower std = better; 0.02 pct-std = perfect, 0.15+ = 0  (weight 10)
    s_smooth = np.clip(1.0 - (m["ret_std"] - 0.02) / 0.13, 0, 1) * 10

    return s_dir + s_wick + s_move + s_cross + s_smooth


def main():
    results = []
    for t in TRADES:
        fname = f"{DATA_DIR}/{t['symbol']}_1m.csv"
        df = pd.read_csv(fname, parse_dates=["timestamp"])
        # strip timezone if present
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        entry_ts = pd.Timestamp(t["entry"])
        # find closest bar at or before entry
        mask = df["timestamp"] <= entry_ts
        if mask.sum() == 0:
            print(f"WARNING: no data before entry for {t['symbol']}")
            continue
        entry_idx = df.loc[mask].index[-1]

        start_idx = max(entry_idx - LOOKBACK, 0)
        window = df.iloc[start_idx:entry_idx].copy()

        if len(window) < 60:
            print(f"WARNING: only {len(window)} bars before entry for {t['symbol']}")

        metrics = analyse_staircase(window, t["side"])
        score = staircase_score(metrics)

        results.append({
            "symbol": t["symbol"],
            "side": t["side"],
            "result": t["result"],
            "bars": len(window),
            "green_pct": f"{metrics['dir_consist']*100:.1f}%",
            "wick_ratio": f"{metrics['avg_wick_ratio']:.3f}",
            "net_move": f"{metrics['net_move_pct']:.2f}%",
            "smma_xings": metrics["smma_crosses"],
            "ret_std": f"{metrics['ret_std']:.4f}",
            "score": f"{score:.1f}",
        })

    # Print summary
    rdf = pd.DataFrame(results)
    print("\n=== ZCT STAIRCASE QUALITY — 120 bars (2h) before entry ===\n")
    print(rdf.to_string(index=False))

    # Averages by result
    scores = [float(r["score"]) for r in results]
    tp_scores = [float(r["score"]) for r in results if r["result"] == "TP"]
    sl_scores = [float(r["score"]) for r in results if r["result"] == "SL"]
    print(f"\n--- Averages ---")
    print(f"  TP winners avg score: {np.mean(tp_scores):.1f}  (n={len(tp_scores)})")
    print(f"  SL losers  avg score: {np.mean(sl_scores):.1f}  (n={len(sl_scores)})")
    print(f"  All trades avg score: {np.mean(scores):.1f}")


if __name__ == "__main__":
    main()
