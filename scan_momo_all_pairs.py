import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from momentum_quality import evaluate_momentum_setup

BASE = "https://data-api.binance.vision/api/v3"


def infer_direction(df: pd.DataFrame) -> str:
    sm30 = df["close"].ewm(alpha=1 / 30, adjust=False).mean().iloc[-1]
    sm120 = df["close"].ewm(alpha=1 / 120, adjust=False).mean().iloc[-1]
    close = float(df["close"].iloc[-1])

    if close > sm30 > sm120:
        return "long"
    if close < sm30 < sm120:
        return "short"
    return "long"


def fetch_usdt_pairs() -> list[str]:
    ex = requests.get(f"{BASE}/exchangeInfo", timeout=20).json()
    symbols = [
        s["symbol"]
        for s in ex.get("symbols", [])
        if s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed")
    ]
    return symbols


def evaluate_symbol(symbol: str):
    try:
        response = requests.get(
            f"{BASE}/klines",
            params={"symbol": symbol, "interval": "1m", "limit": 500},
            timeout=6,
        )
        if response.status_code != 200:
            return "skip", symbol, None

        raw = response.json()
        if not isinstance(raw, list) or len(raw) < 120:
            return "skip", symbol, None

        df = pd.DataFrame(
            [
                {
                    "timestamp": pd.to_datetime(k[0], unit="ms"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
                for k in raw
            ]
        ).set_index("timestamp")

        direction = infer_direction(df)
        quality = evaluate_momentum_setup(
            df,
            direction=direction,
            min_quality_score=0.60,
            symbol=symbol,
            enforce_extended_rules=True,
        )

        if quality.passed:
            return "pass", symbol, {
                "direction": direction,
                "score": quality.score,
                "checks": quality.checks,
                "metrics": quality.metrics,
            }

        return "fail", symbol, {
            "direction": direction,
            "score": quality.score,
            "checks": quality.checks,
        }
    except Exception:
        return "skip", symbol, None


def main():
    symbols = fetch_usdt_pairs()
    print(f"total_usdt_pairs {len(symbols)}")

    passes = []
    fails = 0
    skips = 0

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(evaluate_symbol, symbol) for symbol in symbols]
        for future in as_completed(futures):
            status, symbol, data = future.result()
            if status == "pass":
                passes.append((symbol, data))
            elif status == "fail":
                fails += 1
            else:
                skips += 1

    passes.sort(key=lambda x: x[1]["score"], reverse=True)

    print(f"evaluated_fails {fails}")
    print(f"skipped {skips}")
    print(f"passed_count {len(passes)}")

    for symbol, data in passes[:100]:
        metrics = data["metrics"]
        checks = data["checks"]
        print(
            f"{symbol:<14} "
            f"dir={data['direction']:<5} "
            f"score={data['score']:.2f} "
            f"staircase={checks['left_side_staircase']} "
            f"approach={checks['slow_grind_approach']} "
            f"volume={checks['volume_not_decreasing']} "
            f"not_choppy={checks['not_choppy']} "
            f"day_change={checks['day_change_ok']} "
            f"vwap_side={checks['vwap_side_ok']} "
            f"entry_fresh_6h={checks['entry_not_crossed_6h']} "
            f"stack={int(metrics['trend_stack_bars_120'])}/120 "
            f"stair={int(metrics['staircase_bars_120'])}/120"
        )


if __name__ == "__main__":
    main()
