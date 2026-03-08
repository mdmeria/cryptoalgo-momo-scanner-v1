from concurrent.futures import ThreadPoolExecutor, as_completed

from monitor_momo_passes import fetch_usdt_pairs, fetch_klines, infer_direction
from momentum_quality import evaluate_momentum_setup


def eval_symbol(symbol: str):
    df = fetch_klines(symbol)
    if df is None:
        return None
    direction = infer_direction(df)
    result = evaluate_momentum_setup(
        df=df,
        direction=direction,
        min_quality_score=0.60,
        symbol=symbol,
        enforce_extended_rules=True,
    )
    return symbol, direction, result


def main() -> None:
    symbols = fetch_usdt_pairs()
    total = 0
    passed = 0
    all_except_balanced = 0
    balanced_false = 0

    examples = []

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(eval_symbol, s) for s in symbols]
        for fut in as_completed(futures):
            out = fut.result()
            if out is None:
                continue
            total += 1
            symbol, direction, r = out
            c = r.checks

            if r.passed:
                passed += 1

            except_balanced_ok = (
                c.get("slow_grind_approach", False)
                and c.get("left_side_staircase", False)
                and c.get("volume_not_decreasing", False)
                and c.get("not_choppy", False)
                and c.get("day_change_ok", False)
                and c.get("vwap_side_ok", False)
                and c.get("first_2h_prev_day_vwap_ok", False)
                and c.get("entry_not_crossed_6h", False)
            )

            if except_balanced_ok and not c.get("balanced_momo_2h", False):
                all_except_balanced += 1
                if len(examples) < 12:
                    examples.append(
                        (
                            symbol,
                            direction,
                            r.metrics.get("dir_move_2h_pct", 0.0),
                            r.metrics.get("dir_bar_ratio_2h", 0.0),
                            r.metrics.get("max_dir_impulse_8m_pct", 0.0),
                            r.metrics.get("efficiency_2h", 0.0),
                        )
                    )

            if not c.get("balanced_momo_2h", False):
                balanced_false += 1

    print(f"evaluated={total} passed={passed}")
    print(f"balanced_false={balanced_false}")
    print(f"all_except_balanced={all_except_balanced}")
    if examples:
        print("examples_failing_only_balanced:")
        for ex in examples:
            print(
                f"  {ex[0]}:{ex[1]} dir_move_2h_pct={ex[2]:.2f} "
                f"dir_bar_ratio_2h={ex[3]:.2f} max_dir_impulse_8m_pct={ex[4]:.2f} efficiency_2h={ex[5]:.2f}"
            )


if __name__ == "__main__":
    main()
