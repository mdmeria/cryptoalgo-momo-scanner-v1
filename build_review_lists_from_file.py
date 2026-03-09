#!/usr/bin/env python3
"""Build pass and near-miss review lists from a specified full scan CSV."""

from __future__ import annotations

import argparse
import os
import pandas as pd

# NOTE: balanced_momo_2h_grind_on removed from gating checks (diagnostic only)
CHECKS = [
    "slow_grind_approach",
    "left_side_staircase",
    "volume_not_decreasing",
    "not_choppy",
    # "balanced_momo_2h_grind_on",  # removed - diagnostic only
    "day_change_ok",
    "vwap_side_ok",
    "first_2h_prev_day_vwap_ok",
    "entry_not_crossed_6h",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source full CSV")
    parser.add_argument("--prefix", required=True, help="Output filename prefix")
    args = parser.parse_args()

    df = pd.read_csv(args.src)
    for c in CHECKS:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.lower().eq("true")

    work = df.copy()
    work["failed_checks_count"] = (~work[CHECKS]).sum(axis=1)

    def failed_labels(row: pd.Series) -> str:
        fails = [c for c in CHECKS if not bool(row[c])]
        return "|".join(fails)

    work["failed_checks"] = work.apply(failed_labels, axis=1)

    passes = work[work["failed_checks_count"] == 0].copy()
    near_1 = work[work["failed_checks_count"] == 1].copy()
    near_2 = work[work["failed_checks_count"] == 2].copy()

    out_dir = os.path.dirname(args.src) or "."
    pass_path = os.path.join(out_dir, f"{args.prefix}_passes.csv")
    near1_path = os.path.join(out_dir, f"{args.prefix}_near_miss_1_fail.csv")
    near2_path = os.path.join(out_dir, f"{args.prefix}_near_miss_2_fail.csv")

    passes.to_csv(pass_path, index=False, encoding="utf-8-sig")
    near_1.to_csv(near1_path, index=False, encoding="utf-8-sig")
    near_2.to_csv(near2_path, index=False, encoding="utf-8-sig")

    print(f"rows_total={len(work)}")
    print(f"passes={len(passes)}")
    print(f"near_miss_1_fail={len(near_1)}")
    print(f"near_miss_2_fail={len(near_2)}")
    print(f"pass_path={pass_path}")
    print(f"near1_path={near1_path}")
    print(f"near2_path={near2_path}")


if __name__ == "__main__":
    main()
