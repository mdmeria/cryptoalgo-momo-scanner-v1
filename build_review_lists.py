#!/usr/bin/env python3
"""Build pass and near-miss review lists from full scan output."""

from __future__ import annotations

import pandas as pd

SRC = r"C:\Projects\CryptoAlgo\binance_30min_last10_full_full.csv"
OUT_DIR = r"C:\Projects\CryptoAlgo"

# Must match scanner's overall_pass_grind_on logic.
CHECKS = [
    "slow_grind_approach",
    "left_side_staircase",
    "volume_not_decreasing",
    "not_choppy",
    "balanced_momo_2h_grind_on",
    "day_change_ok",
    "vwap_side_ok",
    "first_2h_prev_day_vwap_ok",
    "entry_not_crossed_6h",
]

df = pd.read_csv(SRC)

for c in CHECKS:
    # Normalize to bool in case CSV stores strings.
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.lower().eq("true")

failed_counts = (~df[CHECKS]).sum(axis=1)

# Add labels for manual triage.
def failed_labels(row: pd.Series) -> str:
    fails = [c for c in CHECKS if not bool(row[c])]
    return "|".join(fails)

work = df.copy()
work["failed_checks_count"] = failed_counts
work["failed_checks"] = work.apply(failed_labels, axis=1)

passes = work[work["failed_checks_count"] == 0].copy()
near_1 = work[work["failed_checks_count"] == 1].copy()
near_2 = work[work["failed_checks_count"] == 2].copy()

# Save review files.
passes_path = f"{OUT_DIR}\\review_passes.csv"
near1_path = f"{OUT_DIR}\\review_near_miss_1_fail.csv"
near2_path = f"{OUT_DIR}\\review_near_miss_2_fail.csv"

passes.to_csv(passes_path, index=False, encoding="utf-8-sig")
near_1.to_csv(near1_path, index=False, encoding="utf-8-sig")
near_2.to_csv(near2_path, index=False, encoding="utf-8-sig")

print("Created:")
print(passes_path)
print(near1_path)
print(near2_path)
print()
print(f"rows_total={len(work)}")
print(f"passes={len(passes)}")
print(f"near_miss_1_fail={len(near_1)}")
print(f"near_miss_2_fail={len(near_2)}")

if len(passes) > 0:
    print("\nTop pass symbols:")
    print(passes["symbol"].value_counts().head(15).to_string())

print("\nTop near-miss (1 fail) failed checks:")
print(near_1["failed_checks"].value_counts().head(15).to_string())

print("\nTop near-miss (2 fail) failed-check pairs:")
print(near_2["failed_checks"].value_counts().head(15).to_string())
