#!/usr/bin/env python3
"""OKX-only dummy trader using full momentum criteria + structural SL/TP."""

from __future__ import annotations

import argparse
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from momentum_quality import MomentumCheckConfig, evaluate_momentum_setup


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class OKXDummyTrader:
    BASE_URL = "https://www.okx.com/api/v5"

    def __init__(
        self,
        log_file: str,
        interval_sec: int,
        max_new_trades_per_cycle: int,
        min_volume_5m: float,
        diagnostics_file: Optional[str] = None,
    ):
        self.log_file = log_file
        self.interval_sec = interval_sec
        self.max_new_trades_per_cycle = max_new_trades_per_cycle
        self.min_volume_5m = min_volume_5m
        self.diagnostics_file = diagnostics_file
        self.session = requests.Session()
        self.iteration = 0
        self._ensure_log_file()
        if self.diagnostics_file:
            self._ensure_diagnostics_file()

    def _ensure_log_file(self) -> None:
        cols = [
            "trade_id",
            "created_at_utc",
            "symbol",
            "direction",
            "entry_price",
            "sl_price",
            "tp_price",
            "risk_pct",
            "tp_distance_pct",
            "status",
            "exit_time_utc",
            "exit_price",
            "exit_reason",
            "pnl_pct",
            "time_in_trade_min",
            "scan_iteration",
            "slow_grind_approach",
            "left_side_staircase",
            "volume_not_decreasing",
            "not_choppy",
            "day_change_ok",
            "vwap_side_ok",
            "first_2h_prev_day_vwap_ok",
            "entry_not_crossed_6h",
            "overall_pass",
            "failed_checks",
            "avg_volume_5m",
            "day_change_pct",
            "entry_cross_count_6h",
            "dir_move_2h_pct",
            "efficiency_2h",
            "smma30_crosses_2h",
            "noise_class_momentum",
            "pre_entry_move_10m_pct",
            "pre_entry_efficiency_10m",
            "pre_entry_grind_10m_ok",
        ]
        try:
            pd.read_csv(self.log_file)
        except Exception:
            pd.DataFrame(columns=cols).to_csv(self.log_file, index=False)

    def _load_trades(self) -> pd.DataFrame:
        return pd.read_csv(self.log_file)

    def _save_trades(self, df: pd.DataFrame) -> None:
        df.to_csv(self.log_file, index=False)

    def _ensure_diagnostics_file(self) -> None:
        import os
        if not os.path.exists(self.diagnostics_file):
            cols = [
                "timestamp_utc",
                "iteration",
                "total_evaluated",
                "total_passed",
                "total_opened",
                "non_evaluable",
                "slow_grind_approach_fails",
                "left_side_staircase_fails",
                "volume_not_decreasing_fails",
                "not_choppy_fails",
                "day_change_ok_fails",
                "vwap_side_ok_fails",
                "first_2h_prev_day_vwap_ok_fails",
                "entry_not_crossed_6h_fails",
            ]
            try:
                df = pd.DataFrame(columns=cols)
                df.to_csv(self.diagnostics_file, index=False)
                print(f"[DIAG] created {self.diagnostics_file}")
            except Exception as exc:
                print(f"[DIAG] create_error={exc}")

    def _write_diagnostics_row(
        self,
        evaluated: int,
        passed: int,
        opened: int,
        non_evaluable: int,
        fail_counter: Counter,
    ) -> None:
        if not self.diagnostics_file:
            return

        row = {
            "timestamp_utc": utc_now_iso(),
            "iteration": self.iteration,
            "total_evaluated": evaluated,
            "total_passed": passed,
            "total_opened": opened,
            "non_evaluable": non_evaluable,
            "slow_grind_approach_fails": fail_counter.get("slow_grind_approach", 0),
            "left_side_staircase_fails": fail_counter.get("left_side_staircase", 0),
            "volume_not_decreasing_fails": fail_counter.get("volume_not_decreasing", 0),
            "not_choppy_fails": fail_counter.get("not_choppy", 0),
            "day_change_ok_fails": fail_counter.get("day_change_ok", 0),
            "vwap_side_ok_fails": fail_counter.get("vwap_side_ok", 0),
            "first_2h_prev_day_vwap_ok_fails": fail_counter.get("first_2h_prev_day_vwap_ok", 0),
            "entry_not_crossed_6h_fails": fail_counter.get("entry_not_crossed_6h", 0),
        }

        try:
            df = pd.DataFrame([row])
            df.to_csv(self.diagnostics_file, mode="a", header=False, index=False)
        except Exception as exc:
            print(f"[DIAG] write_error={exc}")

    def _get_all_usdt_swaps(self) -> list[str]:
        try:
            r = self.session.get(
                f"{self.BASE_URL}/public/instruments",
                params={"instType": "SWAP"},
                timeout=20,
            )
            if r.status_code != 200:
                return []
            payload = r.json()
            if payload.get("code") != "0":
                return []
            out = []
            for row in payload.get("data", []):
                inst_id = row.get("instId", "")
                if inst_id.endswith("-USDT-SWAP") and row.get("state") == "live":
                    out.append(inst_id)
            return sorted(set(out))
        except Exception:
            return []

    def _fetch_ohlcv(self, inst_id: str, bar: str, limit: int) -> Optional[pd.DataFrame]:
        try:
            max_batch = 300
            remaining = int(limit)
            before_ts: Optional[str] = None
            collected: list[list[str]] = []

            while remaining > 0:
                batch_size = min(remaining, max_batch)
                params = {"instId": inst_id, "bar": bar, "limit": batch_size}
                if before_ts is not None:
                    # OKX candles pagination: use "after" to request older candles.
                    params["after"] = before_ts

                r = self.session.get(
                    f"{self.BASE_URL}/market/candles",
                    params=params,
                    timeout=10,
                )
                if r.status_code != 200:
                    break
                payload = r.json()
                if payload.get("code") != "0":
                    break
                rows = payload.get("data", [])
                if not rows:
                    break

                collected.extend(rows)
                remaining -= len(rows)
                # Move cursor strictly older to avoid receiving the same window repeatedly.
                before_ts = str(int(rows[-1][0]) - 1)

                if len(rows) < batch_size:
                    break

            if not collected:
                return None

            rows = list(reversed(collected))
            df = pd.DataFrame(
                rows,
                columns=["ts", "o", "h", "l", "c", "vol", "volCcy", "volQuote", "confirm"],
            )
            out = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce"), unit="ms", utc=True),
                    "open": pd.to_numeric(df["o"], errors="coerce"),
                    "high": pd.to_numeric(df["h"], errors="coerce"),
                    "low": pd.to_numeric(df["l"], errors="coerce"),
                    "close": pd.to_numeric(df["c"], errors="coerce"),
                    "volume": pd.to_numeric(df["vol"], errors="coerce"),
                }
            ).dropna().drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            if out.empty:
                return None
            return out.set_index("timestamp")
        except Exception:
            return None

    def _fetch_ticker(self, inst_id: str) -> Optional[dict]:
        try:
            r = self.session.get(
                f"{self.BASE_URL}/market/ticker",
                params={"instId": inst_id},
                timeout=10,
            )
            if r.status_code != 200:
                return None
            payload = r.json()
            if payload.get("code") != "0":
                return None
            data = payload.get("data", [])
            if not data:
                return None
            return data[0]
        except Exception:
            return None

    def _fetch_ticker_price(self, inst_id: str) -> Optional[float]:
        ticker = self._fetch_ticker(inst_id)
        if not ticker:
            return None
        return to_float(ticker.get("last"), default=0.0)

    def _fetch_day_change_pct(self, inst_id: str) -> float:
        ticker = self._fetch_ticker(inst_id)
        if not ticker:
            return float("nan")
        # OKX provides open24h and last; this matches the intent of 24h day-change gating.
        last_px = to_float(ticker.get("last"), default=0.0)
        open24h = to_float(ticker.get("open24h"), default=0.0)
        if open24h <= 0:
            return float("nan")
        return ((last_px - open24h) / open24h) * 100.0

    @staticmethod
    def _session_vwap(df: pd.DataFrame) -> float:
        if df.empty:
            return float("nan")
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"]
        denom = float(vol.sum())
        if denom <= 0:
            return float("nan")
        return float((tp * vol).sum() / denom)

    @staticmethod
    def _find_second_swing_low(bars: pd.DataFrame) -> float:
        if len(bars) < 4:
            return float(bars["low"].min())
        first = float(bars["low"].iloc[-4:].min())
        if len(bars) >= 8:
            second = float(bars["low"].iloc[-8:-4].min())
            return second if second > first else first
        return first

    @staticmethod
    def _find_second_swing_high(bars: pd.DataFrame) -> float:
        if len(bars) < 4:
            return float(bars["high"].max())
        first = float(bars["high"].iloc[-4:].max())
        if len(bars) >= 8:
            second = float(bars["high"].iloc[-8:-4].max())
            return second if second < first else first
        return first

    @staticmethod
    def _calculate_atr_1h(bars_1h: pd.DataFrame, length: int = 14) -> float:
        if bars_1h is None or len(bars_1h) < length:
            return 0.0
        high = bars_1h["high"]
        low = bars_1h["low"]
        close = bars_1h["close"]
        tr = np.maximum(
            high - low,
            np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
        )
        atr = tr.rolling(length).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else 0.0

    @staticmethod
    def _find_resistance_level(bars_1m: pd.DataFrame, direction: str, entry_price: float) -> float:
        if bars_1m is None or bars_1m.empty:
            return entry_price * (1.03 if direction == "long" else 0.97)

        recent = bars_1m.tail(15)
        if direction == "long":
            cands = recent[recent["high"] > entry_price * 1.001]["high"]
            return float(cands.min()) if not cands.empty else entry_price * 1.03
        cands = recent[recent["low"] < entry_price * 0.999]["low"]
        return float(cands.max()) if not cands.empty else entry_price * 0.97

    def _build_order_setup(self, symbol: str, direction: str, entry_price: float) -> Optional[dict]:
        bars_30m = self._fetch_ohlcv(symbol, "30m", 8)
        bars_1h = self._fetch_ohlcv(symbol, "1H", 20)
        bars_1m = self._fetch_ohlcv(symbol, "1m", 20)
        if bars_30m is None or bars_1h is None or bars_1m is None:
            return None

        if direction == "long":
            swing_level = self._find_second_swing_low(bars_30m)
        else:
            swing_level = self._find_second_swing_high(bars_30m)
        level = self._find_resistance_level(bars_1m, direction, entry_price)

        atr = self._calculate_atr_1h(bars_1h, 14)
        atr_buffer = atr * 0.5 if atr > 0 else entry_price * 0.005

        min_tp_pct = 1.0
        max_tp_pct = 3.5

        if direction == "long":
            if swing_level >= entry_price:
                swing_level = entry_price * 0.96
            if swing_level < entry_price * 0.90:
                swing_level = entry_price * 0.94

            sl_price = min(swing_level - atr_buffer, entry_price * 0.97)
            tp_to_res = level - entry_price
            tp_pct_raw = (tp_to_res / entry_price) * 100
            tp_distance = entry_price * min(max(tp_pct_raw, min_tp_pct), max_tp_pct) / 100
            tp_price = entry_price + tp_distance
            risk_pct = ((entry_price - sl_price) / entry_price) * 100
            tp_pct = (tp_distance / entry_price) * 100
        else:
            if swing_level <= entry_price:
                swing_level = entry_price * 1.04
            if swing_level > entry_price * 1.10:
                swing_level = entry_price * 1.06

            sl_price = max(swing_level + atr_buffer, entry_price * 1.03)
            tp_to_sup = entry_price - level
            tp_pct_raw = (tp_to_sup / entry_price) * 100
            tp_distance = entry_price * min(max(tp_pct_raw, min_tp_pct), max_tp_pct) / 100
            tp_price = entry_price - tp_distance
            risk_pct = ((sl_price - entry_price) / entry_price) * 100
            tp_pct = (tp_distance / entry_price) * 100

        return {
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
            "risk_pct": float(risk_pct),
            "tp_distance_pct": float(tp_pct),
        }

    def _extended_checks(self, symbol: str, df_1m: pd.DataFrame, direction: str, eval_ts: pd.Timestamp) -> tuple[bool, bool, bool, bool, float, int]:
        idx = len(df_1m) - 1
        if idx < 370:
            return False, False, False, False, float("nan"), -1

        day_change_pct = self._fetch_day_change_pct(symbol)
        day_change_ok = day_change_pct >= 5.0 if direction == "long" else day_change_pct <= -5.0

        close_now = float(df_1m["close"].iloc[idx])
        today = eval_ts.date()
        yesterday = (eval_ts - pd.Timedelta(days=1)).date()

        today_df = df_1m[df_1m.index.date == today]
        prev_df = df_1m[df_1m.index.date == yesterday]

        current_vwap = self._session_vwap(today_df)
        prev_vwap = self._session_vwap(prev_df)

        if direction == "long":
            vwap_side_ok = bool(pd.notna(current_vwap) and close_now > current_vwap)
        else:
            vwap_side_ok = bool(pd.notna(current_vwap) and close_now < current_vwap)

        if eval_ts.hour < 2:
            if direction == "long":
                first_2h_ok = bool(pd.notna(prev_vwap) and close_now > prev_vwap and day_change_ok)
            else:
                first_2h_ok = bool(pd.notna(prev_vwap) and close_now < prev_vwap and day_change_ok)
        else:
            first_2h_ok = True

        if direction == "long":
            entry_price = float(df_1m["high"].iloc[-10:].max())
        else:
            entry_price = float(df_1m["low"].iloc[-10:].min())
        prior = df_1m.iloc[-370:-10]
        crossed = (prior["low"] <= entry_price) & (prior["high"] >= entry_price)
        entry_cross_count = int(crossed.sum())
        entry_not_crossed_6h = entry_cross_count == 0

        return (
            bool(day_change_ok),
            bool(vwap_side_ok),
            bool(first_2h_ok),
            bool(entry_not_crossed_6h),
            float(day_change_pct),
            int(entry_cross_count),
        )

    def _evaluate_symbol(self, symbol: str) -> Optional[dict]:
        df_1m = self._fetch_ohlcv(symbol, "1m", 600)
        if df_1m is None or len(df_1m) < 400:
            return {
                "symbol": symbol,
                "overall_pass": False,
                "non_evaluable": "insufficient_data",
                "failed_checks_list": [],
            }

        # Volume gate (last 5 bars average, USD-equivalent)
        vol_usd_5m = df_1m["volume"].tail(5) * df_1m["close"].tail(5)
        avg_vol_5m = float(vol_usd_5m.mean())
        if avg_vol_5m < self.min_volume_5m:
            return {
                "symbol": symbol,
                "overall_pass": False,
                "non_evaluable": "low_volume_5m",
                "failed_checks_list": [],
            }

        eval_ts = df_1m.index[-1]
        best_fail: Optional[dict] = None

        for direction in ("long", "short"):
            quality = evaluate_momentum_setup(
                df=df_1m,
                direction=direction,
                enforce_extended_rules=False,
                check_config=MomentumCheckConfig(require_grind_subchecks_in_balanced_2h=False),
            )

            day_change_ok, vwap_side_ok, first_2h_ok, entry_ok, day_change_pct, entry_cross_count = self._extended_checks(
                symbol=symbol,
                df_1m=df_1m,
                direction=direction,
                eval_ts=eval_ts,
            )

            checks = {
                "slow_grind_approach": bool(quality.checks.get("slow_grind_approach", False)),
                "left_side_staircase": bool(quality.checks.get("left_side_staircase", False)),
                "volume_not_decreasing": bool(quality.checks.get("volume_not_decreasing", False)),
                "not_choppy": bool(quality.checks.get("not_choppy", False)),
                "day_change_ok": day_change_ok,
                "vwap_side_ok": vwap_side_ok,
                "first_2h_prev_day_vwap_ok": first_2h_ok,
                "entry_not_crossed_6h": entry_ok,
            }

            overall_pass = all(checks.values())
            if not overall_pass:
                failed = [k for k, v in checks.items() if not v]
                candidate = {
                    "symbol": symbol,
                    "direction": direction,
                    "overall_pass": False,
                    "failed_checks_list": failed,
                    "checks_true_count": sum(1 for v in checks.values() if v),
                }
                if best_fail is None or candidate["checks_true_count"] > best_fail["checks_true_count"]:
                    best_fail = candidate
                continue

            entry = self._fetch_ticker_price(symbol)
            if entry is None or entry <= 0:
                return None

            setup = self._build_order_setup(symbol, direction, entry)
            if setup is None:
                return None

            failed = [k for k, v in checks.items() if not v]
            return {
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry,
                "setup": setup,
                "checks": checks,
                "overall_pass": overall_pass,
                "failed_checks": "|".join(failed),
                "failed_checks_list": failed,
                "avg_volume_5m": avg_vol_5m,
                "day_change_pct": day_change_pct,
                "entry_cross_count_6h": entry_cross_count,
                "metrics": quality.metrics,
            }

        if best_fail is not None:
            return best_fail
        return {
            "symbol": symbol,
            "overall_pass": False,
            "non_evaluable": "unknown",
            "failed_checks_list": [],
        }

    def _update_exits(self, trades: pd.DataFrame) -> pd.DataFrame:
        open_idx = trades.index[trades["status"] == "OPEN"].tolist()
        if not open_idx:
            return trades

        now = datetime.now(timezone.utc)
        for idx in open_idx:
            row = trades.loc[idx]
            symbol = row["symbol"]
            px = self._fetch_ticker_price(symbol)
            if px is None or px <= 0:
                continue

            direction = row["direction"]
            sl = to_float(row["sl_price"])
            tp = to_float(row["tp_price"])
            entry = to_float(row["entry_price"])

            exit_reason = ""
            exit_px = None

            if direction == "long":
                if px >= tp:
                    exit_reason = "TP_HIT"
                    exit_px = tp
                elif px <= sl:
                    exit_reason = "SL_HIT"
                    exit_px = sl
            else:
                if px <= tp:
                    exit_reason = "TP_HIT"
                    exit_px = tp
                elif px >= sl:
                    exit_reason = "SL_HIT"
                    exit_px = sl

            if exit_reason:
                created = pd.to_datetime(row["created_at_utc"], utc=True)
                mins = (now - created).total_seconds() / 60.0
                pnl_pct = ((exit_px - entry) / entry) * 100 if direction == "long" else ((entry - exit_px) / entry) * 100
                trades.at[idx, "status"] = "CLOSED"
                trades.at[idx, "exit_time_utc"] = now.isoformat()
                trades.at[idx, "exit_price"] = round(exit_px, 8)
                trades.at[idx, "exit_reason"] = exit_reason
                trades.at[idx, "pnl_pct"] = round(pnl_pct, 4)
                trades.at[idx, "time_in_trade_min"] = round(mins, 2)
                print(f"[EXIT] {symbol} {direction} {exit_reason} pnl={pnl_pct:.2f}%")

        return trades

    def _open_new_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        open_symbols = set(trades.loc[trades["status"] == "OPEN", "symbol"].tolist())

        universe = self._get_all_usdt_swaps()
        if not universe:
            print("[SCAN] no OKX USDT swaps fetched")
            return trades

        print(f"[SCAN] evaluating {len(universe)} OKX USDT swaps")
        opened = 0
        passed_count = 0
        evaluated_count = 0
        non_evaluable_count = 0
        fail_counter: Counter = Counter()

        for symbol in universe:
            if opened >= self.max_new_trades_per_cycle:
                break
            if symbol in open_symbols:
                continue

            eval_result = self._evaluate_symbol(symbol)
            if eval_result is None:
                non_evaluable_count += 1
                continue
            if eval_result.get("non_evaluable"):
                non_evaluable_count += 1
                continue

            evaluated_count += 1
            if not bool(eval_result.get("overall_pass", False)):
                for check_name in eval_result.get("failed_checks_list", []):
                    fail_counter[check_name] += 1
                continue

            passed_count += 1
            direction = eval_result["direction"]
            entry = eval_result["entry_price"]
            setup = eval_result["setup"]
            checks = eval_result["checks"]
            metrics = eval_result["metrics"]

            row = {
                "trade_id": str(uuid.uuid4())[:8],
                "created_at_utc": utc_now_iso(),
                "symbol": symbol,
                "direction": direction,
                "entry_price": round(entry, 8),
                "sl_price": round(setup["sl_price"], 8),
                "tp_price": round(setup["tp_price"], 8),
                "risk_pct": round(setup["risk_pct"], 4),
                "tp_distance_pct": round(setup["tp_distance_pct"], 4),
                "status": "OPEN",
                "exit_time_utc": "",
                "exit_price": "",
                "exit_reason": "",
                "pnl_pct": "",
                "time_in_trade_min": "",
                "scan_iteration": self.iteration,
                "slow_grind_approach": checks["slow_grind_approach"],
                "left_side_staircase": checks["left_side_staircase"],
                "volume_not_decreasing": checks["volume_not_decreasing"],
                "not_choppy": checks["not_choppy"],
                "day_change_ok": checks["day_change_ok"],
                "vwap_side_ok": checks["vwap_side_ok"],
                "first_2h_prev_day_vwap_ok": checks["first_2h_prev_day_vwap_ok"],
                "entry_not_crossed_6h": checks["entry_not_crossed_6h"],
                "overall_pass": True,
                "failed_checks": "",
                "avg_volume_5m": round(eval_result["avg_volume_5m"], 4),
                "day_change_pct": round(eval_result["day_change_pct"], 4),
                "entry_cross_count_6h": int(eval_result["entry_cross_count_6h"]),
                "dir_move_2h_pct": round(float(metrics.get("dir_move_2h_pct", np.nan)), 4),
                "efficiency_2h": round(float(metrics.get("efficiency_2h", np.nan)), 4),
                "smma30_crosses_2h": round(float(metrics.get("smma30_crosses_2h", np.nan)), 4),
                "noise_class_momentum": round(float(metrics.get("noise_class_momentum", np.nan)), 4),
                "pre_entry_move_10m_pct": round(float(metrics.get("pre_entry_move_10m_pct", np.nan)), 4),
                "pre_entry_efficiency_10m": round(float(metrics.get("pre_entry_efficiency_10m", np.nan)), 4),
                "pre_entry_grind_10m_ok": bool(float(metrics.get("pre_entry_grind_10m_ok", 0.0)) > 0.5),
            }

            trades = pd.concat([trades, pd.DataFrame([row])], ignore_index=True)
            open_symbols.add(symbol)
            opened += 1
            print(
                f"[ENTRY] {symbol} {direction} entry={entry:.6f} sl={setup['sl_price']:.6f} "
                f"tp={setup['tp_price']:.6f} tp%={setup['tp_distance_pct']:.2f}"
            )

        print(f"[SCAN] evaluated={evaluated_count} passed={passed_count} opened={opened} non_evaluable={non_evaluable_count}")
        if fail_counter:
            top_failed = ", ".join([f"{k}={v}" for k, v in fail_counter.most_common(6)])
            print(f"[SCAN] top_failed_checks: {top_failed}")
        
        self._write_diagnostics_row(
            evaluated=evaluated_count,
            passed=passed_count,
            opened=opened,
            non_evaluable=non_evaluable_count,
            fail_counter=fail_counter,
        )
        return trades

    def run_cycle(self) -> None:
        self.iteration += 1
        trades = self._load_trades()
        trades = self._update_exits(trades)
        trades = self._open_new_trades(trades)
        self._save_trades(trades)

        open_n = int((trades["status"] == "OPEN").sum())
        closed_n = int((trades["status"] == "CLOSED").sum())
        print(f"[CYCLE {self.iteration}] open={open_n} closed={closed_n} total={len(trades)}")

    def run(self, once: bool = False) -> None:
        print("=" * 70)
        print("OKX DUMMY TRADER (PAPER MODE, FULL CRITERIA)")
        print("=" * 70)
        print(f"log_file={self.log_file}")
        if self.diagnostics_file:
            print(f"diagnostics_file={self.diagnostics_file}")
        print(f"interval_sec={self.interval_sec}")
        print(f"max_new_trades_per_cycle={self.max_new_trades_per_cycle}")
        print(f"min_volume_5m={self.min_volume_5m}")
        print()

        if once:
            self.run_cycle()
            return

        while True:
            try:
                self.run_cycle()
                time.sleep(self.interval_sec)
            except KeyboardInterrupt:
                print("Stopped by user")
                break
            except Exception as exc:
                print(f"cycle_error={exc}")
                time.sleep(self.interval_sec)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OKX dummy trader with full strategy criteria")
    p.add_argument("--log-file", default="okx_dummy_trades.csv")
    p.add_argument("--diagnostics-file", default="", help="CSV file to log per-cycle failure diagnostics")
    p.add_argument("--interval-sec", type=int, default=300)
    p.add_argument("--max-new-trades", type=int, default=2)
    p.add_argument("--min-volume-5m", type=float, default=50000.0)
    p.add_argument("--once", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bot = OKXDummyTrader(
        log_file=args.log_file,
        interval_sec=args.interval_sec,
        max_new_trades_per_cycle=args.max_new_trades,
        min_volume_5m=args.min_volume_5m,
        diagnostics_file=args.diagnostics_file if args.diagnostics_file else None,
    )
    bot.run(once=args.once)
