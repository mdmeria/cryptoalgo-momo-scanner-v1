# MOMO LONG STRATEGY - COMPLETE PARAMETER SHEET

**Strategy Type:** Mean-Reversion / Momentum Grind (Directional Breakout)  
**Direction:** LONG  
**Market:** OKX (spot/perpetual SWAPS)  
**Updated:** 2026-03-10  

---

## 1. SCANNER ENTRY GATE (Orion Pre-Filter)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Min 5m Trade Count | ≥ 1,000 | Ensures sufficient market liquidity & volatility |
| Currency Pair | USDT (swap format: `{SYMBOL}-USDT-SWAP`) | Standardized contract format |
| Data Lookback (scanner) | 600 bars @ 1m | 10 hours of recent data |

---

## 2. CORE MOMENTUM CHECKLIST (Mandatory - All Must Pass)

### A. Slow Grind Approach (Last 10 Minutes)

**Purpose:** Validate clean, directional entry WITHOUT opposite-direction candles

| Parameter | Condition | Value | Notes |
|-----------|-----------|-------|-------|
| Net Move (10 bars) | **net_move > 0** | Upside | Entry must be upward |
| Opposite Candles | opposite_candles ≤ 4 | ≤ 4 downward bars | Allow minor pullbacks |
| Minimum Move Distance | NA | NA | No hard floor, but must show directional intent |

**Rule Summary:**  
Last 10 1-minute candles: at least 6 bars up, at most 4 bars down. No sideways/choppy entry.

---

### B. Pre-Entry Directional (Last 30 Minutes)

**Purpose:** Entry zone must not be in a sideways / low-efficiency pattern

| Parameter | Condition | Value | Notes |
|-----------|-----------|-------|-------|
| 30m Direction Move | **dir_move ≥ 0.004** | ≥ 0.4% | Minimum directional impulse over 30m |
| 30m Dir Bar Ratio | **dir_bar_ratio ≥ 0.50** | ≥ 50% | At least half the bars must be up |
| 30m Efficiency | **efficiency ≥ 0.20** | ≥ 20% | Dir move to path ratio (not squiggly) |
| **Entry Zone Spike Filter** | **concentration ≤ 0.60** | ≤ 60% | Last 10 bars cannot exceed 60% of 30m move |

**Last 10 Minute Grind (Critical):**

| Parameter | Condition | Value | Notes |
|-----------|-----------|-------|-------|
| 10m Direction Move | **dir_move_10m ≥ 0.0020** | ≥ 0.2% | Minimal directional continuation into entry |
| 10m Dir Bar Ratio | **dir_bar_ratio_10m ≥ 0.60** | ≥ 60% | Majority bars up in final minute grind |
| 10m Efficiency | **0.30 ≤ efficiency_10m ≤ 0.95** | 30–95% | Not a straight ramp or sideways spike |
| 10m Opposite Candles | **opp_candles_10m ≤ 4** | ≤ 4 down bars | Limited pullback resistance in final grind |

**Rule Summary:**  
Entry approach must accelerate into the final 10 minutes with at least 60% up bars, without spiking >60% of the 30m move in the last 10 bars.

---

### C. Left Side Staircase (2-Hour Context)

**Purpose:** Establish directional regime confidence using 30/120 SMMA stack

| Parameter | Condition | Value (120 bars) | Notes |
|-----------|-----------|------------------|-------|
| **SMMA30 Slope** | **smma30_slope > 0** | Positive slope | 30-bar SMMA must trend UP |
| **SMMA120 Slope** | **smma120_slope > 0** | Positive slope | 120-bar SMMA must trend UP |
| **Trend Stack Bars** | **trend_stack_bars ≥ 84** | ≥ 84 / 120 bars (70%) | SMMA30 must be ABOVE SMMA120 for ≥70% of 2h |
| **Staircase Bars** | **staircase_bars ≥ 72** | ≥ 72 / 120 bars (60%) | Close must be ABOVE SMMA30 for ≥60% of 2h |
| **Current Position** | **close[-1] > smma30[-1]** | Price above 30SMMA | Most recent candle must confirm uptrend |

**Rule Summary:**  
Over the last 2 hours: SMMA30 > SMMA120 for at least 70% (84 bars), price above SMMA30 for at least 60% (72 bars), both MAs trending up. No regime flip into entry.

---

### D. Volume Not Decreasing (Last 30 Minutes)

**Purpose:** Reject declining momentum volume (loss of interest)

| Parameter | Condition | Value | Notes |
|-----------|-----------|-------|-------|
| 20-Bar Volume MA Slope | **vol_slope > -0.02** | > -2% / bar | Volume trend may be flat or rising, not heavy decay |

**Rule Summary:**  
Volume moving average should not be declining sharply. Allows flat volume but rejects obvious volume exhaust.

---

### E. Not Choppy (Noise Classification from 30SMMA)

**Purpose:** Reject high-noise directional moves that reverse quickly

**Noise Model:**
- **Sideways 30SMMA** (|slope| ≤ 0.0005) → Always HIGH noise → **REJECT**
- **Trending 30SMMA** (`slope > 0` for long) + **Cross Count:**
  - ≤ 3 crosses in 2h → **LOW noise** → ACCEPT
  - 4–6 crosses → **MEDIUM noise** → ACCEPT
  - ≥ 7 crosses → **HIGH noise** → **REJECT**

**Rule Summary:**  
The 30SMMA must be trending upward with at most 6 side-touches of price in the 2h window. More than 6 crossover = chop.

---

### F. Balanced Momentum 2-Hour Profile

**Purpose:** Ensure 2h is a directional grind, not a spike or sideways

| Parameter | Condition | Value | Notes |
|-----------|-----------|-------|-------|
| 2h Direction Move | **0.004 ≤ dir_move ≤ 0.350** | 0.4% – 35% | Meaningful directional move, not ramp/spike |
| 2h Dir Bar Ratio | **0.45 ≤ dir_bar_ratio ≤ 0.98** | 45% – 98% | Directional but not one-sided (avoids over-extended) |
| Max Directional Impulse (8m) | **max_dir_impulse_8m ≤ 0.025** | ≤ 2.5% per 8m window | No single 8-minute spike >2.5% |
| 2h Efficiency | **0.12 ≤ efficiency ≤ 0.90** | 12% – 90% | Path-vs-move ratio: not squiggly, not straight |
| Regime Age | **regime_age_bars ≥ 5** | ≥ 5 bars old | Current move initiated at least 5 bars ago (not freshly born) |

**Rule Summary:**  
2-hour move should be between 0.4% – 35%, with efficiency (net/path) between 12% – 90%. Max any 8-minute impulse is 2.5%. Regime must be at least 5 bars old.

---

### G. Parallel to 30SMMA (Price Stays on Trend Side)

**Purpose:** Price should not repeatedly cross the 30SMMA; stays "above" for a long

| Parameter | Condition | Value | Notes |
|-----------|-----------|-------|-------|
| Touch Tolerance | 0.15% deviation | Treat minor touches as "not crossed" | Avoids false crosses on wicks |
| SMMA30 Crosses 2h | **cross_count ≤ 4** | ≤ 4 side-touches in 2h | Price can touch but not cut through |
| Trend Side Ratio | **trend_side_ratio ≥ 0.85** | ≥ 85% of bars above SMMA30 | Price spends majority of 2h above 30SMMA |

**Rule Summary:**  
Price must remain above 30SMMA for ≥85% of the 2-hour window, with at most 4 touches/crossovers. No extended periods below.

---

### H. SMMA Spread Slowly Increasing

**Purpose:** The gap between 30SMMA and 120SMMA (signal of strength) should be growing

| Parameter | Condition | Value | Notes |
|-----------|-----------|-------|-------|
| Spread Slope | **spread_slope > 0** | Positive | Gap (30SMMA - 120SMMA) must be widening |
| Spread Up Ratio | **spread_up_ratio ≥ 0.52** | ≥ 52% of bars | Gap increases on majority of bars over 2h |

**Rule Summary:**  
The 30/120 gap must be widening (30SMMA pulling away from 120SMMA above it). Indicates strengthening momentum.

---

## 3. EXTENDED CHECKS (Applied at Entry, May Reject Even if Core Passes)

### I. Day Change Gate

**Purpose:** Market sentiment filter: long only if 24h change positive

| Parameter | Condition (LONG) | Value | Notes |
|-----------|------------------|-------|-------|
| 24-Hour Price Change % | **day_change_pct ≥ 5.0%** | ≥ +5% | Asset must be up at least 5% over 24h |

**Source:** OKX ticker 24h change OR historical 5m candle approximation  
**Rule Summary:** LONG setups require positive sentiment: ≥ +5% 24h change.

---

### J. VWAP Side Gate

**Purpose:** Entry should be above today's session VWAP

| Parameter | Condition (LONG) | Value | Notes |
|-----------|------------------|-------|-------|
| Current Day VWAP | **close_now > current_vwap** | Price > today's VWAP | Must be above VWAP |

**Rule Summary:**  
Entry price must be above the volume-weighted average price of the current trading day.

---

### K. First 2H Previous Day VWAP Gate

**Purpose:** During market open (00:00 – 02:00 UTC), additional confirmation from previous day

| Parameter | Condition (LONG, First 2h Only) | Value | Notes |
|-----------|----------------------------------|-------|-------|
| Time Window | **eval_ts.hour < 2** | 00:00 – 02:00 UTC | Only applies to first 2h of trading day |
| Previous Day VWAP Requirement | **close_now > prev_day_vwap AND day_change_pct ≥ 5.0** | Price > yesterday VWAP + ≥5% 24h change | During market open, must clear previous day VWAP |

**Rule Summary:**  
During first 2 hours of session, price must be above yesterday's VWAP AND symbol must have +5% 24h change.

---

### L. Entry Not Crossed in 6H

**Purpose:** Entry price level should not have been touched in prior 6 hours (true breakout, not retry)

| Parameter | Condition | Value | Notes |
|-----------|-----------|-------|-------|
| Entry Price Definition | max(high[-10:]) | Last 10 bars high | Potential breakout level |
| 6-Hour Prior Window | bars -370 to -10 | Prior 360 bars @ 1m (~6h) | Historical lookback |
| Cross Count | **entry_cross_count == 0** | Zero crosses | Entry level must NOT have been touched/crossed before |

**Rule Summary:**  
The entry level (highest high in last 10 bars) should NOT have been breached in the 6-hour window before entry. True fresh breakout.

---

## 4. STRUCTURAL SL/TP CALCULATION

### Stop-Loss (SL) Placement

**Method:** Multi-factor structural SL  
1. Find 2nd swing low (4–8 bar lookback in 30m candles)
2. Apply ATR(14) on 1-hour bars × 0.5 as buffer
3. **For LONG:** `SL = min(swing_low - atr_buffer, entry_price * 0.97)`
4. Clamp SL to not go above 97% of entry

**Risk Calculation (OKX):**  
`risk_pct = ((entry_price - sl_price) / entry_price) * 100`

**Example (FLOW-USDT-SWAP):**
- Entry: 0.05757
- Swing Low: ~0.0529
- ATR Buffer: ~0.0005
- SL: 0.05294
- Risk%: 8.05%

---

### Take-Profit (TP) Placement

**Method:** Resistance-based TP with capping  
1. Find 1-minute resistance (recent highs above entry)
2. Calculate distance: `tp_distance_points = resistance - entry`
3. Calculate as %, clamp to 1.0% – 3.5%
4. **For LONG:** `TP = entry_price + tp_distance_points`
5. Default cap: 1.0% – 3.5% (dynamically adjusted per setup)

**Example (FLOW-USDT-SWAP):**
- Entry: 0.05757
- TP: 0.05815
- TP%: 1.0%

---

## 5. RISK/REWARD PROFILE (Current Trades)

| Trade | Entry | SL | TP | Risk% | Reward% | RR Ratio |
|-------|-------|----|----|-------|---------|----------|
| FLOW-USDT-SWAP | 0.05757 | 0.05294 | 0.05815 | 8.05% | 1.00% | 0.12 |
| OPN-USDT-SWAP | 0.3399 | 0.30774 | 0.35010 | 9.46% | 3.00% | 0.32 |
| EDEN-USDT-SWAP | 0.04223 | 0.03837 | 0.04350 | 9.14% | 3.00% | 0.33 |

**Issue:** RR ratios < 1:1 → Win rate needs to be **>50%** to be profitable.  
**Recommendation:** Adjust TP targets upward (2.5% – 5% range) to achieve ≥1:1 RR.

---

## 6. SUMMARY CHECKLIST

**MOMO LONG Entry Requires ALL of the Following:**

- [ ] 1. **Slow Grind Approach** (Last 10m: up-skewed, ≤4 down bars)
- [ ] 2. **Pre-Entry Directional** (30m: ≥0.4% move, ≥50% up bars, ≥20% efficiency, ≤60% concentration, 10m grind OK)
- [ ] 3. **Left Side Staircase** (2h: SMMA30 > SMMA120 for ≥70%, close > SMMA30 for ≥60%)
- [ ] 4. **Volume Not Decreasing** (vol_slope > -2% per bar)
- [ ] 5. **Not Choppy** (30SMMA trending, ≤6 crosses in 2h)
- [ ] 6. **Balanced 2h Profile** (0.4%–35% move, 45–98% up bar ratio, ≤2.5% per 8m, 12–90% efficiency, regime ≥5 bars old)
- [ ] 7. **Parallel to 30SMMA** (≤4 crosses in 2h, ≥85% bars above SMMA30)
- [ ] 8. **SMMA Spread Increasing** (gap widening, ≥52% bars up ratio)
- [ ] 9. **Day Change ≥ +5%** (24h sentiment)
- [ ] 10. **Above Current VWAP** (today's session VWAP)
- [ ] 11. **First 2H: Above Prev Day VWAP** (if eval_time.hour < 2)
- [ ] 12. **Entry Not Crossed in 6H** (fresh breakout level)

**If ALL pass → LONG entry with structural SL/TP per above.**

---

## 7. KEY TUNING KNOBS (If Poor Win Rate)

| Parameter | Current | Soften (More Signals) | Tighten (Fewer Signals) |
|-----------|---------|----------------------|------------------------|
| Staircase bars | ≥72/120 (60%) | ≥60/120 (50%) | ≥90/120 (75%) |
| 30m dir_bar_ratio | ≥50% | ≥40% | ≥60% |
| Pre-entry spike concentration | ≤60% | ≤70% | ≤50% |
| SMMA20 crosses | ≤4 | ≤6 | ≤3 |
| 2h efficiency | 12–90% | 10–95% | 15–85% |
| day_change_pct | ≥5% | ≥3% | ≥7% |
| TP distance | 1–3.5% | 0.8–4% | 1.5–3% |
| Risk%/SL placement | 8–10% | 10–12% | 5–8% |

---

## Files & References

- **Strategy Logic:** [momentum_quality.py](momentum_quality.py#L814–L870)
- **Live Trader:** [okx_dummy_trader.py](okx_dummy_trader.py#L518–L602)
- **Current Trades Log:** [okx_dummy_trades_full_criteria.csv](okx_dummy_trades_full_criteria.csv)
- **Diagnostics & Failures:** [okx_scan_diagnostics_coin_failures.csv](okx_scan_diagnostics_coin_failures.csv)
