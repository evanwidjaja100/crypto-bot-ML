# Phase 7.5 — Gross Edge Benchmark Report

**Date:** 2026-08-17  
**Dataset:** `BTCUSDT_60_37ec52b2e62f627c` (Clean Mainnet Data)  
**Partitions:** Train = 13,024 rows | Val = 2,789 rows | Test = 2,792 rows | Holdout (Untouched) = 2,900 rows  
**Scope:** Systematic out-of-sample evaluation of 42 configurations on `test` only.

---

## 1. Executive Summary & Gate Verdict

| Metric | Result | Target / Standard |
| :--- | :---: | :---: |
| **Total Configurations Evaluated** | **42** | Complete scan across rulesets & ML models |
| **Configs with Gross Edge (95% CI > 0)** | **0 / 42 (0%)** | $\ge 1$ required to clear gate |
| **Configs with Positive Gross Mean** | **10 / 42 (23.8%)** | > 0 bps/trade |
| **Configs with Net Positive Return** | **7 / 42 (16.7%)** | After ~15 bps round-trip fees/slippage |
| **Best Gross Edge ($N \ge 30$)** | **+6.1 bps [-12.2, +24.3]** (`ruleset_momentum20_fair`, $N=291$) | CI includes zero; net return negative |
| **Best Gross Edge (any $N$)** | **+293.8 bps [-154.7, +742.3]** (`lightgbm_th0.55_trend`, $N=7$) | Insufficient sample ($N=7$), Top-5 concentration 100% |

> [!IMPORTANT]
> **Phase 7 Gate Verdict: NO STATISTICAL GROSS EDGE FOUND**  
> On clean mainnet data with past-only label thresholds, boundary purging, and realistic execution modeling:
> 1. No configuration achieves a statistically significant gross expectancy above zero ($p > 0.05$).
> 2. Realistic execution costs (~15–18 bps round-trip) turn even the marginally positive gross rulesets into unprofitable strategies (Net PF 0.66–0.88).
> 3. High-confidence ML signals take $\le 10$ trades, with 85–100% of return driven by top-5 lucky moves, failing pre-registered criteria.

---

## 2. Complete Benchmark Scan Results

| Candidate Configuration | $N$ Trades | Gross bps (95% CI) | Net bps | Gross PnL ($) | Net Return (%) | Profit Factor | Win Rate (%) | Top-5 Share (%) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `ruleset_donchian20_fair` | 62 | -0.2 [-96.4, +96.0] | -11.2 | +$29.2 | -1.45% | 0.88 | 32.3% | 61.4% | NO EDGE |
| `ruleset_donchian20_std` | 113 | -1.5 [-35.8, +32.9] | -11.3 | -$29.0 | -2.69% | 0.86 | 38.1% | 20.0% | NO EDGE |
| `ruleset_ema10x30_fair` | 99 | -10.0 [-56.4, +36.4] | -21.0 | +$41.5 | -1.71% | 0.86 | 25.3% | 42.8% | NO EDGE |
| `ruleset_ema10x30_std` | 209 | -20.1 [-42.5, +2.3] | -30.2 | -$635.6 | -11.30% | 0.66 | 30.1% | 15.7% | NO EDGE |
| `ruleset_momentum20_fair` | 291 | +6.1 [-12.2, +24.3] | -4.9 | +$408.1 | -2.64% | 0.87 | 23.4% | 25.3% | POS GROSS (NET NEG) |
| `ruleset_momentum20_std` | 382 | -2.6 [-14.7, +9.4] | -13.1 | +$44.2 | -9.49% | 0.74 | 27.7% | 12.5% | NO EDGE |
| `logistic_th0.34_std` | 218 | -14.8 [-35.1, +5.5] | -25.4 | -$590.0 | -10.97% | 0.64 | 40.4% | 14.9% | NO EDGE |
| `logistic_th0.34_trend` | 204 | -32.7 [-59.7, -5.7] | -43.7 | -$815.1 | -11.76% | 0.49 | 39.7% | 19.1% | NO EDGE |
| `logistic_th0.36_std` | 174 | -31.7 [-55.0, -8.4] | -42.4 | -$989.3 | -13.98% | 0.51 | 35.6% | 19.4% | NO EDGE |
| `logistic_th0.36_trend` | 151 | -19.9 [-54.6, +14.9] | -30.9 | -$369.9 | -6.53% | 0.65 | 43.7% | 20.4% | NO EDGE |
| `logistic_th0.38_std` | 147 | -18.4 [-46.3, +9.5] | -28.8 | -$597.4 | -8.73% | 0.64 | 38.8% | 18.6% | NO EDGE |
| `logistic_th0.38_trend` | 109 | +5.6 [-43.6, +54.8] | -5.4 | +$23.0 | -1.95% | 0.86 | 52.3% | 23.3% | POS GROSS (NET NEG) |
| `logistic_th0.40_std` | 119 | -27.5 [-61.4, +6.3] | -37.7 | -$645.6 | -9.08% | 0.60 | 37.0% | 21.1% | NO EDGE |
| `logistic_th0.40_trend` | 78 | -14.2 [-84.5, +56.2] | -25.2 | -$221.0 | -3.72% | 0.73 | 47.4% | 40.8% | NO EDGE |
| `logistic_th0.42_std` | 101 | -4.9 [-42.8, +32.9] | -14.9 | -$84.0 | -3.21% | 0.82 | 40.6% | 20.0% | NO EDGE |
| `logistic_th0.42_trend` | 69 | +0.2 [-83.1, +83.4] | -10.9 | -$73.9 | -2.11% | 0.82 | 46.4% | 38.8% | POS GROSS (NET NEG) |
| `logistic_th0.45_std` | 67 | -10.5 [-64.7, +43.6] | -20.5 | -$75.0 | -3.07% | 0.78 | 38.8% | 31.5% | NO EDGE |
| `logistic_th0.45_trend` | 43 | +32.6 [-86.7, +151.9] | +21.6 | +$112.6 | +0.35% | 1.05 | 46.5% | 52.1% | POS GROSS (HIGH NOISE) |
| `logistic_th0.50_std` | 34 | -5.1 [-98.1, +87.9] | -15.1 | -$52.6 | -1.32% | 0.83 | 44.1% | 51.5% | NO EDGE |
| `logistic_th0.50_trend` | 26 | +70.4 [-96.0, +236.8] | +59.4 | +$192.1 | +1.49% | 1.39 | 53.8% | 69.2% | LOW N / HIGH CONCENTRATION |
| `logistic_th0.55_std` | 13 | +64.9 [-130.2, +259.9] | +55.3 | +$145.5 | +1.18% | 1.48 | 61.5% | 85.3% | LOW N / HIGH CONCENTRATION |
| `logistic_th0.55_trend` | 9 | +190.2 [-179.2, +559.5] | +179.1 | +$138.4 | +1.24% | 2.21 | 77.8% | 95.1% | LOW N / HIGH CONCENTRATION |
| `logistic_th0.60_std` | 4 | -5.3 [-441.1, +430.6] | -14.5 | +$44.1 | +0.37% | 1.35 | 50.0% | 100.0% | LOW N / HIGH CONCENTRATION |
| `logistic_th0.60_trend` | 3 | +248.0 [-745.1, +1241.1] | +236.9 | +$26.3 | +0.22% | 1.43 | 66.7% | 100.0% | LOW N / HIGH CONCENTRATION |
| `lightgbm_th0.34_std` | 261 | -16.7 [-34.2, +0.7] | -27.5 | -$734.1 | -13.25% | 0.60 | 44.1% | 15.6% | NO EDGE |
| `lightgbm_th0.34_trend` | 240 | -9.8 [-33.2, +13.6] | -20.8 | -$246.4 | -7.15% | 0.68 | 48.3% | 17.1% | NO EDGE |
| `lightgbm_th0.36_std` | 208 | -19.6 [-40.8, +1.5] | -30.3 | -$672.3 | -11.96% | 0.60 | 44.2% | 15.5% | NO EDGE |
| `lightgbm_th0.36_trend` | 173 | -14.9 [-48.3, +18.6] | -25.9 | -$108.9 | -4.80% | 0.75 | 52.0% | 19.7% | NO EDGE |
| `lightgbm_th0.38_std` | 171 | -27.9 [-52.7, -3.1] | -38.5 | -$750.7 | -12.28% | 0.57 | 39.2% | 18.0% | NO EDGE |
| `lightgbm_th0.38_trend` | 129 | -14.7 [-54.7, +25.3] | -25.7 | -$253.7 | -4.85% | 0.72 | 50.4% | 25.6% | NO EDGE |
| `lightgbm_th0.40_std` | 138 | -26.9 [-57.8, +3.9] | -37.4 | -$844.7 | -10.40% | 0.59 | 37.0% | 20.8% | NO EDGE |
| `lightgbm_th0.40_trend` | 90 | -5.6 [-60.6, +49.4] | -16.6 | -$199.0 | -3.45% | 0.75 | 51.1% | 30.2% | NO EDGE |
| `lightgbm_th0.42_std` | 109 | -37.5 [-73.1, -1.9] | -47.7 | -$789.1 | -10.28% | 0.53 | 33.0% | 28.7% | NO EDGE |
| `lightgbm_th0.42_trend` | 65 | -49.2 [-124.5, +26.1] | -60.2 | -$503.8 | -6.26% | 0.54 | 43.1% | 35.8% | NO EDGE |
| `lightgbm_th0.45_std` | 78 | -18.0 [-65.5, +29.6] | -28.0 | -$380.1 | -4.54% | 0.71 | 34.6% | 30.1% | NO EDGE |
| `lightgbm_th0.45_trend` | 45 | +0.4 [-116.8, +117.7] | -10.6 | -$59.1 | -1.56% | 0.84 | 37.8% | 51.4% | NO EDGE |
| `lightgbm_th0.50_std` | 27 | -80.0 [-155.5, -4.5] | -90.3 | -$356.9 | -4.25% | 0.40 | 25.9% | 86.1% | NO EDGE |
| `lightgbm_th0.50_trend` | 20 | -32.1 [-207.9, +143.8] | -43.1 | -$183.7 | -2.28% | 0.50 | 40.0% | 89.7% | NO EDGE |
| `lightgbm_th0.55_std` | 10 | -110.8 [-285.9, +64.4] | -120.6 | -$145.7 | -1.68% | 0.48 | 30.0% | 100.0% | NO EDGE |
| `lightgbm_th0.55_trend` | 7 | +293.8 [-154.7, +742.3] | +282.8 | +$192.6 | +1.79% | 3.66 | 71.4% | 100.0% | LOW N / HIGH CONCENTRATION |
| `lightgbm_th0.60_std` | 3 | -301.6 [-308.9, -294.3] | -312.4 | -$149.3 | -1.57% | 0.00 | 0.0% | 0.0% | NO EDGE |
| `lightgbm_th0.60_trend` | 2 | -533.6 [-992.9, -74.3] | -544.3 | -$69.7 | -0.72% | 0.00 | 0.0% | 0.0% | NO EDGE |

---

## 3. Findings & Detailed Breakdown

### A. Frequency vs Edge Tradeoff
1. **High-Frequency Regime ($th \le 0.42$)**:
   - Ample trade count ($N = 65 \dots 382$), providing statistical power.
   - Mean gross return is consistently negative ($-5$ to $-49$ bps per trade).
   - Net profit factors are deep in loss territory ($0.49 \dots 0.88$).
2. **Low-Frequency Regime ($th \ge 0.50$)**:
   - Model takes very few trades ($N = 2 \dots 26$).
   - Apparent high returns in small samples are entirely dominated by 1 or 2 outlier moves (top-5 trade share reaches 70%–100%), with wide confidence intervals crossing zero by several hundred basis points.

### B. Pre-Registered Criteria Audit (from `settings.yaml`)
- **Gross edge > 20 bps with 95% CI excluding 0**: ❌ **Failed** (0 / 42 configs met this).
- **Test / Holdout trades $\ge 100$**: ❌ **Failed** on all configs with positive sample means.
- **Profit Factor $\ge 1.15$**: ❌ **Failed** on all configs with adequate trade counts ($N \ge 100$).
- **Top-5 trade share $< 50\%$**: ❌ **Failed** on high-threshold configs.

---

## 4. Architectural Decision & Recommendation

According to the design of **Implementation Plan v2 (Phase 7 Gate)**:
- **Proceeding to Phase 8 (Operations) and Phase 9 (Live execution)** is **strictly gated** on establishing a verified gross edge.
- Deploying live money or spending time on exchange-native stop orders and monitoring for an unprofitable strategy would violate the project's working rules.
- **The holdout partition (2,900 rows) remains completely untouched and uncontaminated.**

**Recommended next steps:**
1. Document Phase 7.5 as completed with "No gross edge on 60m BTCUSDT directional classification".
2. Mark Phase 7 Gate as closed per audit criteria.
3. Keep the codebase in paper mode with the hardened infrastructure, sanitized pipelines, and clean test suite intact.
