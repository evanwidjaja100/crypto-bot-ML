"""Alpha Research V2: Evaluate Triple-Barrier, Cross-Sectional Momentum, and Funding Carry strategies.

Evaluates candidates on the test partition with confidence intervals and trade statistics.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        import lightgbm.basic as _b

        _b._LIB.LGBM_DatasetSetField.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        _b._LIB.LGBM_DatasetSetField.restype = ctypes.c_int32
    except Exception:
        pass

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.engine import BacktestEngine
from src.config import LgbmSettings, load_settings
from src.data_ingestion.candle_downloader import CandleStore
from src.data_ingestion.funding_downloader import FundingStore
from src.data_ingestion.intervals import INTERVAL_MS
from src.features.cross_sectional import align_basket, compute_cross_sectional_features
from src.features.pipeline import build_feature_frame
from src.labels.triple_barrier import (
    CLASS_LONG_WIN,
    CLASS_SHORT_WIN,
    add_triple_barrier_labels,
)
from src.models.train import train_lgbm
from src.monitoring.logging_setup import setup_logging
from src.strategy.signal_engine import FLAT, OPEN_LONG, OPEN_SHORT, PositionState, SignalDecision

log = logging.getLogger("evaluate_basket_edge")


def compute_metrics(
    trades_df: pd.DataFrame, gross_engine: dict, net_engine: dict
) -> dict[str, Any]:
    n = len(trades_df) if trades_df is not None else 0
    if n == 0:
        return {
            "n_trades": 0,
            "mean_gross_bps": 0.0,
            "ci95_gross_bps": (0.0, 0.0),
            "mean_net_bps": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "net_return": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "top5_share": 0.0,
        }

    entry_p = trades_df["entry_price"].to_numpy(dtype=float)
    exit_p = trades_df["exit_price"].to_numpy(dtype=float)
    direction = trades_df["direction"].to_numpy(dtype=float)
    qty = trades_df["qty"].to_numpy(dtype=float)
    notional = entry_p * qty
    fees = trades_df["fees"].to_numpy(dtype=float)
    net_pnls = trades_df["net_pnl"].to_numpy(dtype=float)

    gross_ret = direction * (exit_p - entry_p) / entry_p
    gross_bps = gross_ret * 10_000.0
    fee_bps = np.where(notional > 0, (fees / notional) * 10_000.0, 0.0)
    net_bps = gross_bps - fee_bps

    mean_gross = float(np.mean(gross_bps))
    std_gross = float(np.std(gross_bps, ddof=1)) if n > 1 else 0.0
    se_gross = std_gross / np.sqrt(n) if n > 1 else 0.0
    ci_gross = (mean_gross - 1.96 * se_gross, mean_gross + 1.96 * se_gross)

    mean_net = float(np.mean(net_bps))

    pos_pnls = [p for p in net_pnls if p > 0]
    top5_share = (
        (sum(sorted(pos_pnls, reverse=True)[:5]) / sum(pos_pnls)) if sum(pos_pnls) > 0 else 0.0
    )

    return {
        "n_trades": n,
        "mean_gross_bps": mean_gross,
        "ci95_gross_bps": ci_gross,
        "mean_net_bps": mean_net,
        "gross_pnl": gross_engine.get("total_gross_pnl", 0.0),
        "net_pnl": net_engine.get("total_net_pnl", 0.0),
        "net_return": net_engine.get("total_return", 0.0),
        "profit_factor": net_engine.get("profit_factor", 0.0),
        "win_rate": net_engine.get("win_rate", 0.0),
        "top5_share": top5_share,
    }


def run_strategy(df: pd.DataFrame, decider, risk_cfg, settings) -> dict[str, Any]:
    # Net run
    net_res = BacktestEngine(
        df,
        decider,
        initial_equity=settings.backtest.initial_equity,
        taker_fee=settings.execution.taker_fee,
        maker_fee=settings.execution.maker_fee,
        slippage_bps=settings.execution.slippage_bps,
        funding_rate=settings.backtest.funding_rate,
        risk_cfg=risk_cfg,
        interval_ms=INTERVAL_MS[settings.interval],
    ).run()

    # Gross run
    gross_res = BacktestEngine(
        df,
        decider,
        initial_equity=settings.backtest.initial_equity,
        taker_fee=0.0,
        maker_fee=0.0,
        slippage_bps=0.0,
        funding_rate=0.0,
        risk_cfg=risk_cfg,
        interval_ms=INTERVAL_MS[settings.interval],
    ).run()

    return compute_metrics(net_res["trades"], gross_res["metrics"], net_res["metrics"])


def main() -> int:
    settings = load_settings()
    setup_logging(settings.logging.log_level)

    store = CandleStore(settings.data.data_dir, network="mainnet")
    funding_store = FundingStore(settings.data.data_dir, network="mainnet")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
    raw_frames = {}
    for sym in symbols:
        df = store.load(sym, "60")
        if df is not None and not df.empty:
            raw_frames[sym] = df

    print("=" * 110)
    print("ALPHA RESEARCH V2: BASKET & TRIPLE-BARRIER BENCHMARK SWEEP")
    print(f"Available Symbols: {list(raw_frames.keys())}")
    print("=" * 110)

    results: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. TRIPLE-BARRIER CLASSIFIERS (BTC, ETH, SOL)
    # -------------------------------------------------------------------------
    for sym, df_raw in raw_frames.items():
        feat_df, f_cols = build_feature_frame(df_raw, settings.features)

        # Test 2 different barrier ratios:
        # Ratio 1: Balanced (TP=2.0x, SL=1.5x ATR, H=12)
        # Ratio 2: Trend Asymmetric (TP=3.0x, SL=1.5x ATR, H=24)
        for tp_m, sl_m, h_bars in [(2.0, 1.5, 12), (3.0, 1.5, 24)]:
            tb_df = (
                add_triple_barrier_labels(
                    feat_df,
                    tp_mult=tp_m,
                    sl_mult=sl_m,
                    max_holding_bars=h_bars,
                )
                .dropna(subset=["tb_label"])
                .reset_index(drop=True)
            )

            n_total = len(tb_df)
            n_holdout = min(2900, int(n_total * 0.15))
            labeled = tb_df.iloc[:-n_holdout].reset_index(drop=True)

            n_train = int(len(labeled) * 0.60)
            n_val = int(len(labeled) * 0.20)
            train_df = labeled.iloc[:n_train].reset_index(drop=True)
            val_df = labeled.iloc[n_train : n_train + n_val].reset_index(drop=True)
            test_df = labeled.iloc[n_train + n_val :].reset_index(drop=True)

            f_cols = [c for c in train_df.columns if c.startswith("f_")]

            lgb = train_lgbm(
                train_df[f_cols],
                train_df["tb_label"].astype(int),
                val_df[f_cols],
                val_df["tb_label"].astype(int),
                LgbmSettings(n_estimators=300, learning_rate=0.03, early_stopping_rounds=30),
                seed=42,
            )

            probs = lgb.predict_proba(test_df[f_cols])
            prob_map = {int(ts): p for ts, p in zip(test_df["ts_ms"].to_numpy(), probs)}

            risk_cfg = settings.risk.model_copy(
                update={
                    "stop_loss_atr_mult": sl_m,
                    "take_profit_atr_mult": tp_m,
                    "max_hold_bars": h_bars,
                }
            )

            for th in [0.38, 0.42, 0.46, 0.50]:

                def make_decider(th_val, p_map, r_cfg):
                    def _decide(row, state: PositionState) -> SignalDecision:
                        p = p_map[int(row["ts_ms"])]
                        p_long = float(p[CLASS_LONG_WIN])
                        p_short = float(p[CLASS_SHORT_WIN])
                        atr_v = float(row.get("f_atr_14", 0.02) * row["close"])

                        if state.direction != 0:
                            if state.bars_in_position >= r_cfg.max_hold_bars:
                                return SignalDecision(FLAT, ["timeout"], p_long, p_short, atr_v)
                            return SignalDecision("HOLD", ["hold"], p_long, p_short, atr_v)

                        if state.cooldown_bars_left > 0:
                            return SignalDecision(FLAT, ["cooldown"], p_long, p_short, atr_v)

                        if p_long > th_val and p_long > p_short:
                            return SignalDecision(OPEN_LONG, ["p_long"], p_long, p_short, atr_v)
                        if p_short > th_val and p_short > p_long:
                            return SignalDecision(OPEN_SHORT, ["p_short"], p_long, p_short, atr_v)
                        return SignalDecision(FLAT, ["none"], p_long, p_short, atr_v)

                    return _decide

                m = run_strategy(test_df, make_decider(th, prob_map, risk_cfg), risk_cfg, settings)
                m["name"] = f"tb_{sym}_tp{tp_m:.1f}_sl{sl_m:.1f}_th{th:.2f}"
                results.append(m)

    # -------------------------------------------------------------------------
    # 2. CROSS-SECTIONAL MOMENTUM & RESIDUAL ALPHA
    # -------------------------------------------------------------------------
    wide, syms = align_basket(raw_frames)
    if len(syms) >= 3:
        cs_feats = compute_cross_sectional_features(wide, syms, benchmark_symbol="BTCUSDT")

        # Test Cross-Sectional Leader Strategy for each altcoin
        for sym in [s for s in syms if s != "BTCUSDT"]:
            sym_df = (
                raw_frames[sym].merge(cs_feats[sym], on="ts_ms", how="inner").reset_index(drop=True)
            )
            n_tot = len(sym_df)
            test_cs = sym_df.iloc[int(n_tot * 0.70) :].reset_index(drop=True)

            risk_cs = settings.risk.model_copy(
                update={
                    "stop_loss_atr_mult": 3.0,
                    "take_profit_atr_mult": 10.0,
                    "max_hold_bars": 48,
                }
            )

            for rank_th in [0.75, 0.90]:

                def make_cs_decider(r_th, r_cfg):
                    def _decide(row, state: PositionState) -> SignalDecision:
                        rank = float(row.get("f_cs_rank_ret_24h", 0.5))
                        res_mom = float(row.get("f_cs_residual_mom", 0.0))
                        c = float(row["close"])
                        atr_v = c * 0.02

                        if state.direction != 0:
                            if state.bars_in_position >= r_cfg.max_hold_bars:
                                return SignalDecision(FLAT, ["time"], 0, 0, atr_v)
                            if rank < 0.50:
                                return SignalDecision(FLAT, ["rank_dropped"], 0, 0, atr_v)
                            return SignalDecision("HOLD", ["hold"], 0, 0, atr_v)

                        if rank >= r_th and res_mom > 0:
                            return SignalDecision(
                                OPEN_LONG, [f"top_rank_{rank:.2f}"], rank, 0, atr_v
                            )
                        return SignalDecision(FLAT, ["none"], 0, 0, atr_v)

                    return _decide

                m = run_strategy(test_cs, make_cs_decider(rank_th, risk_cs), risk_cs, settings)
                m["name"] = f"cs_lead_{sym}_top{int(rank_th * 100)}"
                results.append(m)

    # -------------------------------------------------------------------------
    # 3. FUNDING RATE CARRY & SQUEEZE STRATEGY
    # -------------------------------------------------------------------------
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]:
        fdf = funding_store.load(sym)
        cdf = raw_frames.get(sym)
        if fdf is not None and cdf is not None and len(fdf) > 50:
            f_merged = cdf.merge(fdf[["ts_ms", "funding_rate"]], on="ts_ms", how="left")
            f_merged["funding_rate"] = f_merged["funding_rate"].ffill().fillna(0.0001)
            # 30-day funding rolling z-score
            f_roll_mean = f_merged["funding_rate"].rolling(720, min_periods=100).mean()
            f_roll_std = (
                f_merged["funding_rate"].rolling(720, min_periods=100).std().replace(0, 1e-6)
            )
            f_merged["f_funding_zscore"] = (f_merged["funding_rate"] - f_roll_mean) / f_roll_std

            n_tot = len(f_merged)
            test_funding = f_merged.iloc[int(n_tot * 0.70) :].reset_index(drop=True)

            risk_fund = settings.risk.model_copy(
                update={"stop_loss_atr_mult": 4.0, "take_profit_atr_mult": 8.0, "max_hold_bars": 72}
            )

            for z_th in [-1.5, -2.0]:

                def make_fund_decider(z_val, r_cfg):
                    def _decide(row, state: PositionState) -> SignalDecision:
                        z = float(row.get("f_funding_zscore", 0.0))
                        c = float(row["close"])
                        atr_v = c * 0.02

                        if state.direction != 0:
                            if state.bars_in_position >= r_cfg.max_hold_bars:
                                return SignalDecision(FLAT, ["time"], 0, 0, atr_v)
                            if z > 0:
                                return SignalDecision(FLAT, ["funding_normalized"], 0, 0, atr_v)
                            return SignalDecision("HOLD", ["hold"], 0, 0, atr_v)

                        if z <= z_val:
                            return SignalDecision(
                                OPEN_LONG, [f"neg_funding_z_{z:.2f}"], 0, 0, atr_v
                            )
                        return SignalDecision(FLAT, ["none"], 0, 0, atr_v)

                    return _decide

                m = run_strategy(
                    test_funding, make_fund_decider(z_th, risk_fund), risk_fund, settings
                )
                m["name"] = f"funding_squeeze_{sym}_z{z_th:.1f}"
                results.append(m)

    # -------------------------------------------------------------------------
    # DISPLAY SUMMARY RESULTS
    # -------------------------------------------------------------------------
    print(
        f"\n{'Candidate':<32} | {'N':>4} | {'Gross bps (95% CI)':>22} | {'Net bps':>8} | {'Gross $':>8} | {'Net Ret':>8} | {'PF':>5} | {'Win%':>5} | {'Verdict':<12}"
    )
    print("-" * 125)

    pos_gross_count = 0
    stat_sig_gross = 0
    pos_net_count = 0

    for r in results:
        ci_str = f"[{r['ci95_gross_bps'][0]:+5.1f}, {r['ci95_gross_bps'][1]:+5.1f}]"
        gross_bps_str = f"{r['mean_gross_bps']:+5.1f} {ci_str}"
        has_gross_edge = r["mean_gross_bps"] > 0 and r["ci95_gross_bps"][0] > 0
        has_net_edge = r["net_return"] > 0 and r["profit_factor"] >= 1.0
        verdict = (
            "★ REAL EDGE"
            if (has_gross_edge and has_net_edge)
            else ("POS GROSS" if r["mean_gross_bps"] > 0 else "NO EDGE")
        )
        if r["mean_gross_bps"] > 0:
            pos_gross_count += 1
        if has_gross_edge:
            stat_sig_gross += 1
        if r["net_return"] > 0:
            pos_net_count += 1

        print(
            f"{r['name']:<32} | {r['n_trades']:>4} | {gross_bps_str:>22} | {r['mean_net_bps']:>+8.1f} | "
            f"{r['gross_pnl']:>8.1f} | {r['net_return']:>7.2%} | {r['profit_factor']:>5.2f} | "
            f"{r['win_rate']:>4.1%} | {verdict:<12}"
        )

    print("\n" + "=" * 110)
    print("ALPHA RESEARCH V2 SWEEP SUMMARY:")
    print(f"Total hypotheses tested: {len(results)}")
    print(f"Positive Gross Return: {pos_gross_count} / {len(results)}")
    print(f"Statistically Significant Gross Edge (95% CI > 0): {stat_sig_gross} / {len(results)}")
    print(
        f"Positive Net Return (after taker fee + slippage + funding): {pos_net_count} / {len(results)}"
    )
    print("=" * 110)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
