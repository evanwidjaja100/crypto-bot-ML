"""Phase 7.5: Rigorous gross edge evaluation across candidate configurations on clean test split.

Evaluates rulesets, linear baselines, and tree models on the test partition only.
Computes gross bps per trade, 95% confidence intervals, fee impact, trade counts,
profit factors, and top-5 trade return concentrations.

Never touches the holdout partition (reserved for post-selection burn-in / verification).
"""

import ctypes
import logging
import sys
from pathlib import Path
from typing import Any, Callable

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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.backtesting.engine import BacktestEngine
from src.config import RiskSettings, load_settings
from src.data_ingestion.intervals import INTERVAL_MS
from src.labels.dataset import assert_no_holdout_leak, load_dataset
from src.models.baseline import train_logistic
from src.models.train import train_lgbm
from src.monitoring.logging_setup import setup_logging
from src.strategy import rulesets
from src.strategy.signal_engine import PositionState, decide

log = logging.getLogger("evaluate_phase7_edge")


def compute_trade_metrics(
    trades_df: pd.DataFrame,
    gross_engine_metrics: dict[str, Any],
    net_engine_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compute trade-level gross & net statistics with 95% confidence intervals."""
    n_trades = len(trades_df) if trades_df is not None else 0
    if n_trades == 0:
        return {
            "n_trades": 0,
            "mean_gross_bps": 0.0,
            "ci95_gross_bps": (0.0, 0.0),
            "t_stat_gross": 0.0,
            "p_val_gross": 1.0,
            "mean_net_bps": 0.0,
            "ci95_net_bps": (0.0, 0.0),
            "gross_pnl": gross_engine_metrics.get("total_gross_pnl", 0.0),
            "net_pnl": net_engine_metrics.get("total_net_pnl", 0.0),
            "net_return": net_engine_metrics.get("total_return", 0.0),
            "profit_factor": net_engine_metrics.get("profit_factor", 0.0),
            "win_rate": net_engine_metrics.get("win_rate", 0.0),
            "sharpe": net_engine_metrics.get("sharpe", 0.0),
            "max_dd": net_engine_metrics.get("max_drawdown", 0.0),
            "total_fees": net_engine_metrics.get("total_fees", 0.0),
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
    gross_bps_arr = gross_ret * 10_000.0
    fee_bps_arr = np.where(notional > 0, (fees / notional) * 10_000.0, 0.0)
    net_bps_arr = gross_bps_arr - fee_bps_arr

    mean_gross = float(np.mean(gross_bps_arr))
    std_gross = float(np.std(gross_bps_arr, ddof=1)) if n_trades > 1 else 0.0
    se_gross = std_gross / np.sqrt(n_trades) if n_trades > 1 else 0.0
    ci_gross = (mean_gross - 1.96 * se_gross, mean_gross + 1.96 * se_gross)

    mean_net = float(np.mean(net_bps_arr))
    std_net = float(np.std(net_bps_arr, ddof=1)) if n_trades > 1 else 0.0
    se_net = std_net / np.sqrt(n_trades) if n_trades > 1 else 0.0
    ci_net = (mean_net - 1.96 * se_net, mean_net + 1.96 * se_net)

    if std_gross > 0:
        t_stat, p_val = stats.ttest_1samp(gross_bps_arr, 0.0)
    else:
        t_stat, p_val = 0.0, 1.0

    # Top-5 trades share of positive net PnL
    positive_pnls = [p for p in net_pnls if p > 0]
    total_pos_pnl = sum(positive_pnls)
    if total_pos_pnl > 0:
        top5_sum = sum(sorted(positive_pnls, reverse=True)[:5])
        top5_share = top5_sum / total_pos_pnl
    else:
        top5_share = 0.0

    return {
        "n_trades": n_trades,
        "mean_gross_bps": mean_gross,
        "ci95_gross_bps": ci_gross,
        "t_stat_gross": float(t_stat),
        "p_val_gross": float(p_val),
        "mean_net_bps": mean_net,
        "ci95_net_bps": ci_net,
        "gross_pnl": gross_engine_metrics.get("total_gross_pnl", 0.0),
        "net_pnl": net_engine_metrics.get("total_net_pnl", 0.0),
        "net_return": net_engine_metrics.get("total_return", 0.0),
        "profit_factor": net_engine_metrics.get("profit_factor", 0.0),
        "win_rate": net_engine_metrics.get("win_rate", 0.0),
        "sharpe": net_engine_metrics.get("sharpe", 0.0),
        "max_dd": net_engine_metrics.get("max_drawdown", 0.0),
        "total_fees": net_engine_metrics.get("total_fees", 0.0),
        "top5_share": top5_share,
    }


def run_candidate(
    name: str,
    test_df: pd.DataFrame,
    decider: Callable[[pd.Series, PositionState], tuple[int, str]],
    risk_cfg: RiskSettings,
    settings: Any,
) -> dict[str, Any]:
    """Runs a single candidate through both gross and net BacktestEngines."""
    # 1. Net execution (with full taker fee, slippage, funding)
    net_engine = BacktestEngine(
        test_df,
        decider,
        initial_equity=settings.backtest.initial_equity,
        taker_fee=settings.execution.taker_fee,
        maker_fee=settings.execution.maker_fee,
        slippage_bps=settings.execution.slippage_bps,
        funding_rate=settings.backtest.funding_rate,
        risk_cfg=risk_cfg,
        interval_ms=INTERVAL_MS[settings.interval],
    )
    net_res = net_engine.run()

    # 2. Gross execution (zero costs)
    gross_engine = BacktestEngine(
        test_df,
        decider,
        initial_equity=settings.backtest.initial_equity,
        taker_fee=0.0,
        maker_fee=0.0,
        slippage_bps=0.0,
        funding_rate=0.0,
        risk_cfg=risk_cfg,
        interval_ms=INTERVAL_MS[settings.interval],
    )
    gross_res = gross_engine.run()

    metrics = compute_trade_metrics(net_res["trades"], gross_res["metrics"], net_res["metrics"])
    metrics["name"] = name
    return metrics


def main() -> int:
    settings = load_settings()
    setup_logging(settings.logging.log_level)

    datasets_root = Path(settings.data.data_dir) / "datasets"
    candidates = sorted(
        datasets_root.glob("BTCUSDT_60_*/"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        print("No 60m dataset found. Run build_features.py first.")
        return 1
    dataset_dir = str(candidates[0])
    splits, meta = load_dataset(dataset_dir)
    assert_no_holdout_leak(splits["train"], splits.get("holdout"))

    test_df = splits["test"].reset_index(drop=True)
    val_df = splits["val"].reset_index(drop=True)
    train_df = splits["train"].reset_index(drop=True)
    feature_cols = [c for c in train_df.columns if c.startswith("f_")]

    print("=" * 100)
    print("PHASE 7.5 GROSS EDGE BENCHMARK SCAN")
    print(
        f"Dataset: {Path(dataset_dir).name} | Symbol: {settings.symbol} | Interval: {settings.interval}"
    )
    print(
        f"Partitions: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} | Holdout (Untouched)={len(splits.get('holdout', []))}"
    )
    print("=" * 100)

    results: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Trend Rulesets
    # -------------------------------------------------------------------------
    test_trend = (
        rulesets.add_trend_columns(test_df)
        .dropna(subset=rulesets.TREND_COLUMNS)
        .reset_index(drop=True)
    )
    trend_risk_fair = settings.risk.model_copy(
        update={"stop_loss_atr_mult": 5.0, "take_profit_atr_mult": 100.0}
    )
    trend_risk_std = settings.risk

    for rname, rdecider in rulesets.RULESETS.items():
        # Trend-fair risk (5xATR stop, trailing)
        res_fair = run_candidate(
            f"ruleset_{rname}_fair", test_trend, rdecider, trend_risk_fair, settings
        )
        results.append(res_fair)
        # Standard risk (1.5xATR stop, 3xATR TP)
        res_std = run_candidate(
            f"ruleset_{rname}_std", test_trend, rdecider, trend_risk_std, settings
        )
        results.append(res_std)

    # -------------------------------------------------------------------------
    # 2. Logistic Regression Baselines across thresholds & risk configs
    # -------------------------------------------------------------------------
    lr_model = train_logistic(
        train_df[feature_cols], train_df["label"].astype(int), seed=settings.model.seed
    )
    lr_proba = lr_model.predict_proba(test_df[feature_cols])
    lr_proba_by_ts = {int(ts): p for ts, p in zip(test_df["ts_ms"].to_numpy(), lr_proba)}

    thresholds = [0.34, 0.36, 0.38, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60]

    for th in thresholds:
        strat = settings.strategy.model_copy(
            update={
                "confidence_long": th,
                "confidence_short": th,
                "confidence_reverse": min(0.95, th + 0.05),
            }
        )

        def make_lr_decider(s_cfg, prob_map, r_cfg):
            return lambda row, state: decide(row, state, s_cfg, r_cfg, prob_map[int(row["ts_ms"])])

        # Standard risk
        res_lr_std = run_candidate(
            f"logistic_th{th:.2f}_std",
            test_df,
            make_lr_decider(strat, lr_proba_by_ts, settings.risk),
            settings.risk,
            settings,
        )
        results.append(res_lr_std)

        # Trend-following risk
        res_lr_trend = run_candidate(
            f"logistic_th{th:.2f}_trend",
            test_df,
            make_lr_decider(strat, lr_proba_by_ts, trend_risk_fair),
            trend_risk_fair,
            settings,
        )
        results.append(res_lr_trend)

    # -------------------------------------------------------------------------
    # 3. LightGBM Models across thresholds & risk configs
    # -------------------------------------------------------------------------
    lgb_model = train_lgbm(
        train_df[feature_cols],
        train_df["label"].astype(int),
        val_df[feature_cols],
        val_df["label"].astype(int),
        settings.model.lgbm,
        seed=settings.model.seed,
    )
    lgb_proba = lgb_model.predict_proba(test_df[feature_cols])
    lgb_proba_by_ts = {int(ts): p for ts, p in zip(test_df["ts_ms"].to_numpy(), lgb_proba)}

    for th in thresholds:
        strat = settings.strategy.model_copy(
            update={
                "confidence_long": th,
                "confidence_short": th,
                "confidence_reverse": min(0.95, th + 0.05),
            }
        )

        def make_lgb_decider(s_cfg, prob_map, r_cfg):
            return lambda row, state: decide(row, state, s_cfg, r_cfg, prob_map[int(row["ts_ms"])])

        # Standard risk
        res_lgb_std = run_candidate(
            f"lightgbm_th{th:.2f}_std",
            test_df,
            make_lgb_decider(strat, lgb_proba_by_ts, settings.risk),
            settings.risk,
            settings,
        )
        results.append(res_lgb_std)

        # Trend-following risk
        res_lgb_trend = run_candidate(
            f"lightgbm_th{th:.2f}_trend",
            test_df,
            make_lgb_decider(strat, lgb_proba_by_ts, trend_risk_fair),
            trend_risk_fair,
            settings,
        )
        results.append(res_lgb_trend)

    # -------------------------------------------------------------------------
    # Display Formatted Results Table
    # -------------------------------------------------------------------------
    print(
        f"\n{'Candidate':<24} | {'N':>4} | {'Gross bps (95% CI)':>22} | {'Net bps':>8} | {'Gross $':>8} | {'Net Ret':>8} | {'PF':>5} | {'Win%':>5} | {'Top5%':>6} | {'Verdict':<12}"
    )
    print("-" * 120)

    any_edge = False
    for r in results:
        ci_str = f"[{r['ci95_gross_bps'][0]:+5.1f}, {r['ci95_gross_bps'][1]:+5.1f}]"
        gross_bps_str = f"{r['mean_gross_bps']:+5.1f} {ci_str}"
        has_gross_edge = r["mean_gross_bps"] > 0 and r["ci95_gross_bps"][0] > 0
        has_net_edge = r["net_return"] > 0 and r["profit_factor"] >= 1.0
        verdict = (
            "GROSS EDGE"
            if has_gross_edge
            else ("POS GROSS" if r["mean_gross_bps"] > 0 else "NO EDGE")
        )
        if has_gross_edge and has_net_edge:
            verdict = "★ REAL EDGE"
            any_edge = True

        print(
            f"{r['name']:<24} | {r['n_trades']:>4} | {gross_bps_str:>22} | {r['mean_net_bps']:>+8.1f} | "
            f"{r['gross_pnl']:>8.1f} | {r['net_return']:>7.2%} | {r['profit_factor']:>5.2f} | "
            f"{r['win_rate']:>4.1%} | {r['top5_share']:>5.1%} | {verdict:<12}"
        )

    print("\n" + "=" * 100)
    print("SUMMARY VERDICT FOR PHASE 7.5 GATE:")
    print(f"Total configurations evaluated: {len(results)}")
    best_gross = max(results, key=lambda x: x["mean_gross_bps"])
    print(
        f"Best Gross bps/trade: {best_gross['name']} @ {best_gross['mean_gross_bps']:+.2f} bps "
        f"[{best_gross['ci95_gross_bps'][0]:+.2f}, {best_gross['ci95_gross_bps'][1]:+.2f}] (N={best_gross['n_trades']})"
    )

    pos_gross_count = sum(1 for r in results if r["mean_gross_bps"] > 0)
    stat_sig_gross_count = sum(1 for r in results if r["ci95_gross_bps"][0] > 0)
    pos_net_count = sum(1 for r in results if r["net_return"] > 0)
    print(f"Configs with positive gross return: {pos_gross_count} / {len(results)}")
    print(
        f"Configs with statistically significant gross edge (95% CI > 0): {stat_sig_gross_count} / {len(results)}"
    )
    print(f"Configs with net positive return after costs: {pos_net_count} / {len(results)}")
    print("=" * 100)

    return 0 if any_edge else 2


if __name__ == "__main__":
    raise SystemExit(main())
