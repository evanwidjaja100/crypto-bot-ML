"""Phase 3: candle data validation (no silent corruption in the cache)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = ("ts_ms", "open", "high", "low", "close", "volume")

# Only minute-level candles have a guaranteed constant spacing (D/W/M are exempt).
SPACING_CHECK_MAX_MS = 86_400_000 - 1


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_duplicates: int = 0
    n_gaps: int = 0
    first_gap_ms: int | None = None
    out_of_order: int = 0
    n_nan: int = 0
    n_price_jumps: int = 0
    first_jump_ms: int | None = None
    max_bar_move_pct: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return (
            f"rows={self.n_rows} dup={self.n_duplicates} gaps={self.n_gaps} "
            f"oob={self.out_of_order} nan={self.n_nan} jumps={self.n_price_jumps} "
            f"max_move={self.max_bar_move_pct:.1f}% errors={len(self.errors)}"
        )


def validate_candles(
    df: pd.DataFrame,
    interval_ms: int,
    *,
    allow_gaps: bool = True,
    max_bar_move_pct: float | None = 25.0,
) -> ValidationReport:
    """Validate an OHLCV frame. Errors are blocking; gaps are warnings by default.

    max_bar_move_pct: blocking threshold for bar-to-bar close moves (percent).
    None disables the jump check — an explicit operator escape hatch.
    """
    report = ValidationReport()
    if df is None or df.empty:
        report.errors.append("empty DataFrame")
        return report

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.errors.append(f"missing columns: {missing}")
        return report

    d = df[list(REQUIRED_COLUMNS)]
    report.n_rows = len(d)

    nan_mask = d.isna()
    if nan_mask.any().any():
        report.n_nan = int(nan_mask.sum().sum())
        report.errors.append(f"{report.n_nan} NaN values in required columns")

    ts = d["ts_ms"]
    diff = ts.diff().dropna()
    report.out_of_order = int((diff < 0).sum())
    if report.out_of_order:
        report.errors.append(f"{report.out_of_order} out-of-order timestamps")
    report.n_duplicates = int(ts.duplicated().sum())
    if report.n_duplicates:
        report.errors.append(f"{report.n_duplicates} duplicate timestamps")

    if (d["high"] < d["low"]).any():
        report.errors.append("rows where high < low")
    if (d["low"] <= 0).any():
        report.errors.append("rows with non-positive low")
    if ((d["close"] > d["high"]) | (d["close"] < d["low"])).any():
        report.errors.append("rows where close outside [low, high]")
    if (d["volume"] < 0).any():
        report.errors.append("rows with negative volume")

    if interval_ms <= SPACING_CHECK_MAX_MS:
        big = diff[diff > interval_ms]
        report.n_gaps = len(big)
        if report.n_gaps:
            report.first_gap_ms = int(ts.loc[big.index[0]])
            msg = f"{report.n_gaps} candle gaps (first at ts={report.first_gap_ms})"
            (report.warnings if allow_gaps else report.errors).append(msg)

    if max_bar_move_pct is not None and interval_ms <= SPACING_CHECK_MAX_MS and len(d) > 1:
        move = d["close"].pct_change(fill_method=None).abs() * 100.0
        report.max_bar_move_pct = float(move.max(skipna=True) or 0.0)
        bad = move[move > max_bar_move_pct]
        if len(bad):
            report.n_price_jumps = len(bad)
            report.first_jump_ms = int(ts.loc[bad.index[0]])
            report.errors.append(
                f"{len(bad)} bar-to-bar close moves > {max_bar_move_pct}% "
                f"(first at ts={report.first_jump_ms}, max {report.max_bar_move_pct:.1f}%)"
            )

    return report
