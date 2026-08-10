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

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return (
            f"rows={self.n_rows} dup={self.n_duplicates} gaps={self.n_gaps} "
            f"oob={self.out_of_order} nan={self.n_nan} errors={len(self.errors)}"
        )


def validate_candles(
    df: pd.DataFrame,
    interval_ms: int,
    *,
    allow_gaps: bool = True,
) -> ValidationReport:
    """Validate an OHLCV frame. Errors are blocking; gaps are warnings by default."""
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

    return report
