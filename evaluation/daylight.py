"""Original all-day/daylight evaluation definitions."""
from __future__ import annotations
from pathlib import Path
import math
import csv
import numpy as np
def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan

def masked_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    mape_min_target: float,
) -> dict[str, float | int]:
    p = pred[mask]
    y = true[mask]
    error = p - y
    abs_error = np.abs(error)
    mse = safe_mean(np.square(error))
    y_mean = safe_mean(y)
    denominator = float(np.sum(np.square(y - y_mean))) if y.size else math.nan
    r2 = 1.0 - float(np.sum(np.square(error))) / denominator if denominator > 0 else math.nan
    mape_mask = y > mape_min_target
    mape = (
        100.0 * safe_mean(np.abs((p[mape_mask] - y[mape_mask]) / y[mape_mask]))
        if np.any(mape_mask) else math.nan
    )
    bias = safe_mean(error)
    return {
        "n": int(y.size),
        "mae": safe_mean(abs_error),
        "mse": mse,
        "rmse": math.sqrt(mse) if np.isfinite(mse) else math.nan,
        "r2": r2,
        "bias": bias,
        "bias_abs": abs(bias),
        "mape_pct": mape,
        "mape_n": int(np.sum(mape_mask)),
    }

def prefix(values: dict[str, float | int], label: str) -> dict[str, float | int]:
    return {f"{label}_{key}": value for key, value in values.items()}

def calculate_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    daylight: np.ndarray,
    mape_min_target: float,
) -> dict[str, float | int]:
    all_mask = np.ones(true.shape, dtype=bool)
    daylight_mask = daylight.astype(bool)
    generating_mask = true > mape_min_target
    night_mask = ~daylight_mask
    return {
        **prefix(masked_metrics(pred, true, all_mask, mape_min_target), "all"),
        **prefix(masked_metrics(pred, true, daylight_mask, mape_min_target), "daylight"),
        **prefix(masked_metrics(pred, true, generating_mask, mape_min_target), "generating"),
        **prefix(masked_metrics(pred, true, night_mask, mape_min_target), "night"),
        "daylight_fraction": safe_mean(daylight_mask.astype(np.float64)),
        "generating_fraction": safe_mean(generating_mask.astype(np.float64)),
    }

def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames, seen = [], set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
