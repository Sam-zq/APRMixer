"""Original ramp metric definitions; evaluation only."""
from __future__ import annotations
import math
import numpy as np
def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan

def safe_rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else math.nan

def safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return math.nan
    return float(numerator / denominator)

def binary_scores(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return precision, recall, f1

def calculate_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    daylight: np.ndarray,
    generation_threshold: float,
    ramp_threshold: float,
) -> dict[str, float | int]:
    error = pred - true
    abs_error = np.abs(error)
    daylight_mask = daylight.astype(bool)
    generating_mask = true > generation_threshold
    night_mask = ~daylight_mask

    result: dict[str, float | int] = {
        "n_points": int(true.size),
        "mae": safe_mean(abs_error),
        "mse": safe_mean(np.square(error)),
        "rmse": safe_rmse(error),
        "bias": safe_mean(error),
        "bias_abs": abs(safe_mean(error)),
        "capacity_nmae_pct": 100.0 * safe_mean(abs_error),
        "capacity_nrmse_pct": 100.0 * safe_rmse(error),
        "p90_abs_error": float(np.quantile(abs_error, 0.90)),
        "p95_abs_error": float(np.quantile(abs_error, 0.95)),
        "p99_abs_error": float(np.quantile(abs_error, 0.99)),
        "max_abs_error": float(np.max(abs_error)),
        "daylight_n": int(np.sum(daylight_mask)),
        "daylight_fraction": safe_mean(daylight_mask.astype(np.float64)),
        "generating_n": int(np.sum(generating_mask)),
        "generating_fraction": safe_mean(generating_mask.astype(np.float64)),
        "night_n": int(np.sum(night_mask)),
    }

    for name, mask in (("daylight", daylight_mask), ("generating", generating_mask), ("night", night_mask)):
        masked_error = error[mask]
        masked_true = true[mask]
        mae = safe_mean(np.abs(masked_error))
        rmse = safe_rmse(masked_error)
        result[f"{name}_mae"] = mae
        result[f"{name}_mse"] = safe_mean(np.square(masked_error))
        result[f"{name}_rmse"] = rmse
        result[f"{name}_bias"] = safe_mean(masked_error)
        result[f"{name}_target_mean"] = safe_mean(masked_true)
        result[f"{name}_nmae_mean_pct"] = 100.0 * safe_div(mae, safe_mean(masked_true))
        result[f"{name}_nrmse_mean_pct"] = 100.0 * safe_div(rmse, safe_mean(masked_true))

    if pred.shape[1] < 2:
        ramp_error = np.empty(0, dtype=np.float64)
        true_ramp = np.empty(0, dtype=np.float64)
        pred_ramp = np.empty(0, dtype=np.float64)
        ramp_daylight = np.empty(0, dtype=bool)
    else:
        pred_ramp = np.diff(pred, axis=1)
        true_ramp = np.diff(true, axis=1)
        ramp_error = pred_ramp - true_ramp
        ramp_daylight = daylight[:, 1:] | daylight[:, :-1]

    true_events = np.abs(true_ramp) >= ramp_threshold
    predicted_events = np.abs(pred_ramp) >= ramp_threshold
    precision, recall, f1 = binary_scores(predicted_events, true_events)
    event_direction = (
        safe_mean((np.sign(pred_ramp[true_events]) == np.sign(true_ramp[true_events])).astype(float))
        if np.any(true_events) else math.nan
    )
    direction_correct = (
        true_events
        & predicted_events
        & (np.sign(pred_ramp) == np.sign(true_ramp))
    )
    correct_count = int(np.sum(direction_correct))
    direction_aware_precision = safe_div(correct_count, int(np.sum(predicted_events)))
    direction_aware_recall = safe_div(correct_count, int(np.sum(true_events)))
    direction_aware_f1 = safe_div(
        2.0 * direction_aware_precision * direction_aware_recall,
        direction_aware_precision + direction_aware_recall,
    )
    result.update({
        "ramp_n": int(true_ramp.size),
        "ramp_mae": safe_mean(np.abs(ramp_error)),
        "ramp_rmse": safe_rmse(ramp_error),
        "ramp_bias": safe_mean(ramp_error),
        "ramp_daylight_mae": safe_mean(np.abs(ramp_error[ramp_daylight])),
        "ramp_true_events": int(np.sum(true_events)),
        "ramp_predicted_events": int(np.sum(predicted_events)),
        "ramp_precision": precision,
        "ramp_recall": recall,
        "ramp_f1": f1,
        "ramp_true_event_direction_accuracy": event_direction,
        "ramp_direction_aware_precision": direction_aware_precision,
        "ramp_direction_aware_recall": direction_aware_recall,
        "ramp_direction_aware_f1": direction_aware_f1,
    })
    return result
