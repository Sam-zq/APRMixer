"""PV-specific dataset with leakage-safe scaling and pvlib physics features."""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset



PHYSICS_COLUMNS = (
    "clear_sky_power_cf",
    "daylight_mask",
    "solar_elevation_norm",
    "solar_azimuth_sin",
    "solar_azimuth_cos",
    "clear_sky_ghi_norm",
)


class Dataset_PV(Dataset):
    """PV dataset used by CSRampFormer.

    Weather covariates are clipped and standardized with training-only
    statistics. Power is converted to an unstandardized capacity factor and
    is always placed in the final channel. Physics features are computed for
    every timestamp but their empirical clear-sky scale is fitted only from
    the training partition.
    """

    def __init__(self, args, root_path, flag="train", size=None,
                 features="MS", data_path="7-7.csv", target="OT",
                 scale=True, timeenc=1, freq="15min",
                 seasonal_patterns=None):
        if size is None:
            size = [192, 0, 16]
        if flag not in ("train", "val", "test"):
            raise ValueError(f"Unsupported flag: {flag}")
        if features != "MS":
            raise ValueError("Dataset_PV requires --features MS")

        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len, self.label_len, self.pred_len = size
        self.target = target
        self.flag = flag
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.power_capacity = float(getattr(args, "power_capacity", 30.0))
        self.scale = False  # Outputs are already physical capacity factors.
        self.freq = freq
        self._read_data()

    @staticmethod
    def _localize_times(times, timezone):
        index = pd.DatetimeIndex(times)
        if index.tz is None:
            return index.tz_localize(
                timezone, ambiguous="infer", nonexistent="shift_forward"
            )
        return index.tz_convert(timezone)

    def _physics_features(self, times, power_cf, train_end):
        mode = getattr(self.args, "pv_physics_mode", "pvlib")
        if mode == "empirical_harmonic":
            return self._empirical_harmonic_physics(times, power_cf, train_end)
        if mode == "empirical":
            return self._empirical_physics(times, power_cf, train_end)
        try:
            from pvlib.location import Location
        except ImportError as exc:
            raise ImportError(
                "Dataset_PV requires pvlib. Install with: pip install pvlib"
            ) from exc

        latitude = float(getattr(self.args, "pv_latitude", 39.9))
        longitude = float(getattr(self.args, "pv_longitude", 116.4))
        altitude = float(getattr(self.args, "pv_altitude", 0.0))
        timezone = getattr(self.args, "pv_timezone", "Asia/Shanghai")
        localized = self._localize_times(times, timezone)
        location = Location(
            latitude, longitude, tz=timezone, altitude=altitude
        )
        solar = location.get_solarposition(localized)
        clearsky = location.get_clearsky(localized, model="ineichen")

        elevation = solar["apparent_elevation"].to_numpy(dtype=np.float64)
        azimuth_rad = np.deg2rad(solar["azimuth"].to_numpy(dtype=np.float64))
        clear_ghi = clearsky["ghi"].to_numpy(dtype=np.float64)
        base_envelope = np.clip(clear_ghi / 1000.0, 0.0, 1.5)

        # Site-specific conversion from clear-sky GHI to capacity factor.
        # Only training data contribute to this upper-envelope coefficient.
        train_base = base_envelope[:train_end]
        train_power = power_cf[:train_end]
        valid = (train_base > 0.05) & np.isfinite(train_power)
        ratio = train_power[valid] / np.maximum(train_base[valid], 1e-4)
        if ratio.size:
            envelope_scale = float(np.nanquantile(ratio, 0.98))
            envelope_scale = float(np.clip(envelope_scale, 0.2, 5.0))
        else:
            envelope_scale = 1.0

        clear_power_cf = np.clip(
            envelope_scale * base_envelope, 0.0, 1.0
        )
        daylight = (elevation > 0.0).astype(np.float64)
        physics = np.column_stack([
            clear_power_cf,
            daylight,
            np.clip(elevation / 90.0, -1.0, 1.0),
            np.sin(azimuth_rad),
            np.cos(azimuth_rad),
            np.clip(clear_ghi / 1200.0, 0.0, 1.5),
        ])
        return physics.astype(np.float32), envelope_scale

    @staticmethod
    def _empirical_physics(times, power_cf, train_end):
        """Build a coordinate-free clear-sky proxy from training data only."""
        index = pd.DatetimeIndex(times)
        train_index = index[:train_end]
        train_power = np.asarray(power_cf[:train_end], dtype=np.float64)
        train_slot = train_index.hour * 4 + train_index.minute // 15
        slot_envelope = np.zeros(96, dtype=np.float64)
        global_upper = float(np.nanquantile(train_power, 0.98))
        for slot_id in range(96):
            values = train_power[train_slot == slot_id]
            slot_envelope[slot_id] = (
                np.nanquantile(values, 0.98) if values.size else global_upper
            )
        padded = np.r_[slot_envelope[-2:], slot_envelope, slot_envelope[:2]]
        slot_envelope = np.convolve(padded, np.ones(5) / 5.0, mode="valid")

        monthly_scale = np.full(12, np.nan, dtype=np.float64)
        reference = max(float(np.nanquantile(train_power, 0.98)), 1e-6)
        for month in range(1, 13):
            values = train_power[train_index.month == month]
            if values.size:
                monthly_scale[month - 1] = np.nanquantile(values, 0.98) / reference
        monthly_scale = pd.Series(monthly_scale).interpolate(
            limit_direction="both"
        ).fillna(1.0).to_numpy()
        monthly_scale = np.clip(monthly_scale, 0.5, 1.5)

        slot = index.hour * 4 + index.minute // 15
        envelope = slot_envelope[slot] * monthly_scale[index.month - 1]
        envelope = np.clip(envelope, 0.0, 1.0)
        daylight = (envelope > 0.01).astype(np.float64)
        minute_angle = 2.0 * np.pi * (index.hour * 60 + index.minute) / 1440.0
        envelope_max = max(float(np.max(envelope)), 1e-6)
        elevation_proxy = np.clip(envelope / envelope_max, 0.0, 1.0)
        physics = np.column_stack([
            envelope,
            daylight,
            elevation_proxy,
            np.sin(minute_angle),
            np.cos(minute_angle),
            elevation_proxy,
        ])
        return physics.astype(np.float32), 1.0
    @staticmethod
    def _empirical_harmonic_physics(times, power_cf, train_end):
        """Coordinate-free upper envelope with continuous annual extrapolation."""
        index = pd.DatetimeIndex(times)
        train_index = index[:train_end]
        train_power = np.asarray(power_cf[:train_end], dtype=np.float64)
        train_slot = train_index.hour * 4 + train_index.minute // 15

        slot_envelope = np.zeros(96, dtype=np.float64)
        global_upper = float(np.nanquantile(train_power, 0.98))
        for slot_id in range(96):
            values = train_power[train_slot == slot_id]
            slot_envelope[slot_id] = (
                np.nanquantile(values, 0.98) if values.size else global_upper
            )
        padded = np.r_[slot_envelope[-2:], slot_envelope, slot_envelope[:2]]
        slot_envelope = np.convolve(padded, np.ones(5) / 5.0, mode="valid")

        # A rolling high quantile prevents individual cloudy days from defining
        # the seasonal curve. The harmonic fit extrapolates to unseen months.
        daily = pd.DataFrame({
            "day": train_index.normalize(),
            "power": train_power,
        }).groupby("day", sort=True)["power"].quantile(0.98)
        daily_upper = daily.rolling(
            window=21, center=True, min_periods=5
        ).quantile(0.90)
        reference = max(float(np.nanmedian(daily_upper)), 1e-6)
        response = np.asarray(daily_upper / reference, dtype=np.float64)
        day_index = pd.DatetimeIndex(daily_upper.index)
        angle = 2.0 * np.pi * (day_index.dayofyear.to_numpy() - 1.0) / 365.25
        design = np.column_stack([
            np.ones_like(angle), np.sin(angle), np.cos(angle)
        ])
        valid = np.isfinite(response)
        ridge = np.diag([1e-6, 0.25, 0.25])
        if valid.sum() >= 10:
            beta = np.linalg.solve(
                design[valid].T @ design[valid] + ridge,
                design[valid].T @ response[valid],
            )
        else:
            beta = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        all_angle = 2.0 * np.pi * (index.dayofyear.to_numpy() - 1.0) / 365.25
        all_design = np.column_stack([
            np.ones_like(all_angle), np.sin(all_angle), np.cos(all_angle)
        ])
        seasonal_scale = np.clip(all_design @ beta, 0.5, 1.5)
        train_scale_median = max(
            float(np.nanmedian(seasonal_scale[:train_end])), 1e-6
        )
        seasonal_scale = seasonal_scale / train_scale_median

        # Fit continuous sunrise and sunset curves from generating intervals.
        # This avoids applying summer daylight slots unchanged in winter.
        generating = train_power > max(0.005, 0.01 * global_upper)
        boundary_frame = pd.DataFrame({
            "day": train_index.normalize(),
            "slot": train_slot,
            "generating": generating,
        })
        active = boundary_frame[boundary_frame["generating"]]
        sunrise = active.groupby("day")["slot"].min().rolling(
            window=15, center=True, min_periods=5
        ).median()
        sunset = active.groupby("day")["slot"].max().rolling(
            window=15, center=True, min_periods=5
        ).median()

        def fit_boundary(series, fallback):
            boundary_days = pd.DatetimeIndex(series.index)
            boundary_angle = (
                2.0 * np.pi * (boundary_days.dayofyear.to_numpy() - 1.0)
                / 365.25
            )
            boundary_design = np.column_stack([
                np.ones_like(boundary_angle),
                np.sin(boundary_angle),
                np.cos(boundary_angle),
            ])
            boundary_values = np.asarray(series, dtype=np.float64)
            boundary_valid = np.isfinite(boundary_values)
            if boundary_valid.sum() < 10:
                return np.array([fallback, 0.0, 0.0], dtype=np.float64)
            return np.linalg.solve(
                boundary_design[boundary_valid].T
                @ boundary_design[boundary_valid]
                + np.diag([1e-6, 1.0, 1.0]),
                boundary_design[boundary_valid].T
                @ boundary_values[boundary_valid],
            )

        sunrise_beta = fit_boundary(sunrise, 24.0)
        sunset_beta = fit_boundary(sunset, 72.0)
        predicted_sunrise = np.clip(all_design @ sunrise_beta, 0.0, 48.0)
        predicted_sunset = np.clip(all_design @ sunset_beta, 48.0, 95.0)

        slot = index.hour * 4 + index.minute // 15
        daylight = (
            (slot >= predicted_sunrise) & (slot <= predicted_sunset)
        ).astype(np.float64)
        envelope = np.clip(
            slot_envelope[slot] * seasonal_scale * daylight, 0.0, 1.0
        )
        minute_angle = 2.0 * np.pi * (index.hour * 60 + index.minute) / 1440.0
        envelope_max = max(float(np.max(envelope)), 1e-6)
        elevation_proxy = np.clip(envelope / envelope_max, 0.0, 1.0)
        physics = np.column_stack([
            envelope,
            daylight,
            elevation_proxy,
            np.sin(minute_angle),
            np.cos(minute_angle),
            elevation_proxy,
        ])
        return physics.astype(np.float32), 1.0
    @staticmethod
    def _calendar_features(times):
        times = pd.DatetimeIndex(times)
        minute_of_day = times.hour * 60 + times.minute
        day_angle = 2.0 * np.pi * minute_of_day / 1440.0
        year_angle = 2.0 * np.pi * (times.dayofyear - 1) / 365.25
        week_angle = 2.0 * np.pi * times.dayofweek / 7.0
        return np.column_stack([
            np.sin(day_angle), np.cos(day_angle),
            np.sin(year_angle), np.cos(year_angle),
            np.sin(week_angle), np.cos(week_angle),
        ]).astype(np.float32)

    def _read_data(self):
        path = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(path)
        date_column = "date" if "date" in df.columns else df.columns[0]
        if self.target not in df.columns:
            raise KeyError(f"Target column {self.target!r} not found in {path}")

        times = pd.to_datetime(df[date_column], errors="coerce")
        if times.isna().any():
            raise ValueError(f"Invalid timestamps in {path}: {times.isna().sum()}")
        if times.duplicated().any():
            raise ValueError(f"Duplicated timestamps in {path}")
        if not times.is_monotonic_increasing:
            order = np.argsort(times.to_numpy())
            df = df.iloc[order].reset_index(drop=True)
            times = times.iloc[order].reset_index(drop=True)

        n = len(df)
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_val = n - num_train - num_test
        border1s = [0, num_train - self.seq_len,
                    n - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, n]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        power = pd.to_numeric(df[self.target], errors="coerce")
        # Small negative nighttime meter noise is non-generating power.
        power = power.clip(lower=0.0)
        train_power_median = float(power.iloc[:num_train].median())
        power = power.ffill().fillna(train_power_median)
        power_cf = np.clip(
            power.to_numpy(dtype=np.float64) / self.power_capacity,
            0.0, 1.0,
        )

        excluded = {date_column, self.target, *PHYSICS_COLUMNS}
        covariate_columns = [c for c in df.columns if c not in excluded]
        covariates = df[covariate_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        # Leakage-safe, forward-only imputation plus training medians.
        covariates = covariates.replace([np.inf, -np.inf], np.nan).ffill()
        train_cov = covariates.iloc[:num_train]
        medians = train_cov.median().fillna(0.0)
        covariates = covariates.fillna(medians)
        lower = train_cov.quantile(0.001).fillna(medians)
        upper = train_cov.quantile(0.999).fillna(medians)
        covariates = covariates.clip(lower=lower, upper=upper, axis=1)

        self.weather_scaler = StandardScaler()
        self.weather_scaler.fit(covariates.iloc[:num_train].to_numpy())
        scaled_covariates = self.weather_scaler.transform(
            covariates.to_numpy()
        ).astype(np.float32)

        mark_mode = getattr(self.args, "pv_mark_mode", "physics")
        if mark_mode == "physics":
            physics, self.envelope_scale = self._physics_features(
                times, power_cf, num_train
            )
            calendar = self._calendar_features(times)
            marks = np.concatenate([calendar, physics], axis=1)
            self.mark_names = [
                "sin_day", "cos_day", "sin_year", "cos_year",
                "sin_week", "cos_week", *PHYSICS_COLUMNS,
            ]
        else:
            raise ValueError(f"Unsupported pv_mark_mode: {mark_mode}")

        values = np.concatenate([
            scaled_covariates,
            power_cf[:, None].astype(np.float32),
        ], axis=1)

        self.feature_names = covariate_columns + ["power_capacity_factor"]
        self.data_x = values[border1:border2]
        self.data_y = values[border1:border2]
        self.data_stamp = marks[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # Target and model output are already capacity factors.
        return data

