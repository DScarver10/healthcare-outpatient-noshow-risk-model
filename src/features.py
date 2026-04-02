"""Feature engineering helpers used across notebooks."""

import pandas as pd


def add_missing_indicator(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Create a simple missing-value indicator for a selected column."""
    featured = df.copy()
    featured[f"{column}_was_missing"] = featured[column].isna().astype(int)
    return featured


def prepare_noshow_features(df: pd.DataFrame, target_col: str = "No-show") -> pd.DataFrame:
    """Build a modeling-ready feature set for the outpatient no-show dataset."""
    featured = df.copy()

    scheduled_dt = pd.to_datetime(featured["ScheduledDay"], errors="coerce", utc=True)
    appointment_dt = pd.to_datetime(featured["AppointmentDay"], errors="coerce", utc=True)

    lead_time_days = (appointment_dt - scheduled_dt).dt.total_seconds().div(24 * 60 * 60)

    featured["Age"] = featured["Age"].clip(lower=0)
    featured["lead_time_days"] = lead_time_days.clip(lower=0).fillna(0)
    featured["scheduled_hour"] = scheduled_dt.dt.hour.fillna(0).astype(int)
    featured["scheduled_weekday"] = scheduled_dt.dt.day_name().fillna("Unknown")
    featured["appointment_weekday"] = appointment_dt.dt.day_name().fillna("Unknown")
    featured["appointment_month"] = appointment_dt.dt.month.fillna(0).astype(int)
    featured["same_day_appointment"] = (featured["lead_time_days"] < 1).astype(int)
    featured["long_wait_30_plus"] = (featured["lead_time_days"] >= 30).astype(int)
    featured["age_group"] = pd.cut(
        featured["Age"],
        bins=[-1, 17, 34, 49, 64, 200],
        labels=["child", "young_adult", "adult", "mid_older", "older"],
    )

    featured[target_col] = featured[target_col].map({"No": 0, "Yes": 1}).astype(int)

    columns_to_drop = ["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"]
    featured = featured.drop(columns=columns_to_drop)

    return featured
