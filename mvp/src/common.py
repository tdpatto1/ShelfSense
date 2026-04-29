from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
ARTIFACTS_DIR = ROOT / 'artifacts'

STORES = ['S001', 'S002', 'S003', 'S004']
FAMILIES = ['produce', 'beverages', 'snacks', 'dairy', 'bakery', 'frozen']

STORE_PROFILES = {
    'S001': {
        'city': 'Tempe',
        'traffic_factor': 1.00,
        'temp_offset_f': 1.5,
        'profile_label': 'campus corridor',
    },
    'S002': {
        'city': 'Phoenix',
        'traffic_factor': 1.18,
        'temp_offset_f': 3.0,
        'profile_label': 'urban flagship',
    },
    'S003': {
        'city': 'Mesa',
        'traffic_factor': 0.93,
        'temp_offset_f': 2.0,
        'profile_label': 'family suburban',
    },
    'S004': {
        'city': 'Scottsdale',
        'traffic_factor': 1.08,
        'temp_offset_f': 0.5,
        'profile_label': 'premium suburban',
    },
}

STORE_CITY = {store_id: profile['city'] for store_id, profile in STORE_PROFILES.items()}

CITY_EVENT_DESCRIPTIONS = {
    'Tempe': [
        ('asu_move_in', 'campus move-in and back-to-school surge'),
        ('tempe_festival_of_the_arts', 'downtown arts festival foot traffic'),
        ('tempe_marathon', 'race weekend convenience demand'),
        ('asu_homecoming', 'football and alumni traffic'),
    ],
    'Phoenix': [
        ('state_fair', 'fairgrounds traffic and concession demand'),
        ('downtown_concert_series', 'arena and concert district traffic'),
        ('sports_game', 'large game-day crowd effects'),
        ('holiday_market', 'seasonal downtown shopping crowds'),
    ],
    'Mesa': [
        ('spring_training', 'spring baseball tourism and snack demand'),
        ('mesa_fall_festival', 'community festival weekend'),
        ('holiday_lights', 'family event traffic'),
        ('swap_meet_weekend', 'weekend family shopping traffic'),
    ],
    'Scottsdale': [
        ('golf_tournament', 'destination sports traffic'),
        ('art_walk', 'gallery district tourism'),
        ('western_week', 'parade and rodeo foot traffic'),
        ('holiday_shopping_weekend', 'high-income seasonal shopping lift'),
    ],
}


@dataclass
class Metrics:
    model_name: str
    target: str
    mae: float
    rmse: float
    wmape: float
    event_day_mae: float
    non_event_day_mae: float

    def to_dict(self) -> Dict[str, float | str]:
        return {
            'model_name': self.model_name,
            'target': self.target,
            'mae': self.mae,
            'rmse': self.rmse,
            'wmape': self.wmape,
            'event_day_mae': self.event_day_mae,
            'non_event_day_mae': self.non_event_day_mae,
        }


def seed_everything(seed: int = 42) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def us_payday_flag(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    return ((dates.dt.day <= 2) | (dates.dt.day.between(14, 16))).astype(int)


def add_calendar_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out['day_of_week'] = out[date_col].dt.dayofweek
    out['day_name'] = out[date_col].dt.day_name()
    out['week_of_year'] = out[date_col].dt.isocalendar().week.astype(int)
    out['month'] = out[date_col].dt.month
    out['day_of_month'] = out[date_col].dt.day
    out['quarter'] = out[date_col].dt.quarter
    out['weekend_flag'] = (out['day_of_week'] >= 5).astype(int)
    out['payday_flag'] = us_payday_flag(out[date_col])
    return out


def nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int) -> pd.Timestamp:
    month_days = pd.date_range(f'{year}-{month:02d}-01', periods=31, freq='D')
    month_days = month_days[month_days.month == month]
    return month_days[month_days.weekday == weekday][occurrence - 1]


def last_weekday_of_month(year: int, month: int, weekday: int) -> pd.Timestamp:
    month_days = pd.date_range(f'{year}-{month:02d}-01', periods=31, freq='D')
    month_days = month_days[month_days.month == month]
    return month_days[month_days.weekday == weekday][-1]


def build_holiday_calendar(dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    dates = pd.to_datetime(pd.Series(list(dates))).drop_duplicates().sort_values()
    records: List[Dict[str, object]] = []
    for year in sorted(dates.dt.year.unique()):
        fixed = {
            pd.Timestamp(year=year, month=1, day=1): 'new_years_day',
            pd.Timestamp(year=year, month=7, day=4): 'independence_day',
            pd.Timestamp(year=year, month=10, day=31): 'halloween',
            pd.Timestamp(year=year, month=12, day=24): 'christmas_eve',
            pd.Timestamp(year=year, month=12, day=25): 'christmas_day',
            pd.Timestamp(year=year, month=12, day=31): 'new_years_eve',
        }
        for dt, name in fixed.items():
            records.append({'date': dt, 'holiday_flag': 1, 'holiday_name': name})

        thanksgiving = nth_weekday_of_month(year, 11, 3, 4)
        black_friday = thanksgiving + pd.Timedelta(days=1)
        memorial_day = last_weekday_of_month(year, 5, 0)
        labor_day = nth_weekday_of_month(year, 9, 0, 1)
        super_bowl = nth_weekday_of_month(year, 2, 6, 2)
        records.extend(
            [
                {'date': thanksgiving, 'holiday_flag': 1, 'holiday_name': 'thanksgiving'},
                {'date': black_friday, 'holiday_flag': 1, 'holiday_name': 'black_friday'},
                {'date': memorial_day, 'holiday_flag': 1, 'holiday_name': 'memorial_day'},
                {'date': labor_day, 'holiday_flag': 1, 'holiday_name': 'labor_day'},
                {'date': super_bowl, 'holiday_flag': 1, 'holiday_name': 'super_bowl'},
            ]
        )
    cal = pd.DataFrame(records).drop_duplicates(subset=['date'])
    if cal.empty:
        return pd.DataFrame({'date': dates, 'holiday_flag': 0, 'holiday_name': 'none'})
    merged = pd.DataFrame({'date': dates}).merge(cal, on='date', how='left')
    merged['holiday_flag'] = merged['holiday_flag'].fillna(0).astype(int)
    merged['holiday_name'] = merged['holiday_name'].fillna('none')
    return merged


def build_city_event_calendar(dates: Iterable[pd.Timestamp], city: str) -> pd.DataFrame:
    dates = pd.to_datetime(pd.Series(list(dates))).drop_duplicates().sort_values()
    records: List[Dict[str, object]] = []
    for year in sorted(dates.dt.year.unique()):
        if city == 'Tempe':
            event_dates = {
                nth_weekday_of_month(year, 2, 6, 4): 'tempe_marathon',
                nth_weekday_of_month(year, 3, 4, 2): 'tempe_festival_of_the_arts',
                nth_weekday_of_month(year, 8, 5, 2): 'asu_move_in',
                nth_weekday_of_month(year, 10, 5, 3): 'asu_homecoming',
            }
        elif city == 'Phoenix':
            event_dates = {
                nth_weekday_of_month(year, 2, 6, 2): 'sports_game',
                nth_weekday_of_month(year, 10, 5, 2): 'state_fair',
                nth_weekday_of_month(year, 10, 5, 3): 'state_fair',
                nth_weekday_of_month(year, 11, 4, 1): 'downtown_concert_series',
                nth_weekday_of_month(year, 12, 5, 1): 'holiday_market',
            }
        elif city == 'Mesa':
            event_dates = {
                nth_weekday_of_month(year, 3, 5, 2): 'spring_training',
                nth_weekday_of_month(year, 4, 5, 1): 'swap_meet_weekend',
                nth_weekday_of_month(year, 10, 5, 4): 'mesa_fall_festival',
                nth_weekday_of_month(year, 12, 5, 2): 'holiday_lights',
            }
        else:
            event_dates = {
                nth_weekday_of_month(year, 1, 5, 4): 'western_week',
                nth_weekday_of_month(year, 2, 5, 1): 'golf_tournament',
                nth_weekday_of_month(year, 3, 5, 1): 'art_walk',
                nth_weekday_of_month(year, 12, 5, 2): 'holiday_shopping_weekend',
            }

        for dt, name in event_dates.items():
            records.append({'date': dt, 'local_event_flag': 1, 'local_event_name': name})

    event_df = pd.DataFrame(records).drop_duplicates(subset=['date'], keep='last') if records else pd.DataFrame()
    merged = pd.DataFrame({'date': dates}).merge(event_df, on='date', how='left')
    merged['local_event_flag'] = merged['local_event_flag'].fillna(0).astype(int)
    merged['local_event_name'] = merged['local_event_name'].fillna('none')
    return merged


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float('nan')
    return float(np.abs(y_true - y_pred).sum() / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_flags: np.ndarray,
    model_name: str,
    target: str,
) -> Metrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    event_flags = np.asarray(event_flags).astype(bool)
    return Metrics(
        model_name=model_name,
        target=target,
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        wmape=wmape(y_true, y_pred),
        event_day_mae=mae(y_true[event_flags], y_pred[event_flags]) if event_flags.any() else float('nan'),
        non_event_day_mae=mae(y_true[~event_flags], y_pred[~event_flags]) if (~event_flags).any() else float('nan'),
    )


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
