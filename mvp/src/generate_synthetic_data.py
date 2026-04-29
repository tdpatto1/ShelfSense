from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    CITY_EVENT_DESCRIPTIONS,
    DATA_DIR,
    FAMILIES,
    STORE_CITY,
    STORE_PROFILES,
    STORES,
    add_calendar_features,
    build_city_event_calendar,
    build_holiday_calendar,
    seed_everything,
)


FAMILY_BASE = {
    'produce': 168,
    'beverages': 150,
    'snacks': 132,
    'dairy': 116,
    'bakery': 97,
    'frozen': 82,
}

HOLIDAY_FAMILY_MULTIPLIER = {
    'produce': 1.18,
    'beverages': 1.30,
    'snacks': 1.26,
    'dairy': 1.10,
    'bakery': 1.24,
    'frozen': 1.11,
}

EVENT_FAMILY_MULTIPLIER = {
    'produce': 1.05,
    'beverages': 1.20,
    'snacks': 1.17,
    'dairy': 1.03,
    'bakery': 1.08,
    'frozen': 1.05,
}

PAYDAY_FAMILY_MULTIPLIER = {
    'produce': 1.03,
    'beverages': 1.09,
    'snacks': 1.08,
    'dairy': 1.04,
    'bakery': 1.05,
    'frozen': 1.06,
}

WEEKEND_FAMILY_MULTIPLIER = {
    'produce': 1.05,
    'beverages': 1.12,
    'snacks': 1.14,
    'dairy': 1.04,
    'bakery': 1.11,
    'frozen': 1.07,
}

HOLIDAY_NAME_MULTIPLIER = {
    'new_years_day': 0.94,
    'independence_day': 1.20,
    'halloween': 1.16,
    'christmas_eve': 1.24,
    'christmas_day': 0.84,
    'new_years_eve': 1.18,
    'thanksgiving': 1.26,
    'black_friday': 1.22,
    'super_bowl': 1.14,
    'memorial_day': 1.13,
    'labor_day': 1.12,
    'none': 1.0,
}

EVENT_NAME_MULTIPLIER = {
    'asu_move_in': 1.18,
    'tempe_festival_of_the_arts': 1.11,
    'tempe_marathon': 1.08,
    'asu_homecoming': 1.16,
    'state_fair': 1.17,
    'downtown_concert_series': 1.12,
    'sports_game': 1.16,
    'holiday_market': 1.10,
    'spring_training': 1.15,
    'mesa_fall_festival': 1.10,
    'holiday_lights': 1.09,
    'swap_meet_weekend': 1.07,
    'golf_tournament': 1.16,
    'art_walk': 1.09,
    'western_week': 1.12,
    'holiday_shopping_weekend': 1.13,
    'none': 1.0,
}

CITY_BASE_TEMP_F = {
    'Tempe': 79.0,
    'Phoenix': 81.0,
    'Mesa': 80.0,
    'Scottsdale': 78.0,
}


def make_store_calendar(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq='D')
    base_calendar = pd.DataFrame({'date': dates})
    base_calendar = add_calendar_features(base_calendar)
    base_calendar = base_calendar.merge(build_holiday_calendar(dates), on='date', how='left')

    store_frames = []
    for store_id in STORES:
        profile = STORE_PROFILES[store_id]
        city = profile['city']
        city_events = build_city_event_calendar(dates, city)
        store_calendar = base_calendar.merge(city_events, on='date', how='left')
        store_calendar['store_id'] = store_id
        store_calendar['city'] = city
        store_calendar['store_profile'] = profile['profile_label']

        day_of_year = store_calendar['date'].dt.dayofyear
        avg_temp_f = (
            CITY_BASE_TEMP_F[city]
            + 18 * np.sin(2 * np.pi * (day_of_year - 172) / 365.25)
            + profile['temp_offset_f']
        )
        store_calendar['avg_temp_f'] = avg_temp_f.round(1)
        store_calendar['temp_index'] = np.clip((store_calendar['avg_temp_f'] - 55) / 55, 0, 1.8)
        store_calendar['heatwave_flag'] = (store_calendar['avg_temp_f'] >= 104).astype(int)
        store_frames.append(store_calendar)

    return pd.concat(store_frames, ignore_index=True)


def make_store_transactions(calendar: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    records = []
    for store_id in STORES:
        profile = STORE_PROFILES[store_id]
        store_calendar = calendar[calendar['store_id'] == store_id].sort_values('date').reset_index(drop=True)
        trend = np.linspace(0.985, 1.055, len(store_calendar))
        for i, row in store_calendar.iterrows():
            base = 865 * profile['traffic_factor'] * trend[i]
            weekday_mult = [0.93, 0.96, 1.00, 1.03, 1.08, 1.15, 1.09][int(row['day_of_week'])]
            holiday_mult = 1.16 if row['holiday_flag'] else 1.0
            event_mult = EVENT_NAME_MULTIPLIER[row['local_event_name']] if row['local_event_flag'] else 1.0
            payday_mult = 1.06 if row['payday_flag'] else 1.0
            heatwave_mult = 1.03 if row['heatwave_flag'] else 1.0

            if row['holiday_name'] == 'christmas_day':
                holiday_mult = 0.76
            if row['holiday_name'] == 'new_years_day':
                holiday_mult = 0.88

            campus_lift = 1.05 if row['local_event_name'] == 'asu_move_in' and store_id == 'S001' else 1.0
            premium_lift = 1.04 if row['local_event_name'] == 'golf_tournament' and store_id == 'S004' else 1.0
            noise = rng.normal(0, 28 + 5 * profile['traffic_factor'])

            transactions = max(
                180,
                base * weekday_mult * holiday_mult * event_mult * payday_mult * heatwave_mult * campus_lift * premium_lift + noise,
            )
            records.append(
                {
                    'date': row['date'],
                    'store_id': store_id,
                    'city': row['city'],
                    'store_profile': row['store_profile'],
                    'transactions': round(transactions, 0),
                    'holiday_flag': row['holiday_flag'],
                    'holiday_name': row['holiday_name'],
                    'local_event_flag': row['local_event_flag'],
                    'local_event_name': row['local_event_name'],
                    'weekend_flag': row['weekend_flag'],
                    'payday_flag': row['payday_flag'],
                    'heatwave_flag': row['heatwave_flag'],
                    'avg_temp_f': row['avg_temp_f'],
                    'temp_index': row['temp_index'],
                    'day_of_week': row['day_of_week'],
                    'month': row['month'],
                    'day_name': row['day_name'],
                    'week_of_year': row['week_of_year'],
                    'quarter': row['quarter'],
                    'day_of_month': row['day_of_month'],
                }
            )
    return pd.DataFrame(records)


def family_temperature_multiplier(family: str, avg_temp_f: float, heatwave_flag: int) -> float:
    if family == 'beverages':
        return 1.0 + max(avg_temp_f - 75, 0) * 0.005 + 0.09 * heatwave_flag
    if family == 'frozen':
        return 1.0 + max(avg_temp_f - 75, 0) * 0.002
    if family == 'produce':
        return 1.0 + max(avg_temp_f - 78, 0) * 0.0015
    return 1.0


def promotion_probability(row: pd.Series, family: str) -> float:
    prob = 0.05
    if family in ['beverages', 'snacks']:
        prob += 0.08
    if family == 'produce':
        prob += 0.03
    if row['payday_flag']:
        prob += 0.05
    if row['weekend_flag']:
        prob += 0.04
    if row['holiday_flag']:
        prob += 0.06
    if row['local_event_flag']:
        prob += 0.04
    return float(np.clip(prob, 0.03, 0.42))


def make_sales_panel(transactions_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    records = []
    family_trend = {
        'produce': np.linspace(1.00, 1.03, transactions_df['date'].nunique()),
        'beverages': np.linspace(1.00, 1.06, transactions_df['date'].nunique()),
        'snacks': np.linspace(1.00, 1.04, transactions_df['date'].nunique()),
        'dairy': np.linspace(1.00, 1.02, transactions_df['date'].nunique()),
        'bakery': np.linspace(1.00, 1.015, transactions_df['date'].nunique()),
        'frozen': np.linspace(1.00, 1.04, transactions_df['date'].nunique()),
    }

    transactions_df = transactions_df.sort_values(['store_id', 'date']).reset_index(drop=True)
    unique_dates = np.sort(transactions_df['date'].unique())
    day_index = {dt: idx for idx, dt in enumerate(unique_dates)}

    for _, row in transactions_df.iterrows():
        t = day_index[row['date']]
        for family in FAMILIES:
            base = FAMILY_BASE[family] * STORE_PROFILES[row['store_id']]['traffic_factor']
            onpromotion = int(rng.random() < promotion_probability(row, family))
            promo_mult = 1.17 if onpromotion else 1.0
            holiday_mult = HOLIDAY_FAMILY_MULTIPLIER[family] if row['holiday_flag'] else 1.0
            holiday_name_mult = HOLIDAY_NAME_MULTIPLIER[row['holiday_name']]
            event_mult = EVENT_FAMILY_MULTIPLIER[family] if row['local_event_flag'] else 1.0
            event_name_mult = EVENT_NAME_MULTIPLIER[row['local_event_name']]
            payday_mult = PAYDAY_FAMILY_MULTIPLIER[family] if row['payday_flag'] else 1.0
            weekend_mult = WEEKEND_FAMILY_MULTIPLIER[family] if row['weekend_flag'] else 1.0
            temp_mult = family_temperature_multiplier(family, row['avg_temp_f'], row['heatwave_flag'])
            traffic_mult = 0.42 + 0.00073 * row['transactions']
            profile_mult = 1.05 if row['store_profile'] == 'premium suburban' and family in ['bakery', 'produce'] else 1.0
            noise = rng.normal(0, 8 + 0.025 * base)

            unit_sales = max(
                0,
                base
                * family_trend[family][t]
                * promo_mult
                * holiday_mult
                * holiday_name_mult
                * event_mult
                * event_name_mult
                * payday_mult
                * weekend_mult
                * temp_mult
                * traffic_mult
                * profile_mult
                + noise,
            )

            records.append(
                {
                    'date': row['date'],
                    'store_id': row['store_id'],
                    'city': row['city'],
                    'store_profile': row['store_profile'],
                    'family': family,
                    'unit_sales': round(unit_sales, 2),
                    'transactions': row['transactions'],
                    'onpromotion': onpromotion,
                    'holiday_flag': row['holiday_flag'],
                    'holiday_name': row['holiday_name'],
                    'local_event_flag': row['local_event_flag'],
                    'local_event_name': row['local_event_name'],
                    'weekend_flag': row['weekend_flag'],
                    'payday_flag': row['payday_flag'],
                    'heatwave_flag': row['heatwave_flag'],
                    'avg_temp_f': row['avg_temp_f'],
                    'temp_index': row['temp_index'],
                    'day_of_week': row['day_of_week'],
                    'day_name': row['day_name'],
                    'month': row['month'],
                    'week_of_year': row['week_of_year'],
                    'quarter': row['quarter'],
                    'day_of_month': row['day_of_month'],
                }
            )
    return pd.DataFrame(records)


def attach_future_promotions(future_sales_df: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = future_sales_df.copy()
    probs = df.apply(lambda row: promotion_probability(row, row['family']), axis=1).to_numpy()
    df['onpromotion'] = (rng.random(len(df)) < probs).astype(int)
    return df


def write_data_notes(output_dir: Path) -> None:
    lines = [
        '# ShelfSense Phase 2 Data Notes',
        '',
        'This Phase 2 dataset is still synthetic for reproducibility, but it is calibrated to match the ShelfSense proposal more closely than a generic random simulator.',
        '',
        '## Fidelity upgrades',
        '',
        '- Store-specific Arizona locations: Tempe, Phoenix, Mesa, and Scottsdale.',
        '- Store-specific local event calendar instead of one shared random event stream.',
        '- Arizona-like temperature curve with city offsets and heatwave flags.',
        '- Grocery-relevant holidays and long-weekend periods.',
        '- Promotions that are more likely around weekends, paydays, holidays, and event weekends.',
        '- Different store profiles to reflect campus, urban, suburban, and premium patterns.',
        '',
        '## Public-signal inspiration',
        '',
        'The calendar is designed to mirror the proposal idea of combining retail demand with outside signals such as holidays and local events. To keep the Phase 2 package reproducible without API keys, the event schedule is a deterministic proxy inspired by the kinds of signals the team proposed to pull from sources such as Nager.Date and Ticketmaster in later phases.',
        '',
        '## City event families included',
        '',
    ]
    for city, events in CITY_EVENT_DESCRIPTIONS.items():
        labels = ', '.join(f'{name} ({description})' for name, description in events)
        lines.append(f'- {city}: {labels}')
    (output_dir / 'data_notes.md').write_text('\n'.join(lines), encoding='utf-8')


def main(output_dir: Path | None = None, seed: int = 42) -> None:
    output_dir = Path(output_dir) if output_dir else DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = seed_everything(seed)

    history_calendar = make_store_calendar('2024-01-01', '2025-12-31')
    transactions_df = make_store_transactions(history_calendar, rng)
    sales_df = make_sales_panel(transactions_df, rng)

    future_calendar = make_store_calendar('2026-01-01', '2026-01-14')
    future_sales_df = []
    for family in FAMILIES:
        tmp = future_calendar.copy()
        tmp['family'] = family
        future_sales_df.append(tmp)
    future_sales_df = pd.concat(future_sales_df, ignore_index=True)
    future_sales_df = attach_future_promotions(future_sales_df, seed=seed + 11)
    future_sales_df = future_sales_df[
        [
            'date', 'store_id', 'city', 'store_profile', 'family', 'onpromotion', 'holiday_flag', 'holiday_name',
            'local_event_flag', 'local_event_name', 'weekend_flag', 'payday_flag', 'heatwave_flag', 'avg_temp_f',
            'temp_index', 'day_of_week', 'day_name', 'month', 'week_of_year', 'quarter', 'day_of_month'
        ]
    ]

    sales_df.to_csv(output_dir / 'historical_sales.csv', index=False)
    transactions_df.to_csv(output_dir / 'historical_transactions.csv', index=False)
    future_sales_df.to_csv(output_dir / 'future_calendar.csv', index=False)

    summary = pd.DataFrame(
        {
            'n_rows_sales': [len(sales_df)],
            'n_rows_transactions': [len(transactions_df)],
            'date_min': [sales_df['date'].min()],
            'date_max': [sales_df['date'].max()],
            'stores': [sales_df['store_id'].nunique()],
            'families': [sales_df['family'].nunique()],
            'event_days': [int(transactions_df['local_event_flag'].sum())],
            'mean_sales': [round(sales_df['unit_sales'].mean(), 2)],
            'mean_transactions': [round(transactions_df['transactions'].mean(), 2)],
        }
    )
    summary.to_csv(output_dir / 'dataset_summary.csv', index=False)
    write_data_notes(output_dir)

    print('Synthetic data generated:')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a calibrated Arizona grocery demand dataset for ShelfSense Phase 2.')
    parser.add_argument('--output-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(output_dir=args.output_dir, seed=args.seed)
