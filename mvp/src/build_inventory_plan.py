from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ARTIFACTS_DIR


CASE_PACKS = {
    'produce': 12,
    'beverages': 24,
    'snacks': 18,
    'dairy': 12,
    'bakery': 8,
    'frozen': 10,
}

SAFETY_STOCK_FACTORS = {
    'produce': 0.12,
    'beverages': 0.18,
    'snacks': 0.16,
    'dairy': 0.12,
    'bakery': 0.10,
    'frozen': 0.13,
}


def ceil_to_case_pack(units: float, case_pack: int) -> int:
    if units <= 0:
        return 0
    return int(np.ceil(units / case_pack) * case_pack)


def action_label(delta_units: float, recommended_order: int, uplift_pct: float) -> str:
    if recommended_order > 0 and uplift_pct >= 12:
        return 'Increase order'
    if recommended_order > 0:
        return 'Replenish'
    if delta_units < -12:
        return 'Trim order'
    return 'Hold steady'


def priority_label(uplift_pct: float, event_flag: int, recommended_order: int) -> str:
    if recommended_order > 0 and (uplift_pct >= 25 or event_flag):
        return 'High'
    if recommended_order > 0 or uplift_pct >= 12:
        return 'Medium'
    return 'Normal'


def driver_text(row: pd.Series) -> str:
    drivers = []
    if row['holiday_name'] != 'none':
        drivers.append(row['holiday_name'].replace('_', ' '))
    if row['local_event_name'] != 'none':
        drivers.append(row['local_event_name'].replace('_', ' '))
    if row['onpromotion']:
        drivers.append('promotion')
    if row['heatwave_flag']:
        drivers.append('heat')
    if row['payday_flag']:
        drivers.append('payday')
    return ', '.join(drivers) if drivers else 'baseline demand'


def build_daily_plan(forecast: pd.DataFrame) -> pd.DataFrame:
    plan = forecast.copy()
    plan['delta_units'] = plan['pred_sales'] - plan['baseline_sales']
    plan['uplift_pct'] = (plan['delta_units'] / plan['baseline_sales']) * 100
    plan['case_pack'] = plan['family'].map(CASE_PACKS)
    plan['safety_stock_units'] = plan['pred_sales'] * plan['family'].map(SAFETY_STOCK_FACTORS)

    # Phase 3 MVP inventory proxy: estimate on-hand stock from recent baseline so the app can create an order plan
    # without requiring a private store inventory database.
    plan['estimated_on_hand_units'] = (plan['baseline_sales'] * 1.10).round(1)
    plan['recommended_order_units'] = (
        plan['pred_sales'] + plan['safety_stock_units'] - plan['estimated_on_hand_units']
    )
    plan['recommended_order_units'] = [
        ceil_to_case_pack(units, case_pack)
        for units, case_pack in zip(plan['recommended_order_units'], plan['case_pack'])
    ]
    plan['event_or_holiday_flag'] = ((plan['holiday_flag'] == 1) | (plan['local_event_flag'] == 1)).astype(int)
    plan['priority'] = [
        priority_label(uplift, event_flag, order_units)
        for uplift, event_flag, order_units in zip(
            plan['uplift_pct'], plan['event_or_holiday_flag'], plan['recommended_order_units']
        )
    ]
    plan['action'] = [
        action_label(delta, order_units, uplift)
        for delta, order_units, uplift in zip(
            plan['delta_units'], plan['recommended_order_units'], plan['uplift_pct']
        )
    ]
    plan['drivers'] = plan.apply(driver_text, axis=1)
    return plan


def build_summary(plan: pd.DataFrame) -> pd.DataFrame:
    summary = (
        plan.groupby(['store_id', 'city', 'store_profile', 'family'], as_index=False)
        .agg(
            forecast_units_14d=('pred_sales', 'sum'),
            baseline_units_14d=('baseline_sales', 'sum'),
            recommended_order_units=('recommended_order_units', 'sum'),
            event_days=('event_or_holiday_flag', 'sum'),
            promo_days=('onpromotion', 'sum'),
            peak_daily_units=('pred_sales', 'max'),
            avg_pred_transactions=('pred_transactions', 'mean'),
        )
    )
    summary['delta_units_14d'] = summary['forecast_units_14d'] - summary['baseline_units_14d']
    summary['uplift_pct_14d'] = (summary['delta_units_14d'] / summary['baseline_units_14d']) * 100
    summary['priority'] = summary.apply(
        lambda row: priority_label(row['uplift_pct_14d'], int(row['event_days'] > 0), int(row['recommended_order_units'])),
        axis=1,
    )
    summary = summary.sort_values(['priority', 'recommended_order_units'], ascending=[True, False])
    return summary.round(
        {
            'forecast_units_14d': 1,
            'baseline_units_14d': 1,
            'delta_units_14d': 1,
            'uplift_pct_14d': 1,
            'peak_daily_units': 1,
            'avg_pred_transactions': 0,
        }
    )


def write_markdown_summary(summary: pd.DataFrame, out_path: Path) -> None:
    top = summary.sort_values(['recommended_order_units', 'uplift_pct_14d'], ascending=False).head(8)
    lines = [
        '# ShelfSense MVP Inventory Summary',
        '',
        'This file summarizes the highest-priority store and family combinations from the generated 14-day forecast.',
        '',
        '| Store | Family | Forecast Units | Uplift % | Recommended Order | Priority |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['store_id']} ({row['city']}) | {row['family']} | {row['forecast_units_14d']:.1f} | "
            f"{row['uplift_pct_14d']:.1f}% | {row['recommended_order_units']:.0f} | {row['priority']} |"
        )
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def main(artifacts_dir: Path | None = None) -> None:
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    forecast = pd.read_csv(artifacts_dir / 'future_sales_forecast.csv', parse_dates=['date'])
    daily_plan = build_daily_plan(forecast)
    summary = build_summary(daily_plan)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    daily_plan.to_csv(artifacts_dir / 'inventory_recommendations.csv', index=False)
    summary.to_csv(artifacts_dir / 'inventory_summary.csv', index=False)
    write_markdown_summary(summary, artifacts_dir / 'inventory_summary.md')
    print(summary.head(10).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build ShelfSense MVP inventory recommendations from forecasts.')
    parser.add_argument('--artifacts-dir', type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()
    main(artifacts_dir=args.artifacts_dir)
