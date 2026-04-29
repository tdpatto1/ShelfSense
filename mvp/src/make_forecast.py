from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR
from modeling import (
    SALES_CATEGORICAL_FEATURES,
    SALES_NUMERIC_FEATURES,
    TRANSACTION_CATEGORICAL_FEATURES,
    TRANSACTION_NUMERIC_FEATURES,
    load_model,
)


def _tx_lag_features(history: pd.DataFrame, store_id: str, current_date: pd.Timestamp) -> Dict[str, float]:
    store_hist = history[history['store_id'] == store_id].sort_values('date')
    out: Dict[str, float] = {}
    for lag in [1, 7, 14]:
        target_date = current_date - pd.Timedelta(days=lag)
        vals = store_hist.loc[store_hist['date'] == target_date, 'transactions']
        out[f'lag_transactions_{lag}'] = float(vals.iloc[-1]) if not vals.empty else np.nan
    last7 = store_hist.loc[store_hist['date'] < current_date].tail(7)['transactions']
    last28 = store_hist.loc[store_hist['date'] < current_date].tail(28)['transactions']
    out['rolling_transactions_7'] = float(last7.mean()) if len(last7) else np.nan
    out['rolling_transactions_28'] = float(last28.mean()) if len(last28) else np.nan
    return out


def _sales_lag_features(
    sales_history: pd.DataFrame,
    tx_history: pd.DataFrame,
    store_id: str,
    family: str,
    current_date: pd.Timestamp,
) -> Dict[str, float]:
    series_hist = sales_history[(sales_history['store_id'] == store_id) & (sales_history['family'] == family)].sort_values('date')
    tx_hist = tx_history[tx_history['store_id'] == store_id].sort_values('date')
    out: Dict[str, float] = {}
    for lag in [1, 7, 14]:
        target_date = current_date - pd.Timedelta(days=lag)
        vals = series_hist.loc[series_hist['date'] == target_date, 'unit_sales']
        out[f'lag_sales_{lag}'] = float(vals.iloc[-1]) if not vals.empty else np.nan
    last7 = series_hist.loc[series_hist['date'] < current_date].tail(7)['unit_sales']
    last28 = series_hist.loc[series_hist['date'] < current_date].tail(28)['unit_sales']
    out['rolling_sales_7'] = float(last7.mean()) if len(last7) else np.nan
    out['rolling_sales_28'] = float(last28.mean()) if len(last28) else np.nan

    for lag in [1, 7]:
        target_date = current_date - pd.Timedelta(days=lag)
        vals = tx_hist.loc[tx_hist['date'] == target_date, 'transactions']
        out[f'lag_transactions_{lag}'] = float(vals.iloc[-1]) if not vals.empty else np.nan
    last_tx7 = tx_hist.loc[tx_hist['date'] < current_date].tail(7)['transactions']
    out['rolling_transactions_7'] = float(last_tx7.mean()) if len(last_tx7) else np.nan
    return out


def predict_transactions(
    history_tx: pd.DataFrame,
    future_calendar: pd.DataFrame,
    transactions_model,
) -> pd.DataFrame:
    tx_history = history_tx[['date', 'store_id', 'city', 'transactions']].copy()
    store_calendar = (
        future_calendar[['date', 'store_id', 'city', 'store_profile', 'holiday_flag', 'holiday_name', 'local_event_flag', 'local_event_name',
                         'weekend_flag', 'payday_flag', 'heatwave_flag', 'avg_temp_f', 'temp_index', 'day_of_week', 'day_name',
                         'month', 'week_of_year', 'quarter', 'day_of_month']]
        .drop_duplicates(subset=['date', 'store_id'])
        .sort_values(['date', 'store_id'])
    )
    preds: List[Dict] = []
    for _, row in store_calendar.iterrows():
        lag_feats = _tx_lag_features(tx_history, row['store_id'], row['date'])
        feature_row = row.to_dict() | lag_feats
        x = pd.DataFrame([feature_row])
        pred = float(np.clip(transactions_model.predict(x[TRANSACTION_NUMERIC_FEATURES + TRANSACTION_CATEGORICAL_FEATURES])[0], 0, None))
        record = row.to_dict()
        record['pred_transactions'] = pred
        preds.append(record)
        tx_history = pd.concat(
            [
                tx_history,
                pd.DataFrame(
                    [
                        {
                            'date': row['date'],
                            'store_id': row['store_id'],
                            'city': row['city'],
                            'transactions': pred,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return pd.DataFrame(preds)


def explain_row(pred_sales: float, baseline_sales: float, row: pd.Series) -> str:
    uplift = ((pred_sales - baseline_sales) / baseline_sales) if baseline_sales > 1e-6 else 0.0
    drivers = []
    if row['holiday_flag']:
        drivers.append(row['holiday_name'].replace('_', ' '))
    if row['local_event_flag']:
        drivers.append(row['local_event_name'].replace('_', ' '))
    if row['onpromotion']:
        drivers.append('promotion activity')
    if row['weekend_flag']:
        drivers.append('weekend traffic')
    if row['payday_flag']:
        drivers.append('payday effect')
    if row['heatwave_flag'] and row['family'] in ['beverages', 'frozen']:
        drivers.append(f'hot weather ({row["avg_temp_f"]:.0f}F)')

    if uplift > 0.12:
        tone = 'above-normal'
    elif uplift < -0.08:
        tone = 'below-normal'
    else:
        tone = 'near-normal'

    if drivers:
        reasons = ', '.join(drivers[:3])
        return f'Expect {tone} {row["family"]} demand because of {reasons}.'
    return f'Expect {tone} demand with no major external spike drivers flagged.'


def predict_sales(
    history_sales: pd.DataFrame,
    predicted_tx: pd.DataFrame,
    future_calendar: pd.DataFrame,
    sales_model,
) -> pd.DataFrame:
    sales_history = history_sales[['date', 'store_id', 'family', 'unit_sales']].copy()
    tx_history = history_sales[['date', 'store_id', 'transactions']].drop_duplicates(subset=['date', 'store_id']).copy()
    tx_history = tx_history.rename(columns={'transactions': 'transactions'})
    tx_history = pd.concat(
        [
            tx_history,
            predicted_tx[['date', 'store_id', 'pred_transactions']].rename(columns={'pred_transactions': 'transactions'}),
        ],
        ignore_index=True,
    )

    future = future_calendar.merge(
        predicted_tx[['date', 'store_id', 'pred_transactions']], on=['date', 'store_id'], how='left'
    )
    future = future.sort_values(['date', 'store_id', 'family']).reset_index(drop=True)

    preds = []
    for _, row in future.iterrows():
        lag_feats = _sales_lag_features(sales_history, tx_history, row['store_id'], row['family'], row['date'])
        feature_row = row.to_dict() | lag_feats | {'transactions': row['pred_transactions']}
        x = pd.DataFrame([feature_row])
        pred = float(np.clip(sales_model.predict(x[SALES_NUMERIC_FEATURES + SALES_CATEGORICAL_FEATURES])[0], 0, None))

        hist_series = sales_history[(sales_history['store_id'] == row['store_id']) & (sales_history['family'] == row['family'])].sort_values('date')
        baseline_sales = float(hist_series.tail(28).groupby(hist_series.tail(28)['date'].dt.dayofweek).mean(numeric_only=True)['unit_sales'].get(row['date'].dayofweek, hist_series.tail(7)['unit_sales'].mean()))
        explanation = explain_row(pred, baseline_sales, row)

        record = row.to_dict()
        record['pred_transactions'] = row['pred_transactions']
        record['pred_sales'] = pred
        record['baseline_sales'] = baseline_sales
        record['explanation'] = explanation
        preds.append(record)

        sales_history = pd.concat(
            [
                sales_history,
                pd.DataFrame(
                    [
                        {
                            'date': row['date'],
                            'store_id': row['store_id'],
                            'family': row['family'],
                            'unit_sales': pred,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return pd.DataFrame(preds)


def build_manager_summary(forecast: pd.DataFrame, out_path: Path) -> None:
    forecast = forecast.copy()
    horizon = forecast['date'].dt.date.min(), forecast['date'].dt.date.max()
    lines = [
        '# ShelfSense MVP Forecast Summary',
        '',
        f'Forecast window: **{horizon[0]} to {horizon[1]}**',
        '',
    ]
    store_totals = (
        forecast.groupby('store_id')[['pred_transactions', 'pred_sales']]
        .sum()
        .sort_values('pred_transactions', ascending=False)
    )
    peak_row = forecast.sort_values('pred_sales', ascending=False).iloc[0]
    lines.append('## Store-level outlook')
    for store_id, row in store_totals.iterrows():
        lines.append(
            f'- {store_id}: projected {row["pred_transactions"]:.0f} transactions and {row["pred_sales"]:.0f} family-level units across the 14-day horizon.'
        )
    lines.append('')
    lines.append('## Highest individual spike')
    lines.append(
        f'- Peak forecast: **{peak_row["store_id"]} / {peak_row["family"]} on {peak_row["date"].date()}** with {peak_row["pred_sales"]:.1f} units. {peak_row["explanation"]}'
    )

    top_spikes = forecast.assign(uplift=(forecast['pred_sales'] - forecast['baseline_sales']) / forecast['baseline_sales']).sort_values('uplift', ascending=False).head(8)
    lines.append('')
    lines.append('## Top uplift days')
    for _, row in top_spikes.iterrows():
        lines.append(
            f'- {row["date"].date()} | {row["store_id"]} | {row["family"]}: {row["pred_sales"]:.1f} units ({row["uplift"]*100:.1f}% vs baseline). {row["explanation"]}'
        )

    out_path.write_text('\n'.join(lines), encoding='utf-8')


def main(data_dir: Path | None = None, models_dir: Path | None = None) -> None:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    models_dir = Path(models_dir) if models_dir else MODELS_DIR

    history_sales = pd.read_csv(data_dir / 'historical_sales.csv', parse_dates=['date'])
    history_tx = pd.read_csv(data_dir / 'historical_transactions.csv', parse_dates=['date'])
    future_calendar = pd.read_csv(data_dir / 'future_calendar.csv', parse_dates=['date'])

    transactions_model = load_model(models_dir / 'transactions_model.joblib')
    sales_model = load_model(models_dir / 'sales_model.joblib')

    predicted_tx = predict_transactions(history_tx, future_calendar, transactions_model)
    forecast = predict_sales(history_sales, predicted_tx, future_calendar, sales_model)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    predicted_tx.to_csv(ARTIFACTS_DIR / 'future_transactions_forecast.csv', index=False)
    forecast.to_csv(ARTIFACTS_DIR / 'future_sales_forecast.csv', index=False)
    build_manager_summary(forecast, ARTIFACTS_DIR / 'manager_summary.md')
    print(forecast.head(10).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate recursive 14-day forecasts for ShelfSense MVP.')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--models-dir', type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    main(data_dir=args.data_dir, models_dir=args.models_dir)
