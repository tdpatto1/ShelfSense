from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, compute_metrics, save_json
from modeling import (
    SALES_CATEGORICAL_FEATURES,
    SALES_NUMERIC_FEATURES,
    TRANSACTION_CATEGORICAL_FEATURES,
    TRANSACTION_NUMERIC_FEATURES,
    add_sales_lag_features,
    add_transaction_lag_features,
    drop_event_features,
    fit_pipeline,
    get_feature_importance_table,
    load_model,
    save_model,
    seasonal_naive,
    time_split,
)


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_transactions(transactions_path: Path) -> pd.DataFrame:
    tx = pd.read_csv(transactions_path, parse_dates=['date'])
    tx_feat = add_transaction_lag_features(tx)
    tx_feat = tx_feat.dropna().reset_index(drop=True)
    train_tx, test_tx = time_split(tx_feat)

    baseline_pred = seasonal_naive(tx_feat, 'transactions').loc[test_tx.index].clip(lower=0)
    event_flags = (test_tx['holiday_flag'] | test_tx['local_event_flag']).astype(int).to_numpy()
    metrics = [
        compute_metrics(test_tx['transactions'], baseline_pred, event_flags, 'seasonal_naive_7', 'transactions').to_dict()
    ]

    model_no_events = fit_pipeline(train_tx, 'transactions', *drop_event_features(train_tx, TRANSACTION_NUMERIC_FEATURES, TRANSACTION_CATEGORICAL_FEATURES)[1:])
    pred_no_events = pd.Series(
        np.clip(
            model_no_events.predict(test_tx[[
                *drop_event_features(test_tx, TRANSACTION_NUMERIC_FEATURES, TRANSACTION_CATEGORICAL_FEATURES)[1],
                *drop_event_features(test_tx, TRANSACTION_NUMERIC_FEATURES, TRANSACTION_CATEGORICAL_FEATURES)[2],
            ]]),
            0,
            None,
        ),
        index=test_tx.index,
    )
    metrics.append(
        compute_metrics(test_tx['transactions'], pred_no_events, event_flags, 'xgboost_no_events', 'transactions').to_dict()
    )

    model_full = fit_pipeline(train_tx, 'transactions', TRANSACTION_NUMERIC_FEATURES, TRANSACTION_CATEGORICAL_FEATURES)
    pred_full = pd.Series(
        np.clip(model_full.predict(test_tx[TRANSACTION_NUMERIC_FEATURES + TRANSACTION_CATEGORICAL_FEATURES]), 0, None),
        index=test_tx.index,
    )
    metrics.append(
        compute_metrics(test_tx['transactions'], pred_full, event_flags, 'xgboost_event_aware', 'transactions').to_dict()
    )

    save_model(model_full, MODELS_DIR / 'transactions_model.joblib')
    tx_out = test_tx.copy()
    tx_out['pred_baseline'] = baseline_pred.values
    tx_out['pred_no_events'] = pred_no_events.values
    tx_out['pred_event_aware'] = pred_full.values
    tx_out.to_csv(ARTIFACTS_DIR / 'transactions_test_predictions.csv', index=False)

    fi = get_feature_importance_table(model_full, TRANSACTION_NUMERIC_FEATURES + TRANSACTION_CATEGORICAL_FEATURES)
    fi.to_csv(ARTIFACTS_DIR / 'transactions_feature_importance.csv', index=False)

    sample_store = 'S002'
    sample = tx_out[tx_out['store_id'] == sample_store].sort_values('date').tail(28)
    plt.figure(figsize=(11, 5))
    plt.plot(sample['date'], sample['transactions'], label='Actual')
    plt.plot(sample['date'], sample['pred_baseline'], label='Baseline')
    plt.plot(sample['date'], sample['pred_event_aware'], label='Event-aware XGBoost')
    plt.xticks(rotation=45)
    plt.ylabel('Transactions')
    plt.title(f'Transaction forecast comparison ({sample_store})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / 'transactions_forecast_plot.png', dpi=180)
    plt.close()

    topfi = fi.head(12).sort_values('importance')
    plt.figure(figsize=(10, 6))
    plt.barh(topfi['feature'], topfi['importance'])
    plt.xlabel('Importance')
    plt.title('Top transaction-model features')
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / 'transactions_feature_importance.png', dpi=180)
    plt.close()

    return pd.DataFrame(metrics)


def evaluate_sales(sales_path: Path) -> pd.DataFrame:
    sales = pd.read_csv(sales_path, parse_dates=['date'])
    sales_feat = add_sales_lag_features(sales)
    sales_feat = sales_feat.dropna().reset_index(drop=True)
    train_sales, test_sales = time_split(sales_feat)

    baseline_pred = seasonal_naive(sales_feat, 'unit_sales').loc[test_sales.index].clip(lower=0)
    event_flags = (test_sales['holiday_flag'] | test_sales['local_event_flag']).astype(int).to_numpy()
    metrics = [
        compute_metrics(test_sales['unit_sales'], baseline_pred, event_flags, 'seasonal_naive_7', 'sales').to_dict()
    ]

    _, sales_num_no_events, sales_cat_no_events = drop_event_features(
        train_sales, SALES_NUMERIC_FEATURES, SALES_CATEGORICAL_FEATURES
    )
    model_no_events = fit_pipeline(train_sales, 'unit_sales', sales_num_no_events, sales_cat_no_events)
    pred_no_events = pd.Series(
        np.clip(model_no_events.predict(test_sales[sales_num_no_events + sales_cat_no_events]), 0, None),
        index=test_sales.index,
    )
    metrics.append(
        compute_metrics(test_sales['unit_sales'], pred_no_events, event_flags, 'xgboost_no_events', 'sales').to_dict()
    )

    model_full = fit_pipeline(train_sales, 'unit_sales', SALES_NUMERIC_FEATURES, SALES_CATEGORICAL_FEATURES)
    pred_full = pd.Series(
        np.clip(model_full.predict(test_sales[SALES_NUMERIC_FEATURES + SALES_CATEGORICAL_FEATURES]), 0, None),
        index=test_sales.index,
    )
    metrics.append(
        compute_metrics(test_sales['unit_sales'], pred_full, event_flags, 'xgboost_event_aware', 'sales').to_dict()
    )

    save_model(model_full, MODELS_DIR / 'sales_model.joblib')
    sales_out = test_sales.copy()
    sales_out['pred_baseline'] = baseline_pred.values
    sales_out['pred_no_events'] = pred_no_events.values
    sales_out['pred_event_aware'] = pred_full.values
    sales_out.to_csv(ARTIFACTS_DIR / 'sales_test_predictions.csv', index=False)

    fi = get_feature_importance_table(model_full, SALES_NUMERIC_FEATURES + SALES_CATEGORICAL_FEATURES)
    fi.to_csv(ARTIFACTS_DIR / 'sales_feature_importance.csv', index=False)

    sample = (
        sales_out[(sales_out['store_id'] == 'S002') & (sales_out['family'] == 'beverages')]
        .sort_values('date')
        .tail(28)
    )
    plt.figure(figsize=(11, 5))
    plt.plot(sample['date'], sample['unit_sales'], label='Actual')
    plt.plot(sample['date'], sample['pred_baseline'], label='Baseline')
    plt.plot(sample['date'], sample['pred_event_aware'], label='Event-aware XGBoost')
    plt.xticks(rotation=45)
    plt.ylabel('Unit sales')
    plt.title('Sales forecast comparison (S002 / beverages)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / 'sales_forecast_plot.png', dpi=180)
    plt.close()

    topfi = fi.head(15).sort_values('importance')
    plt.figure(figsize=(10, 6))
    plt.barh(topfi['feature'], topfi['importance'])
    plt.xlabel('Importance')
    plt.title('Top sales-model features')
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / 'sales_feature_importance.png', dpi=180)
    plt.close()

    event_mae = (
        sales_out.assign(
            abs_err_baseline=np.abs(sales_out['unit_sales'] - sales_out['pred_baseline']),
            abs_err_event=np.abs(sales_out['unit_sales'] - sales_out['pred_event_aware']),
        )
        .groupby('holiday_flag')
        [['abs_err_baseline', 'abs_err_event']]
        .mean()
        .rename(index={0: 'non_holiday', 1: 'holiday'})
    )
    event_mae.plot(kind='bar', figsize=(8, 5))
    plt.ylabel('Mean absolute error')
    plt.title('Holiday-day error: baseline vs event-aware model')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / 'holiday_error_comparison.png', dpi=180)
    plt.close()

    return pd.DataFrame(metrics)


def summarize_results(metrics_df: pd.DataFrame) -> None:
    metrics_df.to_csv(ARTIFACTS_DIR / 'metrics.csv', index=False)
    best_rows = (
        metrics_df.sort_values(['target', 'wmape'])
        .groupby('target', as_index=False)
        .first()
        .to_dict(orient='records')
    )
    save_json(ARTIFACTS_DIR / 'metrics_summary.json', {'best_models': best_rows})


def main(data_dir: Path | None = None) -> None:
    ensure_dirs()
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    tx_metrics = evaluate_transactions(data_dir / 'historical_transactions.csv')
    sales_metrics = evaluate_sales(data_dir / 'historical_sales.csv')
    metrics = pd.concat([tx_metrics, sales_metrics], ignore_index=True)
    summarize_results(metrics)
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate ShelfSense MVP forecasting models.')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    args = parser.parse_args()
    main(data_dir=args.data_dir)
