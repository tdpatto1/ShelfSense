from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from common import ARTIFACTS_DIR, MODELS_DIR


SALES_NUMERIC_FEATURES = [
    'transactions',
    'onpromotion',
    'holiday_flag',
    'local_event_flag',
    'weekend_flag',
    'payday_flag',
    'heatwave_flag',
    'temp_index',
    'day_of_week',
    'month',
    'week_of_year',
    'quarter',
    'day_of_month',
    'lag_sales_1',
    'lag_sales_7',
    'lag_sales_14',
    'rolling_sales_7',
    'rolling_sales_28',
    'lag_transactions_1',
    'lag_transactions_7',
    'rolling_transactions_7',
]

SALES_CATEGORICAL_FEATURES = ['store_id', 'family', 'holiday_name', 'local_event_name', 'city', 'day_name']

TRANSACTION_NUMERIC_FEATURES = [
    'holiday_flag',
    'local_event_flag',
    'weekend_flag',
    'payday_flag',
    'heatwave_flag',
    'temp_index',
    'day_of_week',
    'month',
    'week_of_year',
    'quarter',
    'day_of_month',
    'lag_transactions_1',
    'lag_transactions_7',
    'lag_transactions_14',
    'rolling_transactions_7',
    'rolling_transactions_28',
]

TRANSACTION_CATEGORICAL_FEATURES = ['store_id', 'city', 'holiday_name', 'local_event_name', 'day_name']

EVENT_COLUMNS = ['holiday_flag', 'local_event_flag', 'holiday_name', 'local_event_name', 'payday_flag', 'weekend_flag']


def add_transaction_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['store_id', 'date']).copy()
    grouped = df.groupby('store_id', sort=False)
    df['lag_transactions_1'] = grouped['transactions'].shift(1)
    df['lag_transactions_7'] = grouped['transactions'].shift(7)
    df['lag_transactions_14'] = grouped['transactions'].shift(14)
    df['rolling_transactions_7'] = grouped['transactions'].transform(lambda s: s.shift(1).rolling(7).mean())
    df['rolling_transactions_28'] = grouped['transactions'].transform(lambda s: s.shift(1).rolling(28).mean())
    return df


def add_sales_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['store_id', 'family', 'date']).copy()
    grouped_sales = df.groupby(['store_id', 'family'], sort=False)
    grouped_tx = df.groupby(['store_id'], sort=False)
    df['lag_sales_1'] = grouped_sales['unit_sales'].shift(1)
    df['lag_sales_7'] = grouped_sales['unit_sales'].shift(7)
    df['lag_sales_14'] = grouped_sales['unit_sales'].shift(14)
    df['rolling_sales_7'] = grouped_sales['unit_sales'].transform(lambda s: s.shift(1).rolling(7).mean())
    df['rolling_sales_28'] = grouped_sales['unit_sales'].transform(lambda s: s.shift(1).rolling(28).mean())
    df['lag_transactions_1'] = grouped_tx['transactions'].shift(1)
    df['lag_transactions_7'] = grouped_tx['transactions'].shift(7)
    df['rolling_transactions_7'] = grouped_tx['transactions'].transform(lambda s: s.shift(1).rolling(7).mean())
    return df


def time_split(df: pd.DataFrame, test_days: int = 56) -> Tuple[pd.DataFrame, pd.DataFrame]:
    max_date = pd.to_datetime(df['date']).max()
    cutoff = max_date - pd.Timedelta(days=test_days - 1)
    train_df = df[pd.to_datetime(df['date']) < cutoff].copy()
    test_df = df[pd.to_datetime(df['date']) >= cutoff].copy()
    return train_df, test_df


def make_model(random_state: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=280,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective='reg:squarederror',
        reg_lambda=1.0,
        min_child_weight=2,
        n_jobs=4,
        random_state=random_state,
    )


def build_pipeline(numeric_features: Sequence[str], categorical_features: Sequence[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), list(numeric_features)),
            (
                'cat',
                Pipeline(
                    [
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore')),
                    ]
                ),
                list(categorical_features),
            ),
        ]
    )

    model = make_model()
    return Pipeline([('preprocess', preprocessor), ('model', model)])


def fit_pipeline(
    train_df: pd.DataFrame,
    target_col: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Pipeline:
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(train_df[list(numeric_features) + list(categorical_features)], train_df[target_col])
    return pipeline


def seasonal_naive(df: pd.DataFrame, target_col: str, lag_days: int = 7) -> pd.Series:
    if target_col == 'transactions':
        grp = df.groupby('store_id', sort=False)[target_col]
        return grp.shift(lag_days)
    grp = df.groupby(['store_id', 'family'], sort=False)[target_col]
    return grp.shift(lag_days)


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def get_feature_importance_table(pipeline: Pipeline, feature_names: Sequence[str]) -> pd.DataFrame:
    model = pipeline.named_steps['model']
    preprocess = pipeline.named_steps['preprocess']
    transformed_names = preprocess.get_feature_names_out(feature_names)
    importances = model.feature_importances_
    df = pd.DataFrame({'feature': transformed_names, 'importance': importances})
    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def drop_event_features(df: pd.DataFrame, numeric_features: Sequence[str], categorical_features: Sequence[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    drop_set = {'holiday_flag', 'local_event_flag', 'payday_flag', 'weekend_flag', 'holiday_name', 'local_event_name'}
    keep_numeric = [col for col in numeric_features if col not in drop_set]
    keep_categorical = [col for col in categorical_features if col not in drop_set]
    cols = keep_numeric + keep_categorical
    return df.copy(), keep_numeric, keep_categorical
