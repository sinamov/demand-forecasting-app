# /api/feature_engineering.py

import pandas as pd
import numpy as np
from datetime import timedelta
import holidays
from sklearn.linear_model import LinearRegression

def fix_leap_year_shift(df):
    df['week'] = pd.to_datetime(df['week']) # Safeguard
    """Adjusts dates for leap year inconsistencies found in the data."""
    problematic_start_date = pd.to_datetime('2012-03-06')
    rows_to_fix_mask = (df['week'] >= problematic_start_date)
    df.loc[rows_to_fix_mask, 'week'] = df.loc[rows_to_fix_mask, 'week'] - pd.Timedelta(days=1)
    return df

def add_price_features(df):
    df['week'] = pd.to_datetime(df['week']) # Safeguard
    """Calculates price difference features."""
    # Fill missing total_price with base_price where applicable
    fill_value = df[df['total_price'].isnull()]['base_price']
    df['total_price'] = df['total_price'].fillna(fill_value)

    df['diff'] = df['base_price'] - df['total_price']
    df['relative_diff_total'] = df['diff'] / df['total_price']
    df['relative_diff_base'] = df['diff'] / df['base_price']
    return df

def extract_time_features(df, history_df_start_date):
    df['week'] = pd.to_datetime(df['week']) # Safeguard
    """Extracts features from the week's date."""
    df['year'] = df['week'].dt.year
    df['end_year'] = df['weekend_date'].dt.year
    df['quarter'] = df['week'].dt.quarter
    df['month'] = df['week'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['end_month'] = df['weekend_date'].dt.month
    df['end_month_sin'] = np.sin(2 * np.pi * df['end_month']/12)
    df['end_month_cos'] = np.cos(2 * np.pi * df['end_month']/12)
    df['is_month_start'] = df['week'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['week'].dt.is_month_end.astype(int)
    df['weeknum'] = df['week'].dt.isocalendar().week.astype(int)
    df['weeknum_sin'] = np.sin(2 * np.pi * df['weeknum']/52)
    df['weeknum_cos'] = np.cos(2 * np.pi * df['weeknum']/52)
    df['week_from_start'] = (df['week'] - history_df_start_date).dt.days // 7
    df['day'] = df['week'].dt.day
    df['weekday'] = df['week'].dt.dayofweek
    return df

def get_holiday_feature(df):
    df['week'] = pd.to_datetime(df['week']) # Safeguard
    """Creates the is_holiday feature."""
    years_in_data = df['week'].dt.year.unique()
    us_holidays = holidays.US(years=years_in_data)
    holiday_dates = set(us_holidays.keys())
    
    holiday_week_starts = set()
    for h_date in holiday_dates:
        for i in range(7):
            possible_start_date = h_date - timedelta(days=i)
            holiday_week_starts.add(pd.to_datetime(possible_start_date))
            
    # Now the comparison is between two consistent datetime types
    df['is_holiday'] = df['week'].isin(holiday_week_starts).astype(int)
    return df

def create_lag_features(df, lags, history):
    df['week'] = pd.to_datetime(df['week']) # Safeguard
    """Creates lag features based on historical sales."""
    if not lags:
        return df
    
    df_copy = df.copy()
    source_data = pd.concat([history, df_copy], ignore_index=True)
    sku_weekly_sales = source_data.groupby(['sku_id', 'week'])['units_sold'].sum().reset_index()
    sku_weekly_sales = sku_weekly_sales.sort_values(by=['sku_id', 'week'])

    for lag in lags:
        feature_name = f'lag_{lag}_weeks'
        sku_weekly_sales[feature_name] = sku_weekly_sales.groupby('sku_id')['units_sold'].shift(lag)
    
    new_feature_cols = [f'lag_{lag}_weeks' for lag in lags]
    cols_to_merge = ['sku_id', 'week'] + new_feature_cols
    df_with_lags = pd.merge(df_copy, sku_weekly_sales[cols_to_merge], on=['sku_id', 'week'], how='left')
    df_with_lags[new_feature_cols] = df_with_lags[new_feature_cols].fillna(0)
    return df_with_lags

def create_moving_average_features(df, window_sizes, history):
    df['week'] = pd.to_datetime(df['week']) # Safeguard
    """Creates moving average features based on historical sales."""
    if not window_sizes:
        return df

    df_copy = df.copy()
    source_data = pd.concat([history, df_copy], ignore_index=True)
    sku_weekly_sales = source_data.groupby(['sku_id', 'week'])['units_sold'].sum().reset_index()
    sku_weekly_sales = sku_weekly_sales.sort_values(by=['sku_id', 'week'])
    
    for window in window_sizes:
        feature_name = f'MA_{window}_weeks'
        sku_weekly_sales[feature_name] = sku_weekly_sales.groupby('sku_id')['units_sold'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
    
    new_feature_cols = [f'MA_{window}_weeks' for window in window_sizes]
    cols_to_merge = ['sku_id', 'week'] + new_feature_cols
    df_with_features = pd.merge(df_copy, sku_weekly_sales[cols_to_merge], on=['sku_id', 'week'], how='left')
    df_with_features[new_feature_cols] = df_with_features[new_feature_cols].fillna(0)
    return df_with_features

def create_moving_average_features_store(df, windows, history):
    df['week'] = pd.to_datetime(df['week']) # Safeguard
    """Creates store-level moving average features."""
    if not windows:
        return df

    df_copy = df.copy()
    source_data = pd.concat([history, df_copy], ignore_index=True)
    store_weekly_sales = source_data.groupby(['store_id', 'week'])['units_sold'].sum().reset_index()
    store_weekly_sales = store_weekly_sales.sort_values(by=['store_id', 'week'])

    for window in windows:
        feature_name = f'store_moving_avg_{window}_weeks'
        store_weekly_sales[feature_name] = store_weekly_sales.groupby('store_id')['units_sold'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
    
    new_feature_cols = [f'store_moving_avg_{w}_weeks' for w in windows]
    cols_to_merge = ['store_id', 'week'] + new_feature_cols
    df_with_features = pd.merge(df_copy, store_weekly_sales[cols_to_merge], on=['store_id', 'week'], how='left')
    df_with_features[new_feature_cols] = df_with_features[new_feature_cols].fillna(0)
    return df_with_features

def prepare_future_dataframe(future_df, history_df):
    """
    Takes the raw user-uploaded dataframe and applies all non-recursive
    feature engineering steps in the correct order.
    """
    future_df['weekend_date'] = future_df['week'] + pd.to_timedelta(6, unit='D')
    future_df = add_price_features(future_df)
    history_start_date = history_df['week'].min()
    future_df = extract_time_features(future_df, history_start_date)
    future_df = get_holiday_feature(future_df)
    return future_df
