# /api/prediction.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from .feature_engineering import (
    prepare_future_dataframe,
    create_lag_features,
    create_moving_average_features,
    create_moving_average_features_store,
)

def generate_forecast(
    artifacts: dict,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame
) -> pd.DataFrame:
    
    future_df = prepare_future_dataframe(future_df, history_df)
    
    session_history_df = history_df.copy()
    all_weekly_predictions = []
    
    prediction_weeks = sorted(future_df['week'].unique())

    for week in prediction_weeks:
        current_week_batch_df = future_df[future_df['week'] == week].copy()
        
        all_lags_needed = set()
        all_mas_needed = set()
        for sku_id in current_week_batch_df['sku_id'].unique():
            if sku_id in artifacts:
                feature_list = artifacts[sku_id]['feature_list']
                all_lags_needed.update([int(f.split('_')[1]) for f in feature_list if f.startswith('lag_')])
                all_mas_needed.update([int(f.split('_')[1]) for f in feature_list if f.startswith('MA_')])

        current_week_batch_df = create_lag_features(current_week_batch_df, list(all_lags_needed), history=session_history_df)
        current_week_batch_df = create_moving_average_features(current_week_batch_df, list(all_mas_needed), history=session_history_df)
        current_week_batch_df = create_moving_average_features_store(current_week_batch_df, [26], history=session_history_df)
        
        predictions_for_this_week = []
        for sku_id, group in current_week_batch_df.groupby('sku_id'):
            if sku_id in artifacts:
                model_bundle = artifacts[sku_id]
                model = model_bundle['model']
                encoder = model_bundle['encoder']
                feature_list = model_bundle['feature_list']

                group['store_encoded'] = encoder.transform(group[['store_id']])
                
                store_id = group['store_id'].iloc[0]
                store_history = session_history_df[session_history_df['store_id'] == store_id]
                if not store_history.empty and len(store_history) > 1:
                    trend_model = LinearRegression()
                    trend_model.fit(store_history[['week_from_start']], store_history['units_sold'])
                    group['store_trend'] = trend_model.predict(group[['week_from_start']])
                else:
                    group['store_trend'] = 0

                for col in feature_list:
                    if col not in group.columns:
                        group[col] = 0
                X_pred = group[feature_list]
                
                log_prediction = model.predict(X_pred)
                prediction = np.expm1(log_prediction)
                final_prediction = np.ceil(np.maximum(0, prediction)).astype(int)

                prediction_record = group[['week', 'sku_id', 'store_id', 'week_from_start']].copy()
                prediction_record['units_sold'] = final_prediction
                predictions_for_this_week.append(prediction_record)

        if predictions_for_this_week:
            weekly_results_df = pd.concat(predictions_for_this_week, ignore_index=True)
            all_weekly_predictions.append(weekly_results_df)
            session_history_df = pd.concat([session_history_df, weekly_results_df], ignore_index=True)

    if not all_weekly_predictions:
        return pd.DataFrame()
        
    return pd.concat(all_weekly_predictions, ignore_index=True)