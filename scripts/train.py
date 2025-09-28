# /scripts/train.py

#GCP
import joblib
from google.cloud import storage
import json

#The Yoozh:
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

import joblib
from category_encoders import MEstimateEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
import sys
import os

#Logistics
import warnings
from pathlib import Path
import random
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)


# Add the 'api' directory to the Python path to import our feature engineering module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'api')))
from feature_engineering import generate_all_features

# --- CONFIGURATION ---
DATA_DIR = '../data/'
ARTIFACTS_DIR = '../artifacts/'
TRAIN_FILE = 'train_0irEZ2H.csv'
TEST_FILE = 'test_nfaJ3J5.csv'

# Ensure the artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- 1. LOAD AND PROCESS DATA ---
print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))

print("Generating features for the entire dataset...")
# Pass both dataframes to ensure consistent feature creation (like encodings, trends)
train_processed, test_processed = generate_all_features(train_df.copy(), test_df.copy())

# Save the processed training data to be used as history in the API
train_processed.to_csv(os.path.join(ARTIFACTS_DIR, 'train_history.csv'), index=False)
print("Saved processed training data as history.")

# --- 2. TRAIN ENCODER AND MODELS ---
print("Training MEstimateEncoder on full training data...")
store_encoder = MEstimateEncoder(cols=['store_id'])
store_encoder.fit(train_processed[['store_id']], train_processed['units_sold'])
train_processed['store_encoded'] = store_encoder.transform(train_processed[['store_id']])

# This would contain your best hyperparameters from Optuna
# For this example, we'll use a placeholder.
best_params = {"216233": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 7,
        "min_samples_leaf": 1,
        "max_depth": None
    },
    "216418": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 8,
        "min_samples_leaf": 2,
        "max_depth": None
    },
    "216419": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 5,
        "min_samples_leaf": 4,
        "max_depth": None
    },
    "216425": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 10,
        "min_samples_leaf": 1,
        "max_depth": 10
    },
    "217217": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 8,
        "min_samples_leaf": 1,
        "max_depth": None
    },
    "217390": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 9,
        "min_samples_leaf": 1,
        "max_depth": 10
    },
    "217777": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 6,
        "min_samples_leaf": 1,
        "max_depth": None
    },
    "219009": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 8,
        "min_samples_leaf": 1,
        "max_depth": None
    },
    "219029": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 7,
        "min_samples_leaf": 1,
        "max_depth": None
    },
    "219844": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 9,
        "min_samples_leaf": 1,
        "max_depth": None
    },
    "222087": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 10,
        "min_samples_leaf": 1,
        "max_depth": 10
    },
    "222765": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 9,
        "min_samples_leaf": 1,
        "max_depth": 10
    },
    "223153": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 10,
        "min_samples_leaf": 1,
        "max_depth": None
    },
    "223245": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 10,
        "min_samples_leaf": 2,
        "max_depth": 10
    },
    "245338": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 8,
        "min_samples_leaf": 1,
        "max_depth": 10
    },
    "245387": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 8,
        "min_samples_leaf": 1,
        "max_depth": None
    },
    "300021": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 10,
        "min_samples_leaf": 2,
        "max_depth": None
    },
    "300291": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 5,
        "min_samples_leaf": 4,
        "max_depth": None
    },
    "320485": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 8,
        "min_samples_leaf": 1,
        "max_depth": 10
    },
    "327492": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 6,
        "min_samples_leaf": 1,
        "max_depth": 10
    },
    "378934": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 8,
        "min_samples_leaf": 2,
        "max_depth": 10
    },
    "398721": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 9,
        "min_samples_leaf": 1,
        "max_depth": 10
    },
    "545621": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 9,
        "min_samples_leaf": 4,
        "max_depth": None
    },
    "546789": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 6,
        "min_samples_leaf": 2,
        "max_depth": 10
    },
    "547934": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_split": 9,
        "min_samples_leaf": 3,
        "max_depth": 10
    },
    "600934": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_depth": None
    },
    "673209": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 9,
        "min_samples_leaf": 4,
        "max_depth": 10
    },
    "679023": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 7,
        "min_samples_leaf": 1,
        "max_depth": 10
    }}





sku_specific_lags = {
    216233: ['lag_1_weeks', 'lag_11_weeks', 'lag_25_weeks', 'lag_36_weeks', 'lag_44_weeks', 'lag_51_weeks', 'lag_52_weeks', 
             'lag_53_weeks', 'lag_55_weeks'],
    216418: ['lag_1_weeks', 'lag_2_weeks', 'lag_5_weeks', 'lag_19_weeks', 'lag_34_weeks', 'lag_37_weeks', 'lag_51_weeks',
            'lag_54_weeks'], 
    216419: ['lag_1_weeks', 'lag_2_weeks','lag_37_weeks', 'lag_40_weeks', 'lag_43_weeks', 'lag_49_weeks', 'lag_55_weeks'],
    216425: ['lag_1_weeks', 'lag_8_weeks', 'lag_35_weeks', 'lag_43_weeks', 'lag_49_weeks'],  
    217217: ['lag_1_weeks','lag_10_weeks', 'lag_25_weeks','lag_36_weeks','lag_39_weeks','lag_41_weeks','lag_54_weeks'],
    217390: ['lag_1_weeks', 'lag_33_weeks', 'lag_42_weeks', 'lag_43_weeks', 'lag_44_weeks', 'lag_47_weeks'],
    217777: ['lag_1_weeks','lag_3_weeks','lag_4_weeks', 'lag_10_weeks','lag_26_weeks','lag_50_weeks', 'lag_54_weeks'],
    219009: ['lag_1_weeks', 'lag_3_weeks', 'lag_34_weeks'],
    219029: ['lag_1_weeks', 'lag_2_weeks', 'lag_5_weeks', 'lag_25_weeks','lag_33_weeks', 'lag_38_weeks', 'lag_39_weeks', 
             'lag_48_weeks','lag_49_weeks', 'lag_50_weeks', 'lag_52_weeks'], 
    219844: ['lag_1_weeks', 'lag_10_weeks', 'lag_36_weeks', 'lag_54_weeks'],
    222087: ['lag_11_weeks', 'lag_24_weeks', 'lag_33_weeks', 'lag_54_weeks', 'lag_55_weeks'],
    222765: ['lag_1_weeks', 'lag_40_weeks', 'lag_43_weeks', 'lag_53_weeks'], 
    223153: ['lag_1_weeks', 'lag_17_weeks', 'lag_34_weeks', 'lag_50_weeks', 'lag_52_weeks', 'lag_54_weeks'],
    223245: ['lag_1_weeks', 'lag_13_weeks', 'lag_24_weeks', 'lag_54_weeks'],
    245338: ['lag_1_weeks', 'lag_19_weeks', 'lag_24_weeks', 'lag_37_weeks', 'lag_43_weeks', 'lag_48_weeks',
             'lag_51_weeks', 'lag_52_weeks', 'lag_53_weeks'],
    245387: ['lag_1_weeks', 'lag_19_weeks', 'lag_24_weeks', 'lag_37_weeks', 'lag_43_weeks', 'lag_44_weeks', 'lag_51_weeks',
             'lag_52_weeks', 'lag_53_weeks'],
    300021: ['lag_1_weeks', 'lag_2_weeks', 'lag_42_weeks', 'lag_54_weeks'],
    300291: ['lag_1_weeks','lag_4_weeks', 'lag_19_weeks','lag_36_weeks', 'lag_50_weeks','lag_53_weeks', 'lag_54_weeks'], 
    320485: ['lag_1_weeks', 'lag_3_weeks', 'lag_20_weeks', 'lag_23_weeks', 'lag_42_weeks', 'lag_47_weeks', 'lag_53_weeks'],
    327492: ['lag_1_weeks','lag_12_weeks', 'lag_28_weeks','lag_34_weeks', 'lag_52_weeks', 'lag_53_weeks'],
    378934: ['lag_1_weeks', 'lag_2_weeks', 'lag_9_weeks', 'lag_22_weeks',  'lag_23_weeks', 'lag_35_weeks', 'lag_42_weeks',
             'lag_53_weeks'],
    398721: ['lag_1_weeks', 'lag_19_weeks', 'lag_24_weeks', 'lag_28_weeks', 'lag_37_weeks', 'lag_43_weeks', 
            'lag_48_weeks', 'lag_49_weeks', 'lag_51_weeks', 'lag_53_weeks'],
    545621: ['lag_1_weeks', 'lag_35_weeks', 'lag_47_weeks', 'lag_51_weeks', 'lag_54_weeks'],
    546789: ['lag_1_weeks','lag_2_weeks', 'lag_23_weeks','lag_25_weeks', 'lag_52_weeks','lag_54_weeks'],
    547934: ['lag_1_weeks'],
    600934: ['lag_1_weeks','lag_2_weeks', 'lag_41_weeks','lag_42_weeks', 'lag_44_weeks','lag_53_weeks'], 
    673209: ['lag_1_weeks', 'lag_28_weeks','lag_39_weeks', 'lag_53_weeks'], 
    679023: ['lag_1_weeks','lag_2_weeks', 'lag_40_weeks','lag_43_weeks']
}

sku_specific_moving_averages = {
    216233: ['MA_1_weeks', 'MA_2_weeks'],
    216418: ['MA_1_weeks', 'MA_2_weeks', 'MA_3_weeks', 'MA_4_weeks', 'MA_5_weeks'], 
    216419: ['MA_1_weeks', 'MA_2_weeks','MA_3_weeks', 'MA_4_weeks', 'MA_5_weeks'],
    216425: ['MA_1_weeks', 'MA_2_weeks'], 
    217217: ['MA_1_weeks', 'MA_2_weeks', 'MA_3_weeks'],
    217390: ['MA_1_weeks'],
    217777: ['MA_1_weeks', 'MA_2_weeks', 'MA_3_weeks'],
    219009: ['MA_1_weeks', 'MA_2_weeks'],
    219029: ['MA_1_weeks'], 
    219844: ['MA_1_weeks', 'MA_2_weeks', 'MA_3_weeks'],
    222087: [],
    222765: ['MA_1_weeks'],
    223153: ['MA_1_weeks'],
    223245: ['MA_1_weeks'],
    245338: ['MA_1_weeks'],
    245387: ['MA_1_weeks'],
    300021: ['MA_1_weeks', 'MA_2_weeks', 'MA_3_weeks', 'MA_4_weeks', 'MA_5_weeks', 'MA_6_weeks'], 
    300291: ['MA_1_weeks'],
    320485: ['MA_1_weeks', 'MA_2_weeks'],
    327492: ['MA_1_weeks', 'MA_2_weeks'],
    378934: ['MA_1_weeks', 'MA_2_weeks'],
    398721: ['MA_1_weeks'],
    545621: ['MA_1_weeks'],
    546789: ['MA_1_weeks'],
    547934: ['MA_1_weeks'],
    600934: ['MA_1_weeks'], 
    673209: ['MA_1_weeks', 'MA_2_weeks'], 
    679023: ['MA_1_weeks']
}



cols = [
    'base_price', 'total_price', 'diff', 'relative_diff_base', 'relative_diff_total',
    'is_featured_sku', 'is_display_sku',
    ]

cols += ['store_encoded']
    # here RMSLE was 4750
cols += ['year','end_year', 'quarter', 'month_sin', 'month_cos', 'end_month_sin', 'end_month_cos',
             'is_month_start', 'is_month_end', 'weeknum_sin', 'weeknum_cos', 'week_from_start','day','is_holiday'] 
    # here 4573

final_feature_sets = {}
for sku_id, custom_lags in sku_specific_lags.items():
    final_feature_sets[sku_id] = cols + custom_lags

for sku_id, custom_MAs in sku_specific_moving_averages.items():
    final_feature_sets[sku_id] = cols + custom_MAs

for sku_id in sku_specific_moving_averages.keys():
    final_feature_sets[sku_id] = cols + ['store_trend','store_moving_avg_26_weeks']
    
    
    
print("Training a model for each SKU and bundling artifacts...")
for sku_id, feature_list in final_feature_sets.items():
    print(f"  Processing SKU: {sku_id}")
    
    # Isolate data for the current SKU
    train_sku = train_processed[train_processed['sku_id'] == sku_id].copy().sort_values('week')
    
    # Get the feature list for this SKU (you would load this from your final_feature_sets)
    # For this example, we'll create a placeholder list.
    
    X_sku = train_sku[feature_list]
    y_sku = np.log1p(train_sku['units_sold'])
    
    
    # Train the final model
    bp = best_params.get(sku_id, {})
    final_model = RandomForestRegressor(random_state=42, n_jobs=-1, **bp)
    final_model.fit(X_sku, y_sku)
    
    # --- 3. BUNDLE ARTIFACTS ---
    artifact_bundle = {
        'model': final_model,
        'encoder': store_encoder,
        'feature_list': feature_list
    }
    
    # Save the entire bundle to a single file
    bundle_path = os.path.join(ARTIFACTS_DIR, f'model_bundle_sku_{sku_id}.joblib')
    joblib.dump(artifact_bundle, bundle_path)
    print(f"    -> Saved artifact bundle to {bundle_path}")

print("\nTraining and artifact bundling complete!")