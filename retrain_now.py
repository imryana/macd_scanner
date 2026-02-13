"""
Direct retraining script — uses existing training_data.csv (97K signals from 502 S&P 500 stocks)
Skips data collection, goes straight to XGBoost + LSTM training
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import pandas as pd
from train_models import ModelTrainingPipeline

pipeline = ModelTrainingPipeline(target_periods=[5, 10, 20])

# Load existing data
print("Loading existing training data (97K signals, 502 S&P 500 stocks)...")
df = pd.read_csv('training_data.csv')
print(f"Loaded {len(df)} samples")
pipeline.data_collected = True

# Step 2: Train XGBoost with hyperparameter optimization
print("\n" + "="*60)
print("STEP 2: TRAIN XGBOOST (with optimization)")
print("="*60)
xgb_model, xgb_metrics = pipeline.step2_train_xgboost(df, target_period=5, optimize=True)

# Step 3: Train LSTM (100 epochs, GPU)
print("\n" + "="*60)
print("STEP 3: TRAIN LSTM (100 epochs)")
print("="*60)
lstm_trainer, lstm_metrics = pipeline.step3_train_lstm(df, target_period=5, epochs=100)

# Step 4: Create ensemble
print("\n" + "="*60)
print("STEP 4: CREATE ENSEMBLE")
print("="*60)
ensemble = pipeline.step4_create_ensemble(target_period=5)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"XGBoost metrics: {xgb_metrics}")
print(f"LSTM metrics: {lstm_metrics}")
