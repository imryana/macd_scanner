"""
Exit Timing Model - Predicts optimal holding duration after MACD crossover
Uses XGBoost regression to recommend dynamic exit day (1-20) per signal
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ExitTimingModel:
    """
    XGBoost Regressor for predicting optimal exit day (1-20)
    after a MACD crossover entry.
    
    Learns from historical data: given the market conditions at entry,
    which day (1-20) produced the maximum favorable return?
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
    
    def prepare_features(self, df):
        """
        Prepare features for training (same as XGBoost classifier)
        """
        exclude_cols = ['ticker', 'date', 'entry_price', 'crossover_type',
                       'optimal_exit_day', 'optimal_exit_return'] + \
                      [col for col in df.columns if 'return' in col or 'profitable' in col or 
                       'max_return' in col or 'max_drawdown' in col or 'daily_return' in col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        X = X.fillna(0)
        
        # Add interaction features (same as ml_model.py)
        X['macd_rsi_interaction'] = X['macd_value'] * X['rsi']
        X['adx_di_interaction'] = X['adx'] * X['di_diff']
        if 'volume_ratio_20d' in X.columns and 'returns_1d' in X.columns:
            X['volume_momentum'] = X['volume_ratio_20d'] * X['returns_1d']
        
        self.feature_names = X.columns.tolist()
        return X
    
    def train(self, df, test_size=0.2):
        """
        Train the exit timing model
        
        Args:
            df: DataFrame with features and optimal_exit_day labels
            test_size: Proportion of data for testing
        """
        print("=" * 60)
        print("Training Exit Timing Model")
        print("=" * 60)
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = df['optimal_exit_day']
        
        # Remove samples with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"\n📊 Dataset Information:")
        print(f"   Total samples: {len(X)}")
        print(f"   Avg optimal exit day: {y.mean():.1f}")
        print(f"   Median optimal exit day: {y.median():.0f}")
        print(f"   Std dev: {y.std():.1f}")
        print(f"   Number of features: {len(self.feature_names)}")
        
        # Distribution of optimal exit days
        print(f"\n📈 Optimal Exit Day Distribution:")
        for bucket_name, bucket_range in [("Day 1-3 (Quick)", (1, 3)), 
                                           ("Day 4-7 (Short)", (4, 7)),
                                           ("Day 8-14 (Medium)", (8, 14)),
                                           ("Day 15-20 (Long)", (15, 20))]:
            count = ((y >= bucket_range[0]) & (y <= bucket_range[1])).sum()
            pct = count / len(y) * 100
            print(f"   {bucket_name}: {count} ({pct:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost Regressor
        print("\n🚀 Training XGBoost Regressor...")
        
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            min_child_weight=10,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=1.0,
            reg_alpha=0.1,
            reg_lambda=5.0,
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
        
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        self.model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Clip predictions to valid range
        test_pred_clipped = np.clip(test_pred, 1, 20)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred_clipped)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred_clipped))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred_clipped)
        
        print(f"\n📈 Model Performance:")
        print(f"   Train MAE:  {train_mae:.2f} days")
        print(f"   Test MAE:   {test_mae:.2f} days")
        print(f"   Train RMSE: {train_rmse:.2f} days")
        print(f"   Test RMSE:  {test_rmse:.2f} days")
        print(f"   Train R²:   {train_r2:.3f}")
        print(f"   Test R²:    {test_r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5,
                                    scoring='neg_mean_absolute_error')
        print(f"   Cross-Val MAE: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
        
        # Bucket accuracy (does it predict the right range?)
        buckets_actual = pd.cut(y_test.reset_index(drop=True), bins=[0, 3, 7, 14, 20], labels=['Quick', 'Short', 'Medium', 'Long'])
        buckets_pred = pd.cut(pd.Series(test_pred_clipped), bins=[0, 3, 7, 14, 20], labels=['Quick', 'Short', 'Medium', 'Long'])
        bucket_accuracy = (buckets_actual == buckets_pred).mean()
        print(f"   Bucket Accuracy: {bucket_accuracy:.1%} (Quick/Short/Medium/Long)")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Important Features (Exit Timing):")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        # Store test data for diagnostics
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.test_pred = test_pred_clipped
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'bucket_accuracy': bucket_accuracy
        }
    
    def predict(self, features_dict):
        """
        Predict optimal exit day for a single signal
        
        Args:
            features_dict: Dictionary with feature values
            
        Returns:
            optimal_day (int 1-20), confidence_range (tuple of low, high)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        df = pd.DataFrame([features_dict])
        X = df.copy()
        
        # Add interaction features
        X['macd_rsi_interaction'] = X.get('macd_value', 0) * X.get('rsi', 50)
        X['adx_di_interaction'] = X.get('adx', 20) * X.get('di_diff', 0)
        X['volume_momentum'] = X.get('volume_ratio_20d', 1) * X.get('returns_1d', 0)
        
        # Ensure all features present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Predict
        raw_pred = self.model.predict(X_scaled)[0]
        optimal_day = int(np.clip(round(raw_pred), 1, 20))
        
        # Estimate confidence range (±2 days based on typical MAE)
        low = max(1, optimal_day - 2)
        high = min(20, optimal_day + 2)
        
        return optimal_day, (low, high)
    
    def get_exit_recommendation(self, features_dict):
        """
        Get a human-readable exit recommendation
        
        Returns:
            dict with exit_day, exit_range, exit_label, description
        """
        optimal_day, (low, high) = self.predict(features_dict)
        
        # Categorize
        if optimal_day <= 3:
            label = "Quick Exit"
            description = f"Take profits early (day {low}-{high}). Signal tends to peak fast."
        elif optimal_day <= 7:
            label = "Short Hold"
            description = f"Hold ~{optimal_day} days (day {low}-{high}). Moderate hold period."
        elif optimal_day <= 14:
            label = "Medium Hold"
            description = f"Hold ~{optimal_day} days (day {low}-{high}). Let the trend develop."
        else:
            label = "Extended Hold"
            description = f"Hold ~{optimal_day} days (day {low}-{high}). Longer-term move expected."
        
        return {
            'exit_day': optimal_day,
            'exit_range_low': low,
            'exit_range_high': high,
            'exit_label': label,
            'exit_description': description
        }
    
    def plot_diagnostics(self, save_path='exit_model_diagnostics.png'):
        """Generate diagnostic plots for exit timing model"""
        if self.model is None or self.X_test is None:
            print("⚠️  No test data. Train model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Predicted vs Actual scatter
        axes[0, 0].scatter(self.y_test, self.test_pred, alpha=0.1, s=5)
        axes[0, 0].plot([1, 20], [1, 20], 'r--', linewidth=2, label='Perfect')
        axes[0, 0].set_xlabel('Actual Optimal Exit Day')
        axes[0, 0].set_ylabel('Predicted Exit Day')
        axes[0, 0].set_title('Predicted vs Actual Exit Day')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residual distribution
        residuals = self.test_pred - self.y_test
        axes[0, 1].hist(residuals, bins=40, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Prediction Error (days)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Residual Distribution (MAE={mean_absolute_error(self.y_test, self.test_pred):.2f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature importance
        top_features = self.feature_importance.head(15)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'])
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'])
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 15 Feature Importance (Exit Timing)')
        axes[1, 0].invert_yaxis()
        
        # 4. Actual exit day distribution
        axes[1, 1].hist(self.y_test, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 1].set_xlabel('Optimal Exit Day')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Optimal Exit Days')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Exit model diagnostics saved to: {save_path}")
        plt.close()
    
    def save(self, filepath='exit_timing_model.pkl'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"💾 Exit timing model saved to: {filepath}")
    
    def load(self, filepath='exit_timing_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        print(f"✅ Exit timing model loaded from: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("Exit Timing Model Training")
    print("=" * 60)
    
    try:
        df = pd.read_csv('training_data.csv')
        print(f"\n✅ Loaded {len(df)} training samples")
    except FileNotFoundError:
        print("\n❌ training_data.csv not found!")
        print("Run: python data_preparation.py first")
        exit(1)
    
    if 'optimal_exit_day' not in df.columns:
        print("\n❌ optimal_exit_day column not found!")
        print("Re-run data collection to include daily return tracking.")
        exit(1)
    
    model = ExitTimingModel()
    metrics = model.train(df)
    model.plot_diagnostics('exit_model_diagnostics.png')
    model.save('exit_timing_model.pkl')
    
    print("\n" + "=" * 60)
    print("✅ Exit timing model training complete!")
    print("=" * 60)
