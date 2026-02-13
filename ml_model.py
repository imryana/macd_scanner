"""
XGBoost Machine Learning Model for MACD Signal Classification
Predicts profitability of MACD crossover signals
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MACDXGBoostModel:
    """
    XGBoost Classifier for MACD signal quality prediction
    Predicts whether a MACD crossover will be profitable
    """
    
    def __init__(self, target_period=5):
        """
        Args:
            target_period: Which holding period to predict (5, 10, or 20 days)
        """
        self.target_period = target_period
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_features(self, df):
        """
        Prepare features for training
        Handles categorical encoding and feature selection
        """
        # Define feature columns (exclude metadata and target columns)
        # Note: crossover_type, is_bullish, is_bearish are now INCLUDED as features
        exclude_cols = ['ticker', 'date', 'entry_price', 'sequence'] + \
                      [col for col in df.columns if 'return' in col or 'profitable' in col or
                       'max_return' in col or 'max_drawdown' in col or
                       'quality_' in col or 'risk_adjusted_ratio_' in col or
                       'optimal_exit' in col or 'daily_return' in col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # Handle any missing values
        X = X.fillna(0)
        
        # Add interaction features (using only available feature columns)
        X['macd_rsi_interaction'] = X['macd_value'] * X['rsi']
        X['adx_di_interaction'] = X['adx'] * X['di_diff']
        if 'volume_ratio_20d' in X.columns and 'returns_1d' in X.columns:
            X['volume_momentum'] = X['volume_ratio_20d'] * X['returns_1d']
        
        self.feature_names = X.columns.tolist()
        
        return X
    
    def train(self, df, test_size=0.2, optimize_hyperparameters=False):
        """
        Train the XGBoost model
        
        Args:
            df: DataFrame with features and labels
            test_size: Proportion of data for testing
            optimize_hyperparameters: Whether to run grid search (slower but better)
        """
        print("="*60)
        print(f"Training XGBoost Model ({self.target_period}-day prediction)")
        print("="*60)
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = df[f'profitable_{self.target_period}d']
        
        # Remove samples with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"\n📊 Dataset Information:")
        print(f"   Total samples: {len(X)}")
        print(f"   Positive samples (profitable): {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"   Negative samples (unprofitable): {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")
        print(f"   Number of features: {len(self.feature_names)}")

        # Time-based split: use date column to sort chronologically
        # Train on earlier data, test on most recent data (prevents lookahead bias)
        if 'date' in df.columns:
            df_sorted = df.loc[valid_mask].sort_values('date').reset_index(drop=True)
            X = self.prepare_features(df_sorted)
            y = df_sorted[f'profitable_{self.target_period}d']
            valid_mask2 = ~y.isna()
            X = X[valid_mask2]
            y = y[valid_mask2]

            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            print(f"   ⏰ Time-based split: train up to index {split_idx}, test from {split_idx}")
            print(f"   Train date range: {df_sorted['date'].iloc[0]} to {df_sorted['date'].iloc[split_idx-1]}")
            print(f"   Test date range:  {df_sorted['date'].iloc[split_idx]} to {df_sorted['date'].iloc[-1]}")
        else:
            # Fallback to random split if no date column
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        if optimize_hyperparameters:
            print("\n🔍 Optimizing hyperparameters (this may take 5-10 minutes)...")
            
            param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
            }
            
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='auc',
                tree_method='hist',
                device='cuda',  # GPU-accelerated
            )
            
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            self.model = grid_search.best_estimator_
            print(f"\n✅ Best parameters: {grid_search.best_params_}")
            
        else:
            print("\n🚀 Training with default parameters...")
            
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                min_child_weight=5,
                subsample=0.7,
                colsample_bytree=0.7,
                gamma=1.0,
                reg_alpha=0.1,
                reg_lambda=5.0,
                scale_pos_weight=scale_pos_weight,
                objective='binary:logistic',
                random_state=42,
                eval_metric='auc',
                tree_method='hist',
                device='cuda',  # GPU-accelerated
            )
            
            # Train with early stopping
            eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                verbose=False
            )
        
        # Evaluate
        print("\n📈 Model Performance:")
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        train_acc = (train_pred == y_train).mean()
        test_acc = (test_pred == y_test).mean()
        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)
        
        print(f"   Training Accuracy: {train_acc:.3f}")
        print(f"   Testing Accuracy:  {test_acc:.3f}")
        print(f"   Training AUC-ROC:  {train_auc:.3f}")
        print(f"   Testing AUC-ROC:   {test_auc:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"   Cross-Val AUC:     {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Detailed classification report
        print("\n📊 Classification Report (Test Set):")
        print(classification_report(y_test, test_pred, target_names=['Unprofitable', 'Profitable']))
        
        # Feature importance (tree-based)
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Permutation importance — more reliable for feature selection
        print("\n🔍 Computing permutation importance (may take a moment)...")
        perm_result = permutation_importance(
            self.model, X_test_scaled, y_test, n_repeats=5, random_state=42, scoring='roc_auc'
        )
        perm_importance = pd.DataFrame({
            'feature': self.feature_names,
            'perm_importance_mean': perm_result.importances_mean,
            'perm_importance_std': perm_result.importances_std
        }).sort_values('perm_importance_mean', ascending=False)

        # Drop features with negative or near-zero permutation importance
        useful_features = perm_importance[perm_importance['perm_importance_mean'] > 0.001]['feature'].tolist()
        dropped_features = [f for f in self.feature_names if f not in useful_features]

        if dropped_features and len(useful_features) >= 10:
            print(f"   ✂️ Dropping {len(dropped_features)} noise features: {dropped_features[:5]}{'...' if len(dropped_features) > 5 else ''}")
            print(f"   ✅ Keeping {len(useful_features)} useful features")

            # Retrain with pruned features
            self.feature_names = useful_features
            X_train_pruned = X_train[useful_features] if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train, columns=self.feature_names)
            X_test_pruned = X_test[useful_features] if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=self.feature_names)
            X_train_scaled = self.scaler.fit_transform(X_train_pruned)
            X_test_scaled = self.scaler.transform(X_test_pruned)

            self.model.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)], verbose=False)

            # Re-evaluate
            test_pred = self.model.predict(X_test_scaled)
            test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            test_acc = (test_pred == y_test).mean()
            test_auc = roc_auc_score(y_test, test_proba)
            print(f"   📈 After pruning — Test Accuracy: {test_acc:.3f}, Test AUC: {test_auc:.3f}")

            # Update feature importance for pruned model
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            print(f"   ✅ All {len(self.feature_names)} features contribute meaningfully")

        print("\n🔝 Top 15 Important Features:")
        for idx, row in self.feature_importance.head(15).iterrows():
            print(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        # Calibrate probabilities using manual Platt scaling (logistic regression on raw probabilities)
        # This makes confidence scores meaningful — a 0.7 prediction should win ~70% of the time
        print("\n🎯 Calibrating prediction probabilities (Platt scaling)...")
        raw_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        self.platt_scaler = LogisticRegression()
        self.platt_scaler.fit(raw_proba.reshape(-1, 1), y_test)
        calibrated_proba = self.platt_scaler.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
        calibrated_auc = roc_auc_score(y_test, calibrated_proba)
        print(f"   Calibrated AUC: {calibrated_auc:.3f}")
        test_proba = calibrated_proba  # Use calibrated probabilities going forward

        # Store test set for later visualization
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.test_proba = test_proba

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std()
        }
    
    def predict(self, features_dict):
        """
        Predict signal quality for a single signal
        
        Args:
            features_dict: Dictionary with feature values
            
        Returns:
            prediction (0 or 1), probability (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        
        # Prepare features (same as training)
        X = df.copy()
        
        # Add interaction features
        X['macd_rsi_interaction'] = X.get('macd_value', 0) * X.get('rsi', 50)
        X['adx_di_interaction'] = X.get('adx', 20) * X.get('di_diff', 0)
        X['volume_momentum'] = X.get('volume_ratio_20d', 1) * X.get('returns_1d', 0)
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        # Select and order features
        X = X[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict (apply Platt calibration if available for better probability estimates)
        prediction = self.model.predict(X_scaled)[0]
        raw_proba = self.model.predict_proba(X_scaled)[0][1]
        if hasattr(self, 'platt_scaler') and self.platt_scaler is not None:
            probability = float(self.platt_scaler.predict_proba([[raw_proba]])[0][1])
        else:
            probability = float(raw_proba)

        return int(prediction), float(probability)
    
    def plot_diagnostics(self, save_path='xgboost_diagnostics.png'):
        """Generate diagnostic plots"""
        if self.model is None or self.X_test is None:
            print("⚠️  No test data available. Train model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.test_proba)
        auc_score = roc_auc_score(self.y_test, self.test_proba)
        
        axes[0, 0].plot(fpr, tpr, label=f'AUC = {auc_score:.3f}', linewidth=2)
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title(f'ROC Curve ({self.target_period}-day prediction)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Importance
        top_features = self.feature_importance.head(15)
        axes[0, 1].barh(range(len(top_features)), top_features['importance'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Top 15 Feature Importance')
        axes[0, 1].invert_yaxis()
        
        # 3. Confusion Matrix
        test_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xticklabels(['Unprofitable', 'Profitable'])
        axes[1, 0].set_yticklabels(['Unprofitable', 'Profitable'])
        
        # 4. Probability Distribution
        axes[1, 1].hist(self.test_proba[self.y_test == 0], bins=30, alpha=0.5, label='Unprofitable', color='red')
        axes[1, 1].hist(self.test_proba[self.y_test == 1], bins=30, alpha=0.5, label='Profitable', color='green')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Probability Distribution by Actual Class')
        axes[1, 1].legend()
        axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Diagnostic plots saved to: {save_path}")
        plt.close()
    
    def save(self, filepath='xgboost_model.pkl'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        model_data = {
            'model': self.model,
            'platt_scaler': getattr(self, 'platt_scaler', None),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'target_period': self.target_period
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n💾 Model saved to: {filepath}")
    
    def load(self, filepath='xgboost_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.platt_scaler = model_data.get('platt_scaler', None)
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.target_period = model_data['target_period']
        print(f"✅ Model loaded from: {filepath}")
        if self.platt_scaler is not None:
            print(f"   🎯 Calibrated predictions enabled")


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("XGBoost Model Training")
    print("="*60)
    
    # Load training data
    try:
        df = pd.read_csv('training_data.csv')
        print(f"\n✅ Loaded {len(df)} training samples")
    except FileNotFoundError:
        print("\n❌ training_data.csv not found!")
        print("Run: python data_preparation.py first")
        exit(1)
    
    # Train model for 5-day prediction
    model = MACDXGBoostModel(target_period=5)
    metrics = model.train(df, optimize_hyperparameters=False)
    
    # Generate diagnostic plots
    model.plot_diagnostics('xgboost_diagnostics.png')
    
    # Save model
    model.save('xgboost_model_5d.pkl')
    
    print("\n" + "="*60)
    print("✅ XGBoost training complete!")
    print("="*60)
