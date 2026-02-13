"""
Ensemble Predictor - Combines XGBoost and LSTM predictions
Uses weighted voting for final signal quality prediction
"""

import numpy as np
import torch
from ml_model import MACDXGBoostModel
from lstm_model import LSTMTrainer, MACDLSTMModel
import warnings
warnings.filterwarnings('ignore')


class EnsemblePredictor:
    """
    Ensemble model combining XGBoost and LSTM predictions
    
    Weighting strategy:
    - XGBoost: Better AUC (0.572-0.595), favored at 70%
    - LSTM: Weaker AUC (~0.545), overfits, reduced to 30%
    """

    def __init__(self, xgboost_weight=0.7, lstm_weight=0.3,
                 confidence_threshold=0.60, target_period=5):
        """
        Args:
            xgboost_weight: Weight for XGBoost prediction (0-1)
            lstm_weight: Weight for LSTM prediction (0-1)
            confidence_threshold: Minimum ensemble confidence to accept signal
            target_period: Prediction horizon (5, 10, or 20 days)
        """
        self.xgboost_weight = xgboost_weight
        self.lstm_weight = lstm_weight
        self.confidence_threshold = confidence_threshold
        self.target_period = target_period
        
        # Normalize weights
        total_weight = xgboost_weight + lstm_weight
        self.xgboost_weight /= total_weight
        self.lstm_weight /= total_weight
        
        # Models
        self.xgboost_model = None
        self.lstm_model = None
        print("="*60)
        print("Ensemble Predictor Initialized")
        print("="*60)
        print(f"XGBoost weight: {self.xgboost_weight:.1%}")
        print(f"LSTM weight:    {self.lstm_weight:.1%}")
        print(f"Confidence threshold: {self.confidence_threshold:.1%}")
        print(f"Target period: {self.target_period} days")
    
    def load_models(self, xgboost_path='xgboost_model_5d.pkl', 
                   lstm_path='lstm_model_5d.pth'):
        """
        Load both trained models
        
        Args:
            xgboost_path: Path to XGBoost model file
            lstm_path: Path to LSTM model file
        """
        print("\n📥 Loading models...")
        
        # Load XGBoost
        try:
            self.xgboost_model = MACDXGBoostModel(target_period=self.target_period)
            self.xgboost_model.load(xgboost_path)
        except FileNotFoundError:
            print(f"⚠️  XGBoost model not found at {xgboost_path}")
            print("   Ensemble will use LSTM only")
            self.xgboost_model = None
        
        # Load LSTM
        try:
            # Detect device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.lstm_model = LSTMTrainer(
                input_size=8,
                hidden_size=64,
                num_layers=1,
                dropout=0.5,
                target_period=self.target_period,
                device=device
            )
            self.lstm_model.load(lstm_path)
        except FileNotFoundError:
            print(f"⚠️  LSTM model not found at {lstm_path}")
            print("   Ensemble will use XGBoost only")
            self.lstm_model = None
        
        if self.xgboost_model is None and self.lstm_model is None:
            raise ValueError("❌ No models available! Train models first.")
        
        print("✅ Models loaded successfully\n")
    
    def predict(self, snapshot_features, sequence_features):
        """
        Make ensemble prediction
        
        Args:
            snapshot_features: Dict of features for XGBoost
            sequence_features: Numpy array (seq_len, num_features) for LSTM
            
        Returns:
            {
                'ensemble_prediction': 0 or 1,
                'ensemble_confidence': float (0-1),
                'xgboost_confidence': float (0-1) or None,
                'lstm_confidence': float (0-1) or None,
                'accept_signal': bool (confidence > threshold),
                'signal_grade': str (A+, A, B, C, D, F)
            }
        """
        xgboost_pred = None
        xgboost_conf = None
        lstm_conf = None
        
        # Get XGBoost prediction
        if self.xgboost_model is not None:
            try:
                xgboost_pred, xgboost_conf = self.xgboost_model.predict(snapshot_features)
            except Exception as e:
                print(f"⚠️  XGBoost prediction error: {e}")
                xgboost_conf = None
        
        # Get LSTM prediction
        if self.lstm_model is not None:
            try:
                lstm_conf = self.lstm_model.predict_single(sequence_features)
            except Exception as e:
                print(f"⚠️  LSTM prediction error: {e}")
                lstm_conf = None
        
        # Ensemble prediction
        if xgboost_conf is not None and lstm_conf is not None:
            # Both models available
            ensemble_conf = (self.xgboost_weight * xgboost_conf + 
                           self.lstm_weight * lstm_conf)
        elif xgboost_conf is not None:
            # Only XGBoost available
            ensemble_conf = xgboost_conf
        elif lstm_conf is not None:
            # Only LSTM available
            ensemble_conf = lstm_conf
        else:
            # No models available
            raise ValueError("No predictions available from either model")
        
        # Binary prediction
        ensemble_pred = 1 if ensemble_conf >= 0.5 else 0
        
        # Accept signal if confidence exceeds threshold
        accept_signal = ensemble_conf >= self.confidence_threshold
        
        # Grade signal (like school grades)
        signal_grade = self._grade_signal(ensemble_conf)
        
        return {
            'ensemble_prediction': ensemble_pred,
            'ensemble_confidence': round(ensemble_conf, 3),
            'xgboost_confidence': round(xgboost_conf, 3) if xgboost_conf is not None else None,
            'lstm_confidence': round(lstm_conf, 3) if lstm_conf is not None else None,
            'accept_signal': accept_signal,
            'signal_grade': signal_grade,
        }
    
    def _grade_signal(self, confidence):
        """
        Grade signal quality based on confidence
        
        A+ (95-100%): Exceptional
        A  (85-95%):  Excellent
        B  (75-85%):  Good
        C  (65-75%):  Fair
        D  (50-65%):  Poor
        F  (<50%):    Reject
        """
        if confidence >= 0.95:
            return 'A+'
        elif confidence >= 0.85:
            return 'A'
        elif confidence >= 0.75:
            return 'B'
        elif confidence >= 0.65:
            return 'C'
        elif confidence >= 0.50:
            return 'D'
        else:
            return 'F'
    
    def batch_predict(self, snapshot_features_list, sequence_features_list):
        """
        Make predictions for multiple signals
        
        Args:
            snapshot_features_list: List of feature dicts
            sequence_features_list: List of sequence arrays
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for snap_feat, seq_feat in zip(snapshot_features_list, sequence_features_list):
            pred = self.predict(snap_feat, seq_feat)
            predictions.append(pred)
        
        return predictions
    
    def optimize_weights(self, validation_data, validation_sequences, validation_labels):
        """
        Optimize ensemble weights using validation data
        
        Args:
            validation_data: List of snapshot feature dicts
            validation_sequences: List of sequence arrays
            validation_labels: True labels (0 or 1)
            
        Returns:
            Best weights and accuracy
        """
        print("\n🔍 Optimizing ensemble weights...")
        
        best_accuracy = 0
        best_weights = (self.xgboost_weight, self.lstm_weight)
        
        # Try different weight combinations
        weight_combinations = [
            (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)
        ]
        
        for xgb_w, lstm_w in weight_combinations:
            self.xgboost_weight = xgb_w
            self.lstm_weight = lstm_w
            
            # Get predictions
            predictions = []
            for snap_feat, seq_feat in zip(validation_data, validation_sequences):
                pred = self.predict(snap_feat, seq_feat)
                predictions.append(pred['ensemble_prediction'])
            
            # Calculate accuracy
            accuracy = np.mean(np.array(predictions) == np.array(validation_labels))
            
            print(f"   XGB: {xgb_w:.1f}, LSTM: {lstm_w:.1f} → Accuracy: {accuracy:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = (xgb_w, lstm_w)
        
        # Set best weights
        self.xgboost_weight, self.lstm_weight = best_weights
        
        print(f"\n✅ Best weights: XGB={self.xgboost_weight:.1f}, LSTM={self.lstm_weight:.1f}")
        print(f"   Validation Accuracy: {best_accuracy:.3f}")
        
        return best_weights, best_accuracy
    
    def set_confidence_threshold(self, threshold):
        """
        Update confidence threshold
        
        Higher threshold = Fewer but higher quality signals
        Lower threshold = More signals but lower quality
        
        Recommended thresholds:
        - Conservative: 0.75-0.80 (high quality, fewer signals)
        - Balanced: 0.65-0.70 (good quality, moderate signals)
        - Aggressive: 0.55-0.60 (more signals, mixed quality)
        """
        old_threshold = self.confidence_threshold
        self.confidence_threshold = threshold
        print(f"📊 Confidence threshold updated: {old_threshold:.1%} → {threshold:.1%}")
    
    def get_statistics(self, predictions):
        """
        Get statistics from a batch of predictions
        
        Args:
            predictions: List of prediction dicts from batch_predict
            
        Returns:
            Dict with statistics
        """
        accepted = [p for p in predictions if p['accept_signal']]
        
        grades = {}
        for grade in ['A+', 'A', 'B', 'C', 'D', 'F']:
            grades[grade] = sum(1 for p in predictions if p['signal_grade'] == grade)
        
        return {
            'total_signals': len(predictions),
            'accepted_signals': len(accepted),
            'rejection_rate': (1 - len(accepted) / len(predictions)) * 100 if predictions else 0,
            'avg_confidence': np.mean([p['ensemble_confidence'] for p in predictions]) if predictions else 0,
            'grade_distribution': grades
        }
    
    def print_prediction(self, prediction_result):
        """Pretty print a prediction result"""
        print("\n" + "="*50)
        print("🎯 Ensemble Prediction")
        print("="*50)
        print(f"Ensemble Confidence: {prediction_result['ensemble_confidence']:.1%}")
        print(f"Signal Grade:        {prediction_result['signal_grade']}")
        print(f"Accept Signal:       {'✅ YES' if prediction_result['accept_signal'] else '❌ NO'}")
        
        if prediction_result['xgboost_confidence'] is not None:
            print(f"\nXGBoost Confidence:  {prediction_result['xgboost_confidence']:.1%}")
        if prediction_result['lstm_confidence'] is not None:
            print(f"LSTM Confidence:     {prediction_result['lstm_confidence']:.1%}")
        
        print("="*50)


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Ensemble Predictor Demo")
    print("="*60)
    
    # Initialize ensemble
    ensemble = EnsemblePredictor(
        xgboost_weight=0.7,
        lstm_weight=0.3,
        confidence_threshold=0.60,
        target_period=5
    )
    
    # Load models
    try:
        ensemble.load_models(
            xgboost_path='xgboost_model_5d.pkl',
            lstm_path='lstm_model_5d.pth'
        )
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        print("Train models first using: python train_models.py")
        exit(1)
    
    # Example prediction
    print("\n" + "="*60)
    print("Example: Testing ensemble on sample signal")
    print("="*60)
    
    # Sample features (bullish crossover with good indicators)
    sample_snapshot = {
        'macd_value': 0.15,
        'macd_signal': 0.10,
        'macd_histogram': 0.05,
        'macd_above_zero': 1,
        'histogram_slope': 0.02,
        'histogram_momentum': 0.04,
        'rsi': 55,
        'rsi_oversold': 0,
        'rsi_overbought': 0,
        'rsi_neutral': 1,
        'rsi_slope': 5,
        'adx': 28,
        'plus_di': 25,
        'minus_di': 15,
        'di_diff': 10,
        'adx_strong': 1,
        'price': 150.0,
        'distance_from_ema200_pct': 3.5,
        'price_above_ema200': 1,
        'returns_1d': 0.5,
        'returns_5d': 2.0,
        'returns_10d': 3.5,
        'returns_20d': 5.0,
        'volume': 1000000,
        'volume_ratio_20d': 1.3,
        'volume_trend': 1,
        'bb_position': 0.6,
        'bb_bandwidth': 0.04,
        'bb_squeeze': 0,
        'volatility_10d': 1.5,
        'volatility_20d': 1.8,
        'crossover_type': 1
    }
    
    # Sample sequence (30 days of normalized indicators)
    sample_sequence = np.random.randn(30, 8) * 0.5  # Random normalized data for demo
    
    # Make prediction
    result = ensemble.predict(sample_snapshot, sample_sequence)
    ensemble.print_prediction(result)
    
    print("\n✅ Ensemble predictor ready for integration!")
