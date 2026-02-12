"""
Master Training Script
Trains both XGBoost and LSTM models end-to-end
"""

import pandas as pd
import numpy as np
import torch
import os
from datetime import datetime
from data_preparation import DataPreparation
from ml_model import MACDXGBoostModel
from lstm_model import LSTMTrainer
from ensemble_predictor import EnsemblePredictor
import warnings
warnings.filterwarnings('ignore')


class ModelTrainingPipeline:
    """
    End-to-end pipeline for training both ML models
    """
    
    def __init__(self, target_periods=[5, 10, 20]):
        self.target_periods = target_periods
        self.data_collected = False
        
    def step1_collect_data(self, tickers=None, lookback='3y', force_recollect=False):
        """
        Step 1: Collect training data
        
        Args:
            tickers: List of tickers (if None, uses default set)
            lookback: Historical period to collect
            force_recollect: Re-collect even if data file exists
        """
        print("\n" + "="*70)
        print("STEP 1: DATA COLLECTION")
        print("="*70)
        
        # Check if data already exists
        if os.path.exists('training_data.csv') and not force_recollect:
            print("\n✅ Training data already exists: training_data.csv")
            print("✅ Using existing data (pass force_recollect=True to re-collect)")
            self.data_collected = True
            return pd.read_csv('training_data.csv')
        
        # Default ticker list (30 liquid stocks for faster training)
        if tickers is None:
            tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'MA',
                'WMT', 'PG', 'JNJ', 'UNH', 'HD', 'BAC', 'XOM', 'CVX', 'ABBV', 'PFE',
                'KO', 'PEP', 'COST', 'AVGO', 'CSCO', 'ADBE', 'NFLX', 'CRM', 'INTC', 'AMD'
            ]
            print(f"\n📊 Using default ticker set: {len(tickers)} stocks")
            print("💡 For full S&P 500, modify 'tickers' parameter")
        
        # Collect data
        prep = DataPreparation(holding_periods=self.target_periods)
        df = prep.collect_training_data(tickers, lookback_period=lookback, save_path='training_data.csv')
        
        if df is None or len(df) == 0:
            raise ValueError("❌ Data collection failed. No signals found.")
        
        self.data_collected = True
        return df
    
    def step2_train_xgboost(self, df, target_period=5, optimize=False):
        """
        Step 2: Train XGBoost model
        
        Args:
            df: Training dataframe
            target_period: Which holding period to predict
            optimize: Run hyperparameter optimization (slower)
        """
        print("\n" + "="*70)
        print(f"STEP 2: TRAIN XGBOOST MODEL ({target_period}-day prediction)")
        print("="*70)
        
        # Train model
        model = MACDXGBoostModel(target_period=target_period)
        metrics = model.train(df, test_size=0.2, optimize_hyperparameters=optimize)
        
        # Generate diagnostics
        model.plot_diagnostics(f'xgboost_diagnostics_{target_period}d.png')
        
        # Save model
        model.save(f'xgboost_model_{target_period}d.pkl')
        
        return model, metrics
    
    def step3_train_lstm(self, df, target_period=5, epochs=50):
        """
        Step 3: Train LSTM model
        
        Args:
            df: Training dataframe (for labels)
            target_period: Which holding period to predict
            epochs: Number of training epochs
        """
        print("\n" + "="*70)
        print(f"STEP 3: TRAIN LSTM MODEL ({target_period}-day prediction)")
        print("="*70)
        
        # Check for sequences file
        sequence_file = 'training_data_sequences.npy'
        if not os.path.exists(sequence_file):
            raise FileNotFoundError(
                f"❌ Sequence file not found: {sequence_file}\n"
                "Run data collection (step 1) first."
            )
        
        # Initialize trainer
        trainer = LSTMTrainer(
            input_size=8,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            target_period=target_period
        )
        
        # Prepare data
        train_loader, val_loader, test_loader = trainer.prepare_sequences(sequence_file, df)
        
        # Train
        trainer.train(train_loader, val_loader, epochs=epochs, early_stopping_patience=10)
        
        # Evaluate
        test_metrics = trainer.evaluate(test_loader)
        
        # Plot training history
        trainer.plot_training_history(f'lstm_training_history_{target_period}d.png')
        
        # Save model
        trainer.save(f'lstm_model_{target_period}d.pth')
        
        return trainer, test_metrics
    
    def step4_create_ensemble(self, target_period=5):
        """
        Step 4: Create ensemble predictor
        
        Args:
            target_period: Which holding period model to use
        """
        print("\n" + "="*70)
        print(f"STEP 4: CREATE ENSEMBLE PREDICTOR ({target_period}-day prediction)")
        print("="*70)
        
        # Initialize ensemble
        ensemble = EnsemblePredictor(
            xgboost_weight=0.4,
            lstm_weight=0.6,
            confidence_threshold=0.65,
            target_period=target_period
        )
        
        # Load models
        ensemble.load_models(
            xgboost_path=f'xgboost_model_{target_period}d.pkl',
            lstm_path=f'lstm_model_{target_period}d.pth'
        )
        
        print("\n✅ Ensemble predictor ready!")
        
        return ensemble
    
    def run_full_pipeline(self, tickers=None, lookback='10y', target_period=5, 
                         optimize_xgboost=False, lstm_epochs=50):
        """
        Run complete training pipeline
        
        Args:
            tickers: List of tickers (None = default set)
            lookback: Historical period for data collection (default: 10y)
            target_period: Which holding period to train for
            optimize_xgboost: Run hyperparameter optimization
            lstm_epochs: LSTM training epochs
        """
        start_time = datetime.now()
        
        print("\n" + "="*70)
        print("🚀 MACD MACHINE LEARNING - FULL TRAINING PIPELINE")
        print("="*70)
        print(f"⏰ Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target period: {target_period} days")
        print(f"🖥️  GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   Device: {torch.cuda.get_device_name(0)}")
        
        results = {}
        
        try:
            # Step 1: Collect data
            df = self.step1_collect_data(tickers=tickers, lookback=lookback)
            results['data_samples'] = len(df)
            
            # Step 2: Train XGBoost
            xgboost_model, xgb_metrics = self.step2_train_xgboost(
                df, target_period=target_period, optimize=optimize_xgboost
            )
            results['xgboost_metrics'] = xgb_metrics
            
            # Step 3: Train LSTM
            lstm_trainer, lstm_metrics = self.step3_train_lstm(
                df, target_period=target_period, epochs=lstm_epochs
            )
            results['lstm_metrics'] = lstm_metrics
            
            # Step 4: Create ensemble
            ensemble = self.step4_create_ensemble(target_period=target_period)
            results['ensemble'] = ensemble
            
            # Summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETE - SUMMARY")
            print("="*70)
            print(f"⏰ Total time: {duration}")
            print(f"\n📊 Data:")
            print(f"   Training samples: {results['data_samples']}")
            print(f"\n🤖 XGBoost Performance:")
            print(f"   Test Accuracy: {xgb_metrics['test_accuracy']:.3f}")
            print(f"   Test AUC:      {xgb_metrics['test_auc']:.3f}")
            print(f"\n🧠 LSTM Performance:")
            print(f"   Test Accuracy: {lstm_metrics['accuracy']:.3f}")
            print(f"   Test AUC:      {lstm_metrics['auc']:.3f}")
            print(f"\n💾 Saved Files:")
            print(f"   ✅ xgboost_model_{target_period}d.pkl")
            print(f"   ✅ lstm_model_{target_period}d.pth")
            print(f"   ✅ xgboost_diagnostics_{target_period}d.png")
            print(f"   ✅ lstm_training_history_{target_period}d.png")
            print("\n🎯 Next Steps:")
            print("   1. Review diagnostic plots")
            print("   2. Test ensemble predictor: python ensemble_predictor.py")
            print("   3. Integrate with scanner: See updated macd_scanner.py")
            print("="*70)
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


def quick_start():
    """Quick start for first-time users"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║          MACD Machine Learning Training Pipeline               ║
    ║                    Quick Start Guide                           ║
    ╚════════════════════════════════════════════════════════════════╝
    
    This will train two ML models to improve your MACD signals:
    1. XGBoost - Fast, interpretable (5 min training)
    2. LSTM - Deep learning, pattern recognition (20-30 min training)
    
    Training Options:
    ─────────────────────────────────────────────────────────────────
    [1] Quick Test (30 stocks, ~15-20 min total)
        → Good for testing, learning the system
    
    [2] Full Training (All S&P 500, ~3-4 hours total)
        → Production-ready, best performance
    
    [3] Custom Training
        → Choose your own settings
    
    [4] Load existing models (skip training)
    ─────────────────────────────────────────────────────────────────
    """)
    
    choice = input("Choose option (1-4): ").strip()
    
    pipeline = ModelTrainingPipeline(target_periods=[5, 10, 20])
    
    if choice == '1':
        print("\n🚀 Quick Test Mode Selected")
        print("   Using 30 stocks, 10-year history")
        results = pipeline.run_full_pipeline(
            tickers=None,  # Default 30 stocks
            lookback='10y',
            target_period=5,
            optimize_xgboost=False,
            lstm_epochs=50
        )
    
    elif choice == '2':
        print("\n🚀 Full Training Mode Selected")
        prep = DataPreparation()
        all_tickers = prep.load_sp500_tickers()
        print(f"   Using {len(all_tickers)} S&P 500 stocks, 10-year history")
        results = pipeline.run_full_pipeline(
            tickers=all_tickers,
            lookback='10y',
            target_period=5,
            optimize_xgboost=True,  # Full optimization
            lstm_epochs=100
        )
    
    elif choice == '3':
        print("\n🚀 Custom Training Mode")
        use_sp500 = input("Use full S&P 500? (y/n): ").lower() == 'y'
        lookback = input("Lookback period (3y/5y/10y/max) [10y]: ").strip() or '10y'
        target = int(input("Target period (5/10/20 days) [5]: ").strip() or '5')
        epochs = int(input("LSTM epochs [50]: ").strip() or '50')
        
        if use_sp500:
            prep = DataPreparation()
            tickers = prep.load_sp500_tickers()
        else:
            tickers = None
        
        results = pipeline.run_full_pipeline(
            tickers=tickers,
            lookback=lookback,
            target_period=target,
            optimize_xgboost=use_sp500,
            lstm_epochs=epochs
        )
    
    elif choice == '4':
        print("\n✅ Loading existing models...")
        ensemble = EnsemblePredictor(target_period=5)
        try:
            ensemble.load_models()
            print("\n✅ Models loaded! Ready to use.")
            print("   Run: python macd_scanner.py (with ML enabled)")
        except Exception as e:
            print(f"\n❌ Error loading models: {e}")
            print("   Train models first (option 1 or 2)")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    import sys
    
    # Check if running directly or with arguments
    if len(sys.argv) > 1:
        # Advanced usage with command line arguments
        if sys.argv[1] == '--quick':
            pipeline = ModelTrainingPipeline()
            pipeline.run_full_pipeline(target_period=5, lstm_epochs=50)
        elif sys.argv[1] == '--full':
            pipeline = ModelTrainingPipeline()
            prep = DataPreparation()
            tickers = prep.load_sp500_tickers()
            pipeline.run_full_pipeline(tickers=tickers, target_period=5, 
                                      optimize_xgboost=True, lstm_epochs=100)
        else:
            print("Usage: python train_models.py [--quick | --full]")
    else:
        # Interactive mode
        quick_start()
