"""
MACD Trading Model Scanner for S&P 500 Stocks
Identifies fresh MACD crossover signals (0-7 days) with EMA-200 confirmation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import ML components (optional)
try:
    from ensemble_predictor import EnsemblePredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML modules not available. Run without ML filtering.")


class MACDScanner:
    """
    MACD Trading Model with multiple confirmation factors:
    - MACD crossovers (12, 26, 9)
    - EMA-200 trend filter
    - Signal freshness (0-7 days)
    - Volume confirmation
    - Price momentum
    - Optional: RSI, ADX, Bollinger Bands
    """
    
    def __init__(self, macd_fast=12, macd_slow=26, macd_signal=9, ema_period=200,
                 use_rsi=True, use_adx=True, use_bollinger=True, use_ema200=True,
                 use_ml_filter=False, ml_confidence_threshold=0.65, ml_target_period=5):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_period = ema_period
        
        # Indicator toggles
        self.use_rsi = use_rsi
        self.use_adx = use_adx
        self.use_bollinger = use_bollinger
        self.use_ema200 = use_ema200
        
        # ML settings
        self.use_ml_filter = use_ml_filter and ML_AVAILABLE
        self.ml_confidence_threshold = ml_confidence_threshold
        self.ml_target_period = ml_target_period
        self.ml_ensemble = None
        
        # Load ML models if requested
        if self.use_ml_filter:
            self._load_ml_models()
        
    def _load_ml_models(self):
        """Load trained ML models for signal filtering"""
        try:
            xgb_path = f'xgboost_model_{self.ml_target_period}d.pkl'
            lstm_path = f'lstm_model_{self.ml_target_period}d.pth'
            
            if not os.path.exists(xgb_path) or not os.path.exists(lstm_path):
                print(f"⚠️  ML models not found. Train models first using: python train_models.py")
                self.use_ml_filter = False
                return
            
            self.ml_ensemble = EnsemblePredictor(
                xgboost_weight=0.4,
                lstm_weight=0.6,
                confidence_threshold=self.ml_confidence_threshold,
                target_period=self.ml_target_period
            )
            self.ml_ensemble.load_models(xgb_path, lstm_path)
            self.ml_ensemble.load_exit_model()  # Load exit timing model if available
            print(f"✅ ML models loaded ({self.ml_target_period}-day prediction, threshold: {self.ml_confidence_threshold:.0%})")
        except Exception as e:
            print(f"⚠️  Error loading ML models: {e}")
            self.use_ml_filter = False
    
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index (RSI)"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_adx(self, data, period=14):
        """Calculate Average Directional Index (ADX) for trend strength"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0
        
        # Smooth the values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        middle_band = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        # Calculate bandwidth (volatility indicator)
        bandwidth = (upper_band - lower_band) / middle_band
        
        # Calculate %B (position within bands)
        percent_b = (data['Close'] - lower_band) / (upper_band - lower_band)
        
        return upper_band, middle_band, lower_band, bandwidth, percent_b
    
    def calculate_macd(self, data):
        """
        Calculate MACD, Signal Line, and Histogram
        Returns: MACD line, Signal line, Histogram
        """
        ema_fast = self.calculate_ema(data['Close'], self.macd_fast)
        ema_slow = self.calculate_ema(data['Close'], self.macd_slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, self.macd_signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def detect_crossover(self, macd_line, signal_line):
        """
        Detect MACD crossovers
        Returns: 1 for bullish crossover, -1 for bearish crossover, 0 for no crossover
        """
        crossover = pd.Series(0, index=macd_line.index)
        
        # Bullish crossover: MACD crosses above Signal
        bullish = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        crossover[bullish] = 1
        
        # Bearish crossover: MACD crosses below Signal
        bearish = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        crossover[bearish] = -1
        
        return crossover
    
    def analyze_stock(self, ticker, period='1y'):
        """
        Analyze a single stock for MACD signals
        Returns: Dictionary with signal details or None
        """
        try:
            # Download data
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if len(data) < self.ema_period:
                return None
            
            # Calculate indicators
            macd_line, signal_line, histogram = self.calculate_macd(data)
            
            # Add to dataframe
            data['MACD'] = macd_line
            data['Signal'] = signal_line
            data['Histogram'] = histogram
            
            # Conditional indicators based on toggles
            if self.use_ema200:
                ema_200 = self.calculate_ema(data['Close'], self.ema_period)
                data['EMA_200'] = ema_200
            
            if self.use_rsi:
                rsi = self.calculate_rsi(data, period=14)
                data['RSI'] = rsi
            
            if self.use_adx:
                adx, plus_di, minus_di = self.calculate_adx(data, period=14)
                data['ADX'] = adx
                data['Plus_DI'] = plus_di
                data['Minus_DI'] = minus_di
            
            if self.use_bollinger:
                bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b = self.calculate_bollinger_bands(data)
                data['BB_Upper'] = bb_upper
                data['BB_Middle'] = bb_middle
                data['BB_Lower'] = bb_lower
                data['BB_Bandwidth'] = bb_bandwidth
                data['BB_PercentB'] = bb_percent_b
            
            # Detect crossovers
            data['Crossover'] = self.detect_crossover(macd_line, signal_line)
            
            # Find most recent crossover
            crossover_days = data[data['Crossover'] != 0]
            
            if len(crossover_days) == 0:
                return None
            
            last_crossover_date = crossover_days.index[-1]
            days_since_crossover = (data.index[-1] - last_crossover_date).days
            
            # Get MACD value at crossover to determine if above/below zero line
            crossover_macd_value = data.loc[last_crossover_date, 'MACD']
            
            # Filter for fresh signals (0-7 days)
            if days_since_crossover > 7:
                return None
            
            # Get latest data
            latest = data.iloc[-1]
            crossover_type = crossover_days.iloc[-1]['Crossover']
            
            # Trading signal logic with multiple confirmations
            signal = self.generate_signal(data, latest, crossover_type, crossover_macd_value)
            
            if signal is None:
                return None
            
            # Calculate additional metrics
            avg_volume_20 = data['Volume'].tail(20).mean()
            volume_ratio = latest['Volume'] / avg_volume_20 if avg_volume_20 > 0 else 1
            
            price_change_pct = ((latest['Close'] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100) if len(data) >= 5 else 0
            
            # Calculate trading levels (Entry, Stop Loss, Take Profit)
            entry_price = latest['Close']
            risk_reward_ratio = 1.5  # 1.5:1 reward to risk
            
            if 'LONG' in signal:
                # For LONG positions
                # Stop loss: 5% below entry
                stop_loss = entry_price * 0.95  # 5% below entry
                
                risk_per_share = entry_price - stop_loss
                take_profit = entry_price + (risk_per_share * risk_reward_ratio)
                
            else:  # SHORT positions
                # For SHORT positions
                # Stop loss: 5% above entry
                stop_loss = entry_price * 1.05  # 5% above entry
                
                risk_per_share = stop_loss - entry_price
                take_profit = entry_price - (risk_per_share * risk_reward_ratio)
            
            # Build result dictionary with conditional fields
            result = {
                'ticker': ticker,
                'signal': signal,
                'days_since_crossover': days_since_crossover,
                'crossover_position': 'Above Zero' if crossover_macd_value > 0 else 'Below Zero',
                'current_price': round(latest['Close'], 2),
                'entry_price': round(entry_price, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'risk_reward_ratio': f"{risk_reward_ratio}:1",
                'macd': round(latest['MACD'], 4),
                'signal_line': round(latest['Signal'], 4),
                'histogram': round(latest['Histogram'], 4)
            }
            
            # Add optional indicator values if enabled
            if self.use_rsi:
                result['rsi'] = round(latest['RSI'], 2)
            
            if self.use_adx:
                result['adx'] = round(latest['ADX'], 2)
            
            if self.use_bollinger:
                result['bb_position'] = round(latest['BB_PercentB'], 2)
            
            if self.use_ema200:
                result['ema_200'] = round(latest['EMA_200'], 2)
                result['price_vs_ema200'] = 'Above' if latest['Close'] > latest['EMA_200'] else 'Below'
                result['distance_from_ema200_pct'] = round((latest['Close'] - latest['EMA_200']) / latest['EMA_200'] * 100, 2)
            
            # Always include volume and price change
            result['volume_ratio'] = round(volume_ratio, 2)
            result['price_change_5d_pct'] = round(price_change_pct, 2)
            result['crossover_date'] = last_crossover_date.strftime('%Y-%m-%d')
            result['current_date'] = data.index[-1].strftime('%Y-%m-%d')
            
            # ML Filtering (if enabled)
            if self.use_ml_filter and self.ml_ensemble is not None:
                try:
                    # Extract features for ML
                    idx = data.index.get_loc(data.index[-1])
                    snapshot_features, sequence_features = self._extract_ml_features(data, idx, crossover_type)
                    
                    if snapshot_features is not None and sequence_features is not None:
                        # Get ML prediction
                        ml_result = self.ml_ensemble.predict(snapshot_features, sequence_features)
                        
                        # Add ML info to result
                        result['ml_confidence'] = ml_result['ensemble_confidence']
                        result['ml_grade'] = ml_result['signal_grade']
                        result['signal'] = f"{signal} ({ml_result['ml_grade']})"  # Add grade to signal
                        
                        # Add exit recommendation if available
                        if 'recommended_exit_day' in ml_result:
                            result['recommended_exit_day'] = ml_result['recommended_exit_day']
                            result['exit_range'] = ml_result['exit_range']
                            result['exit_label'] = ml_result['exit_label']
                        
                        # Filter out low-confidence signals
                        if not ml_result['accept_signal']:
                            return None  # Reject this signal
                except Exception as e:
                    print(f"⚠️  ML prediction error for {ticker}: {e}")
                    # Continue without ML filtering if error occurs
            
            return result
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            return None
    
    def generate_signal(self, data, latest, crossover_type, crossover_macd_value):
        """
        Generate trading signal with multiple confirmation factors
        
        MACD Zero Line Position:
        - Crossovers ABOVE zero line = Stronger bullish signals (uptrend confirmation)
        - Crossovers BELOW zero line = Stronger bearish signals (downtrend confirmation)
        
        LONG Signal Criteria:
        - Bullish MACD crossover (MACD crosses above Signal)
        - Crossover above zero line preferred (stronger signal)
        - Price above EMA-200 (uptrend confirmation)
        - RSI not overbought (< 70) or recovering from oversold
        - ADX > 20 (strong trend)
        - Bollinger Band position favorable (not extremely overbought)
        
        SHORT Signal Criteria:
        - Bearish MACD crossover (MACD crosses below Signal)
        - Crossover below zero line preferred (stronger signal)
        - Price below EMA-200 (downtrend confirmation)
        - RSI not oversold (> 30) or recovering from overbought
        - ADX > 20 (strong trend)
        - Bollinger Band position favorable (not extremely oversold)
        """
        
        macd_above_zero = latest['MACD'] > 0
        crossover_above_zero = crossover_macd_value > 0
        histogram_increasing = latest['Histogram'] > data['Histogram'].iloc[-2] if len(data) >= 2 else False
        
        # EMA-200 conditions (if enabled)
        if self.use_ema200:
            price_above_ema200 = latest['Close'] > latest['EMA_200']
        else:
            price_above_ema200 = True  # Neutral if disabled
        
        # RSI conditions (if enabled)
        if self.use_rsi:
            rsi = latest['RSI']
            rsi_bullish = rsi < 70 and rsi > 30  # Not overbought, not oversold
            rsi_oversold = rsi < 35  # Potential bounce zone
            rsi_overbought = rsi > 65  # Potential reversal zone
        else:
            rsi_bullish = True  # Neutral if disabled
            rsi_oversold = False
            rsi_overbought = False
        
        # ADX conditions (if enabled)
        if self.use_adx:
            adx = latest['ADX']
            strong_trend = adx > 25  # Strong trend
            moderate_trend = adx > 20  # Moderate trend
        else:
            strong_trend = True  # Neutral if disabled
            moderate_trend = True
        
        # Bollinger Band conditions (if enabled)
        if self.use_bollinger:
            bb_position = latest['BB_PercentB']
            bb_oversold = bb_position < 0.2  # Near lower band
            bb_overbought = bb_position > 0.8  # Near upper band
            bb_neutral = 0.3 < bb_position < 0.7  # In middle zone
        else:
            bb_oversold = False
            bb_overbought = False
            bb_neutral = True  # Neutral if disabled
        
        # LONG Signal
        if crossover_type == 1:  # Bullish crossover
            # STRONG LONG: Crossover above zero line with multiple confirmations
            if (crossover_above_zero and price_above_ema200 and 
                strong_trend and rsi_bullish and not bb_overbought):
                return 'STRONG LONG'
            # STRONG LONG: Above zero + oversold bounce
            elif (crossover_above_zero and price_above_ema200 and 
                  (rsi_oversold or bb_oversold) and moderate_trend):
                return 'STRONG LONG'
            # LONG: Crossover above zero with some confirmations
            elif (crossover_above_zero and price_above_ema200 and 
                  rsi < 70 and moderate_trend):
                return 'LONG'
            # LONG: Crossover below zero but strong recovery signals
            elif (not crossover_above_zero and price_above_ema200 and 
                  strong_trend and rsi_oversold and bb_oversold):
                return 'LONG'
            # LONG: Price above EMA-200 with decent conditions
            elif (price_above_ema200 and rsi_bullish and 
                  (moderate_trend or histogram_increasing) and (bb_neutral or not self.use_bollinger)):
                return 'LONG'
            else:
                return None  # Weak signal, skip
        
        # SHORT Signal
        elif crossover_type == -1:  # Bearish crossover
            # STRONG SHORT: Crossover below zero line with multiple confirmations
            if (not crossover_above_zero and not price_above_ema200 and 
                strong_trend and rsi_bullish and not bb_oversold):
                return 'STRONG SHORT'
            # STRONG SHORT: Below zero + overbought reversal
            elif (not crossover_above_zero and not price_above_ema200 and 
                  (rsi_overbought or bb_overbought) and moderate_trend):
                return 'STRONG SHORT'
            # SHORT: Crossover below zero with some confirmations
            elif (not crossover_above_zero and not price_above_ema200 and 
                  rsi > 30 and moderate_trend):
                return 'SHORT'
            # SHORT: Crossover above zero but strong reversal signals
            elif (crossover_above_zero and not price_above_ema200 and 
                  strong_trend and rsi_overbought and bb_overbought):
                return 'SHORT'
            # SHORT: Price below EMA-200 with decent conditions
            elif (not price_above_ema200 and rsi_bullish and 
                  (moderate_trend or not histogram_increasing) and (bb_neutral or not self.use_bollinger)):
                return 'SHORT'
            else:
                return None  # Weak signal, skip
        
        return None
    
    def _extract_ml_features(self, data, idx, crossover_type):
        """
        Extract ML features for ensemble prediction
        Returns snapshot features (dict) and sequence features (numpy array)
        """
        if idx < 20:
            return None, None
        
        window = data.iloc[max(0, idx-60):idx+1]
        latest = window.iloc[-1]
        
        # Snapshot features for XGBoost
        snapshot_features = {
            'macd_value': latest['MACD'],
            'macd_signal': latest['Signal'],
            'macd_histogram': latest['Histogram'],
            'macd_above_zero': int(latest['MACD'] > 0),
            'histogram_slope': (window['Histogram'].iloc[-1] - window['Histogram'].iloc[-5]) / 5 if len(window) >= 5 else 0,
            'histogram_momentum': window['Histogram'].iloc[-5:].mean() if len(window) >= 5 else 0,
            'rsi': latest.get('RSI', 50),
            'rsi_oversold': int(latest.get('RSI', 50) < 30),
            'rsi_overbought': int(latest.get('RSI', 50) > 70),
            'rsi_neutral': int(30 <= latest.get('RSI', 50) <= 70),
            'rsi_slope': (latest.get('RSI', 50) - window.get('RSI', pd.Series([50]*len(window))).iloc[-5]) / 5 if len(window) >= 5 else 0,
            'adx': latest.get('ADX', 20),
            'plus_di': latest.get('Plus_DI', 20),
            'minus_di': latest.get('Minus_DI', 20),
            'di_diff': latest.get('Plus_DI', 20) - latest.get('Minus_DI', 20),
            'adx_strong': int(latest.get('ADX', 20) > 25),
            'price': latest['Close'],
            'distance_from_ema200_pct': ((latest['Close'] - latest.get('EMA_200', latest['Close'])) / latest.get('EMA_200', latest['Close']) * 100) if 'EMA_200' in latest and latest.get('EMA_200', 0) > 0 else 0,
            'price_above_ema200': int(latest['Close'] > latest.get('EMA_200', latest['Close'])),
            'returns_1d': ((latest['Close'] - window.iloc[-2]['Close']) / window.iloc[-2]['Close'] * 100) if len(window) >= 2 else 0,
            'returns_5d': ((latest['Close'] - window.iloc[-6]['Close']) / window.iloc[-6]['Close'] * 100) if len(window) >= 6 else 0,
            'returns_10d': ((latest['Close'] - window.iloc[-11]['Close']) / window.iloc[-11]['Close'] * 100) if len(window) >= 11 else 0,
            'returns_20d': ((latest['Close'] - window.iloc[-21]['Close']) / window.iloc[-21]['Close'] * 100) if len(window) >= 21 else 0,
            'volume_ratio_20d': latest['Volume'] / window['Volume'].mean() if window['Volume'].mean() > 0 else 1,
            'volume_trend': int(window['Volume'].iloc[-5:].mean() > window['Volume'].iloc[-11:-5].mean()) if len(window) >= 11 else 0,
            'bb_position': latest.get('BB_PercentB', 0.5),
            'bb_bandwidth': latest.get('BB_Bandwidth', 0.04),
            'bb_squeeze': int(latest.get('BB_Bandwidth', 0.04) < window.get('BB_Bandwidth', pd.Series([0.04]*len(window))).quantile(0.2)) if len(window) >= 20 else 0,
            'volatility_10d': window['Close'].pct_change().iloc[-10:].std() * 100 if len(window) >= 10 else 0,
            'volatility_20d': window['Close'].pct_change().iloc[-20:].std() * 100 if len(window) >= 20 else 0,
            'crossover_type': crossover_type,
        }
        
        # Sequence features for LSTM (last 30 days)
        sequence_length = min(30, len(window))
        sequence_cols = ['Close', 'Volume', 'MACD', 'Signal']
        
        # Add optional indicators if they exist
        if 'RSI' in window.columns:
            sequence_cols.append('RSI')
        else:
            window['RSI'] = 50
            sequence_cols.append('RSI')
        
        if 'ADX' in window.columns:
            sequence_cols.append('ADX')
        else:
            window['ADX'] = 20
            sequence_cols.append('ADX')
        
        if 'BB_PercentB' in window.columns:
            sequence_cols.append('BB_PercentB')
        else:
            window['BB_PercentB'] = 0.5
            sequence_cols.append('BB_PercentB')
        
        if 'EMA_200' in window.columns:
            sequence_cols.append('EMA_200')
        else:
            window['EMA_200'] = window['Close'].mean()
            sequence_cols.append('EMA_200')
        
        sequence_data = window.iloc[-sequence_length:][sequence_cols]
        
        # Normalize sequence
        sequence_normalized = sequence_data.copy()
        for col in sequence_data.columns:
            if sequence_data[col].std() > 0:
                sequence_normalized[col] = (sequence_data[col] - sequence_data[col].mean()) / sequence_data[col].std()
            else:
                sequence_normalized[col] = 0
        
        return snapshot_features, sequence_normalized.values
    
    def get_sp500_tickers(self):
        """
        Get list of S&P 500 tickers from Wikipedia
        """
        try:
            import requests
            from io import StringIO
            
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            tables = pd.read_html(StringIO(response.text))
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            # Clean tickers (replace dots with dashes for Yahoo Finance)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            return tickers
        except Exception as e:
            print(f"Error fetching S&P 500 list: {str(e)}")
            # Fallback to a larger sample if Wikipedia fails
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'WMT',
                    'JNJ', 'PG', 'MA', 'HD', 'BAC', 'XOM', 'LLY', 'AVGO', 'CVX', 'ABBV',
                    'PFE', 'KO', 'COST', 'MRK', 'PEP', 'TMO', 'CSCO', 'WFC', 'ACN', 'MCD',
                    'ABT', 'DHR', 'ADBE', 'TXN', 'DIS', 'VZ', 'CMCSA', 'NEE', 'NKE', 'CRM',
                    'INTC', 'UNP', 'PM', 'RTX', 'HON', 'UPS', 'QCOM', 'AMD', 'INTU', 'AMAT']
    
    def scan_all_stocks(self, tickers=None):
        """
        Scan all S&P 500 stocks for fresh MACD signals
        Returns: DataFrame with signals sorted by strength
        """
        if tickers is None:
            print("Fetching S&P 500 ticker list...")
            tickers = self.get_sp500_tickers()
        
        print(f"Scanning {len(tickers)} stocks for fresh MACD signals (0-7 days)...")
        print("This may take a few minutes...\n")
        
        results = []
        processed = 0
        
        for ticker in tickers:
            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed}/{len(tickers)} stocks... Found {len(results)} signals so far")
            
            result = self.analyze_stock(ticker)
            if result:
                results.append(result)
        
        print(f"\nCompleted! Found {len(results)} fresh signals out of {len(tickers)} stocks.\n")
        
        if len(results) == 0:
            return pd.DataFrame()
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(results)
        
        # Sort by signal strength and days since crossover
        signal_priority = {'STRONG LONG': 1, 'LONG': 2, 'SHORT': 3, 'STRONG SHORT': 4}
        df['priority'] = df['signal'].map(signal_priority)
        df = df.sort_values(['priority', 'days_since_crossover'])
        df = df.drop('priority', axis=1)
        
        return df


def main():
    """
    Main function to run the MACD scanner
    """
    print("=" * 80)
    print("MACD Trading Model Scanner - S&P 500")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nScanning for fresh MACD signals (0-7 days after crossover)...")
    print("-" * 80)
    
    # ============ INDICATOR TOGGLES ============
    # Set to True/False to enable/disable each indicator
    USE_RSI = True        # Relative Strength Index (overbought/oversold)
    USE_ADX = True        # Average Directional Index (trend strength)
    USE_BOLLINGER = True  # Bollinger Bands (volatility/position)
    USE_EMA200 = True     # 200-period EMA (trend filter)
    # ==========================================
    
    # ============ MACHINE LEARNING FILTER ============
    # Set to True to enable ML-based signal filtering (requires trained models)
    # This will filter out low-quality signals using XGBoost + LSTM ensemble
    USE_ML_FILTER = True            # Enable/disable ML filtering
    ML_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence (0.5-0.95)
    ML_TARGET_PERIOD = 5            # Prediction horizon: 5, 10, or 20 days
    # =================================================
    
    print("\nActive Indicators:")
    print(f"  RSI: {'ON' if USE_RSI else 'OFF'}")
    print(f"  ADX: {'ON' if USE_ADX else 'OFF'}")
    print(f"  Bollinger Bands: {'ON' if USE_BOLLINGER else 'OFF'}")
    print(f"  EMA-200: {'ON' if USE_EMA200 else 'OFF'}")
    
    if ML_AVAILABLE and USE_ML_FILTER:
        print(f"\n🤖 Machine Learning Filter: ON")
        print(f"   Confidence Threshold: {ML_CONFIDENCE_THRESHOLD:.0%}")
        print(f"   Prediction Period: {ML_TARGET_PERIOD} days")
    else:
        print(f"\n🤖 Machine Learning Filter: OFF")
    
    print("-" * 80)
    
    # Initialize scanner with toggles
    scanner = MACDScanner(
        macd_fast=12, 
        macd_slow=26, 
        macd_signal=9, 
        ema_period=200,
        use_rsi=USE_RSI,
        use_adx=USE_ADX,
        use_bollinger=USE_BOLLINGER,
        use_ema200=USE_EMA200,
        use_ml_filter=USE_ML_FILTER,
        ml_confidence_threshold=ML_CONFIDENCE_THRESHOLD,
        ml_target_period=ML_TARGET_PERIOD
    )
    
    # Scan all stocks
    results_df = scanner.scan_all_stocks()
    
    if len(results_df) == 0:
        print("No fresh signals found in the S&P 500.")
        return
    
    # Display results
    print("\n" + "=" * 80)
    print("FRESH MACD SIGNALS")
    print("=" * 80)
    
    # Separate LONG and SHORT signals
    long_signals = results_df[results_df['signal'].str.contains('LONG')]
    short_signals = results_df[results_df['signal'].str.contains('SHORT')]
    
    if len(long_signals) > 0:
        print(f"\n📈 LONG SIGNALS ({len(long_signals)} stocks):")
        print("-" * 80)
        print(long_signals.to_string(index=False))
    
    if len(short_signals) > 0:
        print(f"\n📉 SHORT SIGNALS ({len(short_signals)} stocks):")
        print("-" * 80)
        print(short_signals.to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total signals found: {len(results_df)}")
    print(f"Strong Long: {len(results_df[results_df['signal'] == 'STRONG LONG'])}")
    print(f"Long: {len(results_df[results_df['signal'] == 'LONG'])}")
    print(f"Short: {len(results_df[results_df['signal'] == 'SHORT'])}")
    print(f"Strong Short: {len(results_df[results_df['signal'] == 'STRONG SHORT'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
