"""
Data Preparation for Machine Learning Models
Collects historical MACD signals and labels them for training
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from macd_scanner import MACDScanner
import warnings
import os
warnings.filterwarnings('ignore')


class DataPreparation:
    """
    Prepare training data by backtesting historical MACD signals
    Labels each signal with future returns (5, 10, 20 days)
    """
    
    def __init__(self, holding_periods=[5, 10, 20]):
        self.holding_periods = holding_periods
        self.scanner = MACDScanner(
            use_rsi=True,
            use_adx=True,
            use_bollinger=True,
            use_ema200=True
        )
    
    def extract_features_at_point(self, data, idx, crossover_type):
        """
        Extract ML features at a specific point in time
        Returns both snapshot features (for XGBoost) and sequences (for LSTM)
        """
        if idx < 20:  # Need at least 20 days of history
            return None, None
        
        # Calculate all indicators if not already present
        if 'MACD' not in data.columns:
            macd_line, signal_line, histogram = self.scanner.calculate_macd(data)
            data['MACD'] = macd_line
            data['Signal'] = signal_line
            data['Histogram'] = histogram
        
        if 'RSI' not in data.columns:
            data['RSI'] = self.scanner.calculate_rsi(data, period=14)
        
        if 'ADX' not in data.columns:
            adx, plus_di, minus_di = self.scanner.calculate_adx(data, period=14)
            data['ADX'] = adx
            data['Plus_DI'] = plus_di
            data['Minus_DI'] = minus_di
        
        if 'EMA_200' not in data.columns:
            data['EMA_200'] = self.scanner.calculate_ema(data['Close'], 200)
        
        if 'BB_Upper' not in data.columns:
            bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b = self.scanner.calculate_bollinger_bands(data)
            data['BB_Upper'] = bb_upper
            data['BB_Middle'] = bb_middle
            data['BB_Lower'] = bb_lower
            data['BB_Bandwidth'] = bb_bandwidth
            data['BB_PercentB'] = bb_percent_b
        
        # Now get window and latest after all indicators are calculated
        window = data.iloc[max(0, idx-60):idx+1]  # 60-day lookback
        latest = window.iloc[-1]
        
        # Snapshot features for XGBoost
        snapshot_features = {
            # MACD features
            'macd_value': latest['MACD'],
            'macd_signal': latest['Signal'],
            'macd_histogram': latest['Histogram'],
            'macd_above_zero': int(latest['MACD'] > 0),
            'histogram_slope': (window['Histogram'].iloc[-1] - window['Histogram'].iloc[-5]) / 5 if len(window) >= 5 else 0,
            'histogram_momentum': window['Histogram'].iloc[-5:].mean() if len(window) >= 5 else 0,
            
            # RSI features
            'rsi': latest['RSI'],
            'rsi_oversold': int(latest['RSI'] < 30),
            'rsi_overbought': int(latest['RSI'] > 70),
            'rsi_neutral': int(30 <= latest['RSI'] <= 70),
            'rsi_slope': (latest['RSI'] - window['RSI'].iloc[-5]) / 5 if len(window) >= 5 else 0,
            
            # ADX features
            'adx': latest['ADX'],
            'plus_di': latest['Plus_DI'],
            'minus_di': latest['Minus_DI'],
            'di_diff': latest['Plus_DI'] - latest['Minus_DI'],
            'adx_strong': int(latest['ADX'] > 25),
            
            # Price features
            'price': latest['Close'],
            'distance_from_ema200_pct': ((latest['Close'] - latest['EMA_200']) / latest['EMA_200'] * 100) if latest['EMA_200'] > 0 else 0,
            'price_above_ema200': int(latest['Close'] > latest['EMA_200']),
            'returns_1d': ((latest['Close'] - window.iloc[-2]['Close']) / window.iloc[-2]['Close'] * 100) if len(window) >= 2 else 0,
            'returns_5d': ((latest['Close'] - window.iloc[-6]['Close']) / window.iloc[-6]['Close'] * 100) if len(window) >= 6 else 0,
            'returns_10d': ((latest['Close'] - window.iloc[-11]['Close']) / window.iloc[-11]['Close'] * 100) if len(window) >= 11 else 0,
            'returns_20d': ((latest['Close'] - window.iloc[-21]['Close']) / window.iloc[-21]['Close'] * 100) if len(window) >= 21 else 0,
            
            # Volume features (ratios only - raw volume not comparable across stocks)
            'volume_ratio_20d': latest['Volume'] / window['Volume'].mean() if window['Volume'].mean() > 0 else 1,
            'volume_trend': int(window['Volume'].iloc[-5:].mean() > window['Volume'].iloc[-11:-5].mean()) if len(window) >= 11 else 0,
            
            # Bollinger Band features
            'bb_position': latest['BB_PercentB'],
            'bb_bandwidth': latest['BB_Bandwidth'],
            'bb_squeeze': int(latest['BB_Bandwidth'] < window['BB_Bandwidth'].quantile(0.2)) if len(window) >= 20 else 0,
            
            # Volatility
            'volatility_10d': window['Close'].pct_change().iloc[-10:].std() * 100 if len(window) >= 10 else 0,
            'volatility_20d': window['Close'].pct_change().iloc[-20:].std() * 100 if len(window) >= 20 else 0,
            
            # Crossover context
            'crossover_type': crossover_type,  # 1 = bullish, -1 = bearish
        }
        
        # Sequence features for LSTM (last 30 days of key indicators)
        sequence_length = min(30, len(window))
        sequence_data = window.iloc[-sequence_length:][['Close', 'Volume', 'MACD', 'Signal', 'RSI', 'ADX', 'BB_PercentB', 'EMA_200']].copy()
        
        # Fill any NaN values with forward fill, then backward fill, then 0
        sequence_data = sequence_data.ffill().bfill().fillna(0)
        
        # Normalize sequence data
        sequence_normalized = sequence_data.copy()
        for col in sequence_data.columns:
            std = sequence_data[col].std()
            if std > 1e-8:  # Use small threshold instead of 0 to avoid division issues
                sequence_normalized[col] = (sequence_data[col] - sequence_data[col].mean()) / std
            else:
                sequence_normalized[col] = 0
        
        # Final safety check - replace any remaining NaN or Inf with 0
        sequence_normalized = sequence_normalized.replace([np.inf, -np.inf], 0).fillna(0)
        
        return snapshot_features, sequence_normalized.values
    
    def label_signal_outcome(self, data, entry_idx, entry_price, crossover_type):
        """
        Label the signal with forward returns and profitability
        Returns dict with outcomes for different holding periods
        """
        outcomes = {}
        
        for period in self.holding_periods:
            exit_idx = entry_idx + period
            
            if exit_idx >= len(data):
                # Not enough future data
                outcomes[f'return_{period}d'] = None
                outcomes[f'profitable_{period}d'] = None
                outcomes[f'max_return_{period}d'] = None
                outcomes[f'max_drawdown_{period}d'] = None
                continue
            
            exit_price = data.iloc[exit_idx]['Close']
            
            # Calculate returns based on signal direction
            if crossover_type == 1:  # Bullish - Long position
                return_pct = (exit_price - entry_price) / entry_price * 100
            else:  # Bearish - Short position
                return_pct = (entry_price - exit_price) / entry_price * 100
            
            # Calculate max favorable move and max adverse move during holding period
            period_data = data.iloc[entry_idx:exit_idx+1]
            
            if crossover_type == 1:  # Long
                max_return = ((period_data['High'].max() - entry_price) / entry_price * 100)
                max_drawdown = ((period_data['Low'].min() - entry_price) / entry_price * 100)
            else:  # Short
                max_return = ((entry_price - period_data['Low'].min()) / entry_price * 100)
                max_drawdown = ((entry_price - period_data['High'].max()) / entry_price * 100)
            
            # Label as profitable if return > 0.5% (accounting for transaction costs)
            profitable = 1 if return_pct > 0.5 else 0
            
            outcomes[f'return_{period}d'] = round(return_pct, 2)
            outcomes[f'profitable_{period}d'] = profitable
            outcomes[f'max_return_{period}d'] = round(max_return, 2)
            outcomes[f'max_drawdown_{period}d'] = round(max_drawdown, 2)
        
        return outcomes
    
    def collect_training_data(self, tickers, lookback_period='3y', save_path='training_data.csv'):
        """
        Collect training data from multiple tickers
        
        Args:
            tickers: List of stock tickers
            lookback_period: How far back to collect data
            save_path: Where to save the dataset
        """
        print(f"🔍 Collecting training data from {len(tickers)} tickers...")
        print(f"📊 Lookback period: {lookback_period}")
        print(f"📈 Holding periods: {self.holding_periods} days\n")
        
        all_data = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
            try:
                print(f"[{i+1}/{len(tickers)}] Processing {ticker}...", end=" ")
                
                # Download historical data
                stock = yf.Ticker(ticker)
                data = stock.history(period=lookback_period)
                
                if len(data) < 200:
                    print("❌ Insufficient data")
                    failed_tickers.append(ticker)
                    continue
                
                # Calculate indicators
                macd_line, signal_line, histogram = self.scanner.calculate_macd(data)
                data['MACD'] = macd_line
                data['Signal'] = signal_line
                data['Histogram'] = histogram
                
                # Detect all crossovers
                crossover = self.scanner.detect_crossover(macd_line, signal_line)
                data['Crossover'] = crossover
                
                # Find all crossover points
                crossover_indices = data[data['Crossover'] != 0].index
                
                if len(crossover_indices) == 0:
                    print("❌ No crossovers found")
                    continue
                
                signals_found = 0
                
                for cross_date in crossover_indices:
                    idx = data.index.get_loc(cross_date)
                    crossover_type = data.loc[cross_date, 'Crossover']
                    entry_price = data.loc[cross_date, 'Close']
                    
                    # Extract features
                    snapshot_features, sequence_features = self.extract_features_at_point(data, idx, crossover_type)
                    
                    if snapshot_features is None:
                        continue
                    
                    # Label outcomes
                    outcomes = self.label_signal_outcome(data, idx, entry_price, crossover_type)
                    
                    # Skip if not enough future data
                    if outcomes[f'return_{self.holding_periods[0]}d'] is None:
                        continue
                    
                    # Combine all features and labels
                    record = {
                        'ticker': ticker,
                        'date': cross_date.strftime('%Y-%m-%d'),
                        'entry_price': entry_price,
                        **snapshot_features,
                        **outcomes
                    }
                    
                    # Store sequence separately (will save to separate file)
                    record['sequence'] = sequence_features.tolist()
                    
                    all_data.append(record)
                    signals_found += 1
                
                print(f"✅ Found {signals_found} signals")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                failed_tickers.append(ticker)
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        if len(df) == 0:
            print("\n❌ No data collected. Check tickers and date ranges.")
            return None
        
        print(f"\n{'='*60}")
        print(f"✅ Data collection complete!")
        print(f"{'='*60}")
        print(f"Total signals collected: {len(df)}")
        print(f"Tickers processed: {len(tickers) - len(failed_tickers)}/{len(tickers)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nSignal distribution:")
        print(f"  Bullish (Long): {len(df[df['crossover_type'] == 1])}")
        print(f"  Bearish (Short): {len(df[df['crossover_type'] == -1])}")
        
        # Show profitability statistics
        for period in self.holding_periods:
            profitable_col = f'profitable_{period}d'
            if profitable_col in df.columns:
                win_rate = df[profitable_col].mean() * 100
                avg_return = df[f'return_{period}d'].mean()
                print(f"\n{period}-day holding period:")
                print(f"  Win rate: {win_rate:.1f}%")
                print(f"  Avg return: {avg_return:.2f}%")
        
        # Save to CSV
        # Separate sequences from main dataframe
        sequences = df['sequence'].tolist()
        df_to_save = df.drop(columns=['sequence'])
        
        df_to_save.to_csv(save_path, index=False)
        print(f"\n💾 Snapshot features saved to: {save_path}")
        
        # Save sequences separately (use allow_pickle for variable-length sequences)
        sequence_path = save_path.replace('.csv', '_sequences.npy')
        np.save(sequence_path, np.array(sequences, dtype=object), allow_pickle=True)
        print(f"💾 Sequence features saved to: {sequence_path}")
        
        if failed_tickers:
            print(f"\n⚠️  Failed tickers ({len(failed_tickers)}): {', '.join(failed_tickers[:10])}")
        
        return df_to_save
    
    def load_sp500_tickers(self):
        """Load S&P 500 tickers from Wikipedia"""
        try:
            import requests
            from io import StringIO
            
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            
            tables = pd.read_html(StringIO(response.text))
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            
            # Clean tickers (remove any special characters)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            return tickers
        except Exception as e:
            print(f"Error loading S&P 500 tickers: {e}")
            # Fallback to a smaller list
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'V', 'JNJ']


if __name__ == "__main__":
    # Example usage
    prep = DataPreparation(holding_periods=[5, 10, 20])
    
    # Option 1: Use full S&P 500 (takes ~2-3 hours)
    # tickers = prep.load_sp500_tickers()
    
    # Option 2: Use subset for testing (takes ~5-10 minutes)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'MA',
               'WMT', 'PG', 'JNJ', 'UNH', 'HD', 'BAC', 'XOM', 'CVX', 'ABBV', 'PFE',
               'KO', 'PEP', 'COST', 'AVGO', 'CSCO', 'ADBE', 'NFLX', 'CRM', 'INTC', 'AMD']
    
    print("="*60)
    print("MACD Machine Learning - Data Collection")
    print("="*60)
    print("\nThis will collect historical MACD signals and label them")
    print("for machine learning training.\n")
    
    # Collect data
    df = prep.collect_training_data(tickers, lookback_period='3y', save_path='training_data.csv')
    
    if df is not None:
        print(f"\n✅ Ready for model training!")
        print(f"Run: python train_models.py")
