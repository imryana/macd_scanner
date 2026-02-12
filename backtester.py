"""
MACD Crossover Backtester
Simulates historical trades based on MACD crossover signals with fixed risk-reward ratio.
Entry on crossover, exit on stop loss or take profit hit (intraday high/low checked).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MACDBacktester:
    """
    Backtest MACD crossover signals on a single stock.
    
    Uses the same signal logic as MACDScanner:
    - Stop loss: configurable % below/above entry
    - Take profit: stop_loss_pct * risk_reward_ratio
    - Checks intraday High/Low to determine if SL or TP was hit each day
    - If neither hit within max_hold_days, exits at close on the last day
    """

    def __init__(self, macd_fast=12, macd_slow=26, macd_signal=9,
                 stop_loss_pct=0.05, risk_reward_ratio=1.5, max_hold_days=20):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stop_loss_pct = stop_loss_pct
        self.risk_reward_ratio = risk_reward_ratio
        self.max_hold_days = max_hold_days

    # ── Indicator calculations ──────────────────────────────────────────

    @staticmethod
    def _ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    def _add_indicators(self, df):
        """Add MACD, EMA-200, RSI to dataframe (in-place)."""
        df['EMA_fast'] = self._ema(df['Close'], self.macd_fast)
        df['EMA_slow'] = self._ema(df['Close'], self.macd_slow)
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['Signal'] = self._ema(df['MACD'], self.macd_signal)
        df['Histogram'] = df['MACD'] - df['Signal']
        df['EMA_200'] = self._ema(df['Close'], 200)

        # RSI-14
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        # ADX (14-period)
        high, low, close = df['High'], df['Low'], df['Close']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = pd.concat([high - low,
                        (high - close.shift()).abs(),
                        (low - close.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean()
        df['Plus_DI'] = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14
        df['Minus_DI'] = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14
        dx = 100 * (df['Plus_DI'] - df['Minus_DI']).abs() / (df['Plus_DI'] + df['Minus_DI']).replace(0, np.nan)
        df['ADX'] = dx.ewm(span=14, adjust=False).mean()

        # Bollinger Bands (20, 2)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper_bb = sma20 + 2 * std20
        lower_bb = sma20 - 2 * std20
        df['BB_PercentB'] = (close - lower_bb) / (upper_bb - lower_bb)
        df['BB_Bandwidth'] = (upper_bb - lower_bb) / sma20

        return df

    def _detect_crossovers(self, df):
        """Return Series: +1 bullish, -1 bearish, 0 none."""
        cross = pd.Series(0, index=df.index)
        bullish = (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))
        bearish = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))
        cross[bullish] = 1
        cross[bearish] = -1
        return cross

    # ── ML confidence grading ───────────────────────────────────────────

    GRADE_ORDER = {'F': 0, 'D': 1, 'C': 2, 'B': 3, 'A': 4, 'A+': 5}

    @staticmethod
    def _grade_meets_minimum(grade, min_grade):
        """Check if a signal grade meets the minimum threshold."""
        order = MACDBacktester.GRADE_ORDER
        return order.get(grade, 0) >= order.get(min_grade, 0)

    def _extract_ml_features(self, data, idx, crossover_type):
        """Extract ML features at a crossover index for confidence grading."""
        if idx < 20:
            return None, None

        window = data.iloc[max(0, idx - 60):idx + 1]
        latest = window.iloc[-1]

        snapshot_features = {
            'macd_value': latest['MACD'],
            'macd_signal': latest['Signal'],
            'macd_histogram': latest['Histogram'],
            'macd_above_zero': int(latest['MACD'] > 0),
            'histogram_slope': (window['Histogram'].iloc[-1] - window['Histogram'].iloc[-5]) / 5 if len(window) >= 5 else 0,
            'histogram_momentum': window['Histogram'].iloc[-5:].mean() if len(window) >= 5 else 0,
            'rsi': latest.get('RSI', 50) if not np.isnan(latest.get('RSI', 50)) else 50,
            'rsi_oversold': int(latest.get('RSI', 50) < 30),
            'rsi_overbought': int(latest.get('RSI', 50) > 70),
            'rsi_neutral': int(30 <= latest.get('RSI', 50) <= 70),
            'rsi_slope': (latest.get('RSI', 50) - window['RSI'].iloc[-5]) / 5 if len(window) >= 5 and 'RSI' in window.columns else 0,
            'adx': latest.get('ADX', 20) if not np.isnan(latest.get('ADX', 20)) else 20,
            'plus_di': latest.get('Plus_DI', 20) if not np.isnan(latest.get('Plus_DI', 20)) else 20,
            'minus_di': latest.get('Minus_DI', 20) if not np.isnan(latest.get('Minus_DI', 20)) else 20,
            'di_diff': (latest.get('Plus_DI', 20) - latest.get('Minus_DI', 20)) if not np.isnan(latest.get('Plus_DI', 20)) else 0,
            'adx_strong': int(latest.get('ADX', 20) > 25) if not np.isnan(latest.get('ADX', 20)) else 0,
            'price': latest['Close'],
            'distance_from_ema200_pct': ((latest['Close'] - latest['EMA_200']) / latest['EMA_200'] * 100) if latest['EMA_200'] > 0 else 0,
            'price_above_ema200': int(latest['Close'] > latest['EMA_200']),
            'returns_1d': ((latest['Close'] - window.iloc[-2]['Close']) / window.iloc[-2]['Close'] * 100) if len(window) >= 2 else 0,
            'returns_5d': ((latest['Close'] - window.iloc[-6]['Close']) / window.iloc[-6]['Close'] * 100) if len(window) >= 6 else 0,
            'returns_10d': ((latest['Close'] - window.iloc[-11]['Close']) / window.iloc[-11]['Close'] * 100) if len(window) >= 11 else 0,
            'returns_20d': ((latest['Close'] - window.iloc[-21]['Close']) / window.iloc[-21]['Close'] * 100) if len(window) >= 21 else 0,
            'volume_ratio_20d': latest['Volume'] / window['Volume'].mean() if window['Volume'].mean() > 0 else 1,
            'volume_trend': int(window['Volume'].iloc[-5:].mean() > window['Volume'].iloc[-11:-5].mean()) if len(window) >= 11 else 0,
            'bb_position': latest.get('BB_PercentB', 0.5) if not np.isnan(latest.get('BB_PercentB', 0.5)) else 0.5,
            'bb_bandwidth': latest.get('BB_Bandwidth', 0.04) if not np.isnan(latest.get('BB_Bandwidth', 0.04)) else 0.04,
            'bb_squeeze': int(latest.get('BB_Bandwidth', 0.04) < window['BB_Bandwidth'].quantile(0.2)) if len(window) >= 20 and 'BB_Bandwidth' in window.columns else 0,
            'volatility_10d': window['Close'].pct_change().iloc[-10:].std() * 100 if len(window) >= 10 else 0,
            'volatility_20d': window['Close'].pct_change().iloc[-20:].std() * 100 if len(window) >= 20 else 0,
            'crossover_type': crossover_type,
        }

        # Sequence features for LSTM (last 30 days)
        seq_len = min(30, len(window))
        seq_df = window.iloc[-seq_len:][['Close', 'Volume', 'MACD', 'Signal', 'RSI', 'ADX', 'BB_PercentB', 'EMA_200']].copy()
        seq_df = seq_df.fillna(method='ffill').fillna(0)
        seq_norm = seq_df.copy()
        for col in seq_df.columns:
            if seq_df[col].std() > 0:
                seq_norm[col] = (seq_df[col] - seq_df[col].mean()) / seq_df[col].std()
            else:
                seq_norm[col] = 0
        return snapshot_features, seq_norm.values

    # ── Core backtest ───────────────────────────────────────────────────

    def _check_signal_quality(self, data, i, direction):
        """
        Check if indicator conditions confirm the crossover signal.
        Mirrors MACDScanner.generate_signal() logic.

        Returns signal string ('STRONG LONG', 'LONG', 'STRONG SHORT', 'SHORT')
        or None if conditions are not met.
        """
        row = data.iloc[i]
        prev_hist = data.iloc[i - 1]['Histogram'] if i > 0 else 0

        macd_above_zero = row['MACD'] > 0
        crossover_above_zero = row['MACD'] > 0  # Crossover MACD value
        histogram_increasing = row['Histogram'] > prev_hist

        # EMA-200
        price_above_ema200 = True
        if not np.isnan(row.get('EMA_200', np.nan)):
            price_above_ema200 = row['Close'] > row['EMA_200']

        # RSI
        rsi = row.get('RSI', 50)
        if np.isnan(rsi):
            rsi = 50
        rsi_bullish = 30 < rsi < 70
        rsi_oversold = rsi < 35
        rsi_overbought = rsi > 65

        # ADX
        adx = row.get('ADX', 25)
        if np.isnan(adx):
            adx = 25
        strong_trend = adx > 25
        moderate_trend = adx > 20

        # Bollinger Bands
        bb_position = row.get('BB_PercentB', 0.5)
        if np.isnan(bb_position):
            bb_position = 0.5
        bb_oversold = bb_position < 0.2
        bb_overbought = bb_position > 0.8
        bb_neutral = 0.3 < bb_position < 0.7

        if direction == 'LONG':
            if (crossover_above_zero and price_above_ema200 and
                    strong_trend and rsi_bullish and not bb_overbought):
                return 'STRONG LONG'
            elif (crossover_above_zero and price_above_ema200 and
                  (rsi_oversold or bb_oversold) and moderate_trend):
                return 'STRONG LONG'
            elif (crossover_above_zero and price_above_ema200 and
                  rsi < 70 and moderate_trend):
                return 'LONG'
            elif (not crossover_above_zero and price_above_ema200 and
                  strong_trend and rsi_oversold and bb_oversold):
                return 'LONG'
            elif (price_above_ema200 and rsi_bullish and
                  (moderate_trend or histogram_increasing) and bb_neutral):
                return 'LONG'
            return None

        elif direction == 'SHORT':
            if (not crossover_above_zero and not price_above_ema200 and
                    strong_trend and rsi_bullish and not bb_oversold):
                return 'STRONG SHORT'
            elif (not crossover_above_zero and not price_above_ema200 and
                  (rsi_overbought or bb_overbought) and moderate_trend):
                return 'STRONG SHORT'
            elif (not crossover_above_zero and not price_above_ema200 and
                  rsi > 30 and moderate_trend):
                return 'SHORT'
            elif (crossover_above_zero and not price_above_ema200 and
                  strong_trend and rsi_overbought and bb_overbought):
                return 'SHORT'
            elif (not price_above_ema200 and rsi_bullish and
                  (moderate_trend or not histogram_increasing) and bb_neutral):
                return 'SHORT'
            return None

        return None

    def run(self, ticker, start_date=None, end_date=None, period='5y', 
            trade_type='both', require_ema200=False,
            ml_predictor=None, min_ml_grade=None, entry_delay=0,
            require_confirmation=False):
        """
        Run backtest on a single ticker.

        Args:
            ticker: Stock symbol (e.g. 'AAPL')
            start_date: Start date string 'YYYY-MM-DD' (optional, overrides period)
            end_date: End date string 'YYYY-MM-DD' (optional)
            period: yfinance period string if start_date not given
            trade_type: 'long', 'short', or 'both'
            require_ema200: If True, only take longs above EMA-200 / shorts below
            ml_predictor: Optional EnsemblePredictor instance for ML confidence filtering
            min_ml_grade: Minimum ML grade to enter trade ('D', 'C', 'B', 'A', 'A+')
            entry_delay: Days to wait after crossover before entering (0 = enter same day)
            require_confirmation: If True, require RSI/ADX/BB/EMA-200 confirmation
            min_ml_grade: Minimum ML grade to enter trade ('D', 'C', 'B', 'A', 'A+')
            entry_delay: Days to wait after crossover before entering (0 = enter same day)

        Returns:
            dict with trades list, summary stats, and equity curve
        """
        # Download data
        stock = yf.Ticker(ticker)
        if start_date:
            data = stock.history(start=start_date, end=end_date)
        else:
            data = stock.history(period=period)

        if len(data) < 50:
            return {'error': f'Not enough data for {ticker} ({len(data)} bars, need 50+)'}

        data = self._add_indicators(data)
        data['Crossover'] = self._detect_crossovers(data)

        trades = []
        i = 0
        indices = data.index.tolist()

        while i < len(data):
            row = data.iloc[i]
            cross = row['Crossover']

            if cross == 0:
                i += 1
                continue

            # Direction filter
            direction = 'LONG' if cross == 1 else 'SHORT'
            if trade_type == 'long' and direction == 'SHORT':
                i += 1
                continue
            if trade_type == 'short' and direction == 'LONG':
                i += 1
                continue

            # Multi-indicator confirmation filter
            signal_quality = None
            if require_confirmation:
                signal_quality = self._check_signal_quality(data, i, direction)
                if signal_quality is None:
                    i += 1
                    continue

            # EMA-200 filter
            if require_ema200:
                if direction == 'LONG' and row['Close'] < row['EMA_200']:
                    i += 1
                    continue
                if direction == 'SHORT' and row['Close'] > row['EMA_200']:
                    i += 1
                    continue

            # ML confidence filter
            ml_grade = None
            if ml_predictor is not None and min_ml_grade is not None:
                try:
                    crossover_type = 1 if direction == 'LONG' else -1
                    features, seq = self._extract_ml_features(data, i, crossover_type)
                    if features is not None and seq is not None:
                        ml_result = ml_predictor.predict(features, seq)
                        ml_grade = ml_result['signal_grade']
                        if not self._grade_meets_minimum(ml_grade, min_ml_grade):
                            i += 1
                            continue
                except Exception:
                    pass  # Allow trade if ML scoring fails

            # Entry delay
            actual_entry_idx = i + entry_delay
            if actual_entry_idx >= len(data):
                i += 1
                continue

            # ── Open trade ──
            entry_row = data.iloc[actual_entry_idx]
            entry_price = entry_row['Close']
            entry_date = indices[actual_entry_idx]

            if direction == 'LONG':
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                take_profit = entry_price * (1 + self.stop_loss_pct * self.risk_reward_ratio)
            else:  # SHORT
                stop_loss = entry_price * (1 + self.stop_loss_pct)
                take_profit = entry_price * (1 - self.stop_loss_pct * self.risk_reward_ratio)

            # ── Simulate day-by-day ──
            exit_price = None
            exit_date = None
            exit_reason = None

            for j in range(actual_entry_idx + 1, min(actual_entry_idx + 1 + self.max_hold_days, len(data))):
                day = data.iloc[j]

                if direction == 'LONG':
                    # Check stop loss first (conservative: assume worst case hit first)
                    if day['Low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_date = indices[j]
                        exit_reason = 'Stop Loss'
                        break
                    # Check take profit
                    if day['High'] >= take_profit:
                        exit_price = take_profit
                        exit_date = indices[j]
                        exit_reason = 'Take Profit'
                        break
                else:  # SHORT
                    if day['High'] >= stop_loss:
                        exit_price = stop_loss
                        exit_date = indices[j]
                        exit_reason = 'Stop Loss'
                        break
                    if day['Low'] <= take_profit:
                        exit_price = take_profit
                        exit_date = indices[j]
                        exit_reason = 'Take Profit'
                        break

            # If neither SL nor TP hit within max_hold_days, exit at close
            if exit_price is None:
                last_j = min(actual_entry_idx + self.max_hold_days, len(data) - 1)
                exit_price = data.iloc[last_j]['Close']
                exit_date = indices[last_j]
                exit_reason = 'Max Hold'

            # Calculate P&L
            if direction == 'LONG':
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100

            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'direction': direction,
                'entry_price': round(entry_price, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'pnl_pct': round(pnl_pct, 2),
                'hold_days': (exit_date - entry_date).days,
                'macd_at_entry': round(row['MACD'], 4),
                'rsi_at_entry': round(row['RSI'], 2) if not np.isnan(row['RSI']) else None,
                'ml_grade': ml_grade,
                'signal_quality': signal_quality,
                'entry_delay_days': entry_delay,
            })

            # Jump past trade (no overlapping trades)
            exit_idx = indices.index(exit_date)
            i = exit_idx + 1

        # ── Build summary ───────────────────────────────────────────────
        if not trades:
            return {
                'ticker': ticker,
                'trades': [],
                'summary': self._empty_summary(ticker),
                'equity_curve': pd.DataFrame()
            }

        trades_df = pd.DataFrame(trades)
        summary = self._compute_summary(trades_df, ticker, data)
        equity = self._build_equity_curve(trades_df)

        return {
            'ticker': ticker,
            'trades': trades_df,
            'summary': summary,
            'equity_curve': equity
        }

    # ── Summary statistics ──────────────────────────────────────────────

    @staticmethod
    def _buy_and_hold_return(price_data):
        """Calculate buy-and-hold return over the entire price dataset."""
        if len(price_data) < 2:
            return 0.0
        start_price = price_data['Close'].iloc[0]
        end_price = price_data['Close'].iloc[-1]
        return round((end_price - start_price) / start_price * 100, 2)

    def _empty_summary(self, ticker):
        return {
            'ticker': ticker,
            'total_trades': 0,
            'win_rate': 0,
            'avg_pnl': 0,
            'total_return': 0,
            'max_win': 0,
            'max_loss': 0,
            'profit_factor': 0,
            'avg_hold_days': 0,
            'tp_hits': 0,
            'sl_hits': 0,
            'max_hold_exits': 0,
            'long_trades': 0,
            'short_trades': 0,
            'long_win_rate': 0,
            'short_win_rate': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'expectancy': 0,
            'buy_hold_return': 0,
            'alpha': 0,
        }

    def _compute_summary(self, df, ticker, price_data):
        wins = df[df['pnl_pct'] > 0]
        losses = df[df['pnl_pct'] <= 0]
        longs = df[df['direction'] == 'LONG']
        shorts = df[df['direction'] == 'SHORT']

        total_trades = len(df)
        win_rate = len(wins) / total_trades * 100 if total_trades else 0
        avg_pnl = df['pnl_pct'].mean()
        total_return = df['pnl_pct'].sum()

        gross_profit = wins['pnl_pct'].sum() if len(wins) else 0
        gross_loss = abs(losses['pnl_pct'].sum()) if len(losses) else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = wins['pnl_pct'].mean() if len(wins) else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) else 0

        # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        win_r = len(wins) / total_trades if total_trades else 0
        loss_r = len(losses) / total_trades if total_trades else 0
        expectancy = (win_r * avg_win) + (loss_r * avg_loss)

        # Consecutive wins/losses
        is_win = (df['pnl_pct'] > 0).astype(int)
        max_consec_wins = self._max_consecutive(is_win, 1)
        max_consec_losses = self._max_consecutive(is_win, 0)

        long_wins = longs[longs['pnl_pct'] > 0]
        short_wins = shorts[shorts['pnl_pct'] > 0]

        return {
            'ticker': ticker,
            'total_trades': total_trades,
            'win_rate': round(win_rate, 1),
            'avg_pnl': round(avg_pnl, 2),
            'total_return': round(total_return, 2),
            'max_win': round(df['pnl_pct'].max(), 2),
            'max_loss': round(df['pnl_pct'].min(), 2),
            'profit_factor': round(profit_factor, 2),
            'avg_hold_days': round(df['hold_days'].mean(), 1),
            'tp_hits': int((df['exit_reason'] == 'Take Profit').sum()),
            'sl_hits': int((df['exit_reason'] == 'Stop Loss').sum()),
            'max_hold_exits': int((df['exit_reason'] == 'Max Hold').sum()),
            'long_trades': len(longs),
            'short_trades': len(shorts),
            'long_win_rate': round(len(long_wins) / len(longs) * 100, 1) if len(longs) else 0,
            'short_win_rate': round(len(short_wins) / len(shorts) * 100, 1) if len(shorts) else 0,
            'max_consecutive_wins': max_consec_wins,
            'max_consecutive_losses': max_consec_losses,
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 2),
            'buy_hold_return': self._buy_and_hold_return(price_data),
            'alpha': round(total_return - self._buy_and_hold_return(price_data), 2),
        }

    @staticmethod
    def _max_consecutive(series, value):
        groups = (series != value).cumsum()
        filtered = series[series == value]
        if len(filtered) == 0:
            return 0
        return filtered.groupby(groups).count().max()

    def _build_equity_curve(self, trades_df):
        """Build cumulative return curve from sequential trades."""
        equity = [0.0]
        for _, trade in trades_df.iterrows():
            equity.append(equity[-1] + trade['pnl_pct'])
        return pd.DataFrame({
            'trade_num': range(len(equity)),
            'cumulative_return': equity
        })

    # ════════════════════════════════════════════════════════════════════
    # Ryan's Potential Trading Behaviour Back Test
    # ════════════════════════════════════════════════════════════════════

    def run_realistic(self, ticker, start_date=None, end_date=None, period='5y',
                      trade_type='both', require_ema200=False,
                      capture_pct=0.65,
                      ml_predictor=None, min_ml_grade=None, entry_delay=0,
                      require_confirmation=False):
        """
        Realistic trader simulation over a 5-day max hold.

        Exit logic (checked each day in order):
          1. Stop Loss hit (intraday Low/High touches SL) → exit at SL price
          2. Take Profit hit (intraday High/Low touches TP) → exit at TP price
          3. After 5 days, if neither SL nor TP was hit:
             - If the trade moved favourably at some point (max favorable excursion > 0),
               the trader realistically captures ~capture_pct (default 65%) of the
               best intraday price reached. A real trader can't pick the exact peak,
               so this models selling "near" the high but not at it.
             - If the trade never went green, exit at day-5 close (small loss / breakeven).

        Args:
            capture_pct: Fraction of max favorable excursion the trader captures (0.0-1.0).
                         0.65 means the trader captures 65% of the best move.
            ml_predictor: Optional EnsemblePredictor instance for ML confidence filtering
            min_ml_grade: Minimum ML grade to enter trade ('D', 'C', 'B', 'A', 'A+')
            entry_delay: Days to wait after crossover before entering (0 = enter same day)
            require_confirmation: If True, require RSI/ADX/BB/EMA-200 confirmation
        """
        stock = yf.Ticker(ticker)
        if start_date:
            data = stock.history(start=start_date, end=end_date)
        else:
            data = stock.history(period=period)

        if len(data) < 50:
            return {'error': f'Not enough data for {ticker} ({len(data)} bars, need 50+)'}

        data = self._add_indicators(data)
        data['Crossover'] = self._detect_crossovers(data)

        trades = []
        i = 0
        indices = data.index.tolist()
        max_hold = 5  # Fixed 5-day hold for this mode

        while i < len(data):
            row = data.iloc[i]
            cross = row['Crossover']

            if cross == 0:
                i += 1
                continue

            direction = 'LONG' if cross == 1 else 'SHORT'
            if trade_type == 'long' and direction == 'SHORT':
                i += 1
                continue
            if trade_type == 'short' and direction == 'LONG':
                i += 1
                continue

            # Multi-indicator confirmation filter
            signal_quality = None
            if require_confirmation:
                signal_quality = self._check_signal_quality(data, i, direction)
                if signal_quality is None:
                    i += 1
                    continue

            if require_ema200:
                if direction == 'LONG' and row['Close'] < row['EMA_200']:
                    i += 1
                    continue
                if direction == 'SHORT' and row['Close'] > row['EMA_200']:
                    i += 1
                    continue

            # ML confidence filter
            ml_grade = None
            if ml_predictor is not None and min_ml_grade is not None:
                try:
                    crossover_type = 1 if direction == 'LONG' else -1
                    features, seq = self._extract_ml_features(data, i, crossover_type)
                    if features is not None and seq is not None:
                        ml_result = ml_predictor.predict(features, seq)
                        ml_grade = ml_result['signal_grade']
                        if not self._grade_meets_minimum(ml_grade, min_ml_grade):
                            i += 1
                            continue
                except Exception:
                    pass  # Allow trade if ML scoring fails

            # Entry delay
            actual_entry_idx = i + entry_delay
            if actual_entry_idx >= len(data):
                i += 1
                continue

            entry_row = data.iloc[actual_entry_idx]
            entry_price = entry_row['Close']
            entry_date = indices[actual_entry_idx]

            if direction == 'LONG':
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                take_profit = entry_price * (1 + self.stop_loss_pct * self.risk_reward_ratio)
            else:
                stop_loss = entry_price * (1 + self.stop_loss_pct)
                take_profit = entry_price * (1 - self.stop_loss_pct * self.risk_reward_ratio)

            # ── Day-by-day simulation ──
            exit_price = None
            exit_date = None
            exit_reason = None
            best_favorable_price = entry_price  # track MFE

            end_j = min(actual_entry_idx + 1 + max_hold, len(data))

            for j in range(actual_entry_idx + 1, end_j):
                day = data.iloc[j]

                if direction == 'LONG':
                    # Track best high seen so far
                    if day['High'] > best_favorable_price:
                        best_favorable_price = day['High']

                    # SL check first (conservative)
                    if day['Low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_date = indices[j]
                        exit_reason = 'Stop Loss'
                        break
                    # TP check
                    if day['High'] >= take_profit:
                        exit_price = take_profit
                        exit_date = indices[j]
                        exit_reason = 'Take Profit'
                        break
                else:  # SHORT
                    # Track best low seen so far
                    if day['Low'] < best_favorable_price:
                        best_favorable_price = day['Low']

                    if day['High'] >= stop_loss:
                        exit_price = stop_loss
                        exit_date = indices[j]
                        exit_reason = 'Stop Loss'
                        break
                    if day['Low'] <= take_profit:
                        exit_price = take_profit
                        exit_date = indices[j]
                        exit_reason = 'Take Profit'
                        break

            # ── End of hold period – realistic exit ──
            if exit_price is None:
                last_j = min(actual_entry_idx + max_hold, len(data) - 1)
                exit_date = indices[last_j]

                if direction == 'LONG':
                    max_move = best_favorable_price - entry_price
                    if max_move > 0:
                        # Trader captures capture_pct of the best high
                        exit_price = entry_price + (max_move * capture_pct)
                        exit_reason = 'Realistic TP'
                    else:
                        exit_price = data.iloc[last_j]['Close']
                        exit_reason = 'Expired'
                else:  # SHORT
                    max_move = entry_price - best_favorable_price
                    if max_move > 0:
                        exit_price = entry_price - (max_move * capture_pct)
                        exit_reason = 'Realistic TP'
                    else:
                        exit_price = data.iloc[last_j]['Close']
                        exit_reason = 'Expired'

            # P&L
            if direction == 'LONG':
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100

            # Max favorable excursion % (for reporting)
            if direction == 'LONG':
                mfe_pct = (best_favorable_price - entry_price) / entry_price * 100
            else:
                mfe_pct = (entry_price - best_favorable_price) / entry_price * 100

            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'direction': direction,
                'entry_price': round(entry_price, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'pnl_pct': round(pnl_pct, 2),
                'mfe_pct': round(mfe_pct, 2),
                'hold_days': (exit_date - entry_date).days,
                'rsi_at_entry': round(row['RSI'], 2) if not np.isnan(row['RSI']) else None,
                'ml_grade': ml_grade,
                'signal_quality': signal_quality,
            })

            exit_idx = indices.index(exit_date)
            i = exit_idx + 1

        # ── Summary ──
        if not trades:
            summary = self._empty_summary(ticker)
            summary['realistic_tp_exits'] = 0
            summary['expired_exits'] = 0
            summary['avg_mfe'] = 0
            return {
                'ticker': ticker,
                'trades': [],
                'summary': summary,
                'equity_curve': pd.DataFrame()
            }

        trades_df = pd.DataFrame(trades)
        summary = self._compute_summary(trades_df, ticker, data)
        # Replace max_hold_exits with the two new categories
        summary['realistic_tp_exits'] = int((trades_df['exit_reason'] == 'Realistic TP').sum())
        summary['expired_exits'] = int((trades_df['exit_reason'] == 'Expired').sum())
        summary['avg_mfe'] = round(trades_df['mfe_pct'].mean(), 2)
        # Recalculate max_hold_exits to 0 (not used in this mode)
        summary['max_hold_exits'] = 0
        equity = self._build_equity_curve(trades_df)

        return {
            'ticker': ticker,
            'trades': trades_df,
            'summary': summary,
            'equity_curve': equity
        }


if __name__ == '__main__':
    bt = MACDBacktester(stop_loss_pct=0.05, risk_reward_ratio=1.5, max_hold_days=20)
    result = bt.run('AAPL', period='5y', trade_type='both')

    if 'error' in result:
        print(result['error'])
    else:
        s = result['summary']
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS: {s['ticker']}")
        print(f"{'='*60}")
        print(f"Total Trades:    {s['total_trades']}")
        print(f"Win Rate:        {s['win_rate']}%")
        print(f"Avg P&L/Trade:   {s['avg_pnl']}%")
        print(f"Total Return:    {s['total_return']}%")
        print(f"Profit Factor:   {s['profit_factor']}")
        print(f"Avg Hold Days:   {s['avg_hold_days']}")
        print(f"TP Hits:         {s['tp_hits']}  |  SL Hits: {s['sl_hits']}  |  Max Hold: {s['max_hold_exits']}")
        print(f"Expectancy:      {s['expectancy']}%")
        print(f"\nTrades:")
        print(result['trades'].to_string(index=False))
