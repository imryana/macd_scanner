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

        return df

    def _detect_crossovers(self, df):
        """Return Series: +1 bullish, -1 bearish, 0 none."""
        cross = pd.Series(0, index=df.index)
        bullish = (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))
        bearish = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))
        cross[bullish] = 1
        cross[bearish] = -1
        return cross

    # ── Core backtest ───────────────────────────────────────────────────

    def run(self, ticker, start_date=None, end_date=None, period='5y', 
            trade_type='both', require_ema200=False):
        """
        Run backtest on a single ticker.

        Args:
            ticker: Stock symbol (e.g. 'AAPL')
            start_date: Start date string 'YYYY-MM-DD' (optional, overrides period)
            end_date: End date string 'YYYY-MM-DD' (optional)
            period: yfinance period string if start_date not given
            trade_type: 'long', 'short', or 'both'
            require_ema200: If True, only take longs above EMA-200 / shorts below

        Returns:
            dict with trades list, summary stats, and equity curve
        """
        # Download data
        stock = yf.Ticker(ticker)
        if start_date:
            data = stock.history(start=start_date, end=end_date)
        else:
            data = stock.history(period=period)

        if len(data) < 200:
            return {'error': f'Not enough data for {ticker} ({len(data)} bars, need 200+)'}

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

            # EMA-200 filter
            if require_ema200:
                if direction == 'LONG' and row['Close'] < row['EMA_200']:
                    i += 1
                    continue
                if direction == 'SHORT' and row['Close'] > row['EMA_200']:
                    i += 1
                    continue

            # ── Open trade ──
            entry_price = row['Close']
            entry_date = indices[i]

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

            for j in range(i + 1, min(i + 1 + self.max_hold_days, len(data))):
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
                last_j = min(i + self.max_hold_days, len(data) - 1)
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
