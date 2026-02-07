# MACD Trading Signal Scanner 📈

A comprehensive Python-based MACD trading model that scans S&P 500 stocks for fresh crossover signals with multiple confirmation factors.

**🌐 [Try the Live Web App](https://your-app-name.streamlit.app)** (Deploy instructions below)

## 🎯 Features

### Technical Indicators (All Toggleable)
- **MACD (12, 26, 9)**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index (overbought/oversold detection)
- **ADX**: Average Directional Index (trend strength measurement)
- **Bollinger Bands**: Volatility and price position indicator
- **EMA-200**: 200-period Exponential Moving Average for trend confirmation
- **Volume Analysis**: Relative volume confirmation
- **Price Momentum**: 5-day price change tracking

### Signal Detection
- **Bullish Crossovers**: MACD line crosses above Signal line
- **Bearish Crossovers**: MACD line crosses below Signal line
- **Fresh Signals**: Identifies crossovers that occurred 0-7 days ago
- **Zero Line Analysis**: Tracks whether crossover happened above or below zero line
- **Signal Strength Classification**:
  - **STRONG LONG**: Bullish crossover above zero + multiple confirmations
  - **LONG**: Bullish crossover with good confirmations
  - **SHORT**: Bearish crossover with confirmations
  - **STRONG SHORT**: Bearish crossover below zero + multiple confirmations

### 🎛️ Indicator Toggles
Enable/disable any indicator to test different strategies:
- Turn OFF all extra indicators for aggressive scanning (more signals)
- Turn ON all indicators for conservative scanning (higher quality signals)
- Mix and match based on your trading style

### 🌐 Web Interface (Streamlit App)
- **Interactive Dashboard**: Real-time scanning with visual results
- **Customizable Settings**: Adjust all parameters through UI
- **Filtered Views**: Separate tabs for long/short signals
- **Analytics**: Charts and statistics about your signals
- **Export Results**: Download signals as CSV
- **Mobile-Friendly**: Works on any device

## 🚀 Quick Start

### Option 1: Run Streamlit Web App (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- 🎛️ Toggle indicators on/off
- 📊 Interactive data tables
- 📈 Visual analytics
- 💾 Download results as CSV
- 🔍 Filter by signal type, volume, etc.

### Option 2: Use Python Script

```python
from macd_scanner import MACDScanner

# Initialize scanner with your chosen indicators
scanner = MACDScanner(
    use_rsi=True,
    use_adx=True,
    use_bollinger=True,
    use_ema200=True
)

# Scan all S&P 500 stocks
results_df = scanner.scan_all_stocks()

# Display results
print(results_df)
```

### Option 3: Run from Command Line

```bash
python macd_scanner.py
```

This will scan all S&P 500 stocks and display results in the terminal.

## 📝 Usage Examples

### Analyze Specific Stocks

```python
from macd_scanner import MACDScanner

scanner = MACDScanner()

# Analyze single stock
result = scanner.analyze_stock('AAPL')
if result:
    print(f"Signal: {result['signal']}")
    print(f"Days since crossover: {result['days_since_crossover']}")
    print(f"Crossover position: {result['crossover_position']}")
    print(f"RSI: {result['rsi']}")
    print(f"ADX: {result['adx']}")

# Scan custom list
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META']
results_df = scanner.scan_all_stocks(tickers=tech_stocks)
```

### Customize Parameters & Indicators

```python
# Custom MACD parameters with selective indicators
scanner = MACDScanner(
    macd_fast=12,         # Fast EMA period
    macd_slow=26,         # Slow EMA period
    macd_signal=9,        # Signal line period
    ema_period=200,       # Long-term trend EMA
    use_rsi=True,         # Enable RSI
    use_adx=False,        # Disable ADX
    use_bollinger=True,   # Enable Bollinger Bands
    use_ema200=True       # Enable EMA-200
)
```

## Output

### Console Output
- Real-time scanning progress
- Signals grouped by type (LONG/SHORT)
- Summary statistics

### CSV Export
Results are automatically saved to `macd_signals_YYYYMMDD_HHMMSS.csv` with columns:
- ticker
- signal (STRONG LONG, LONG, SHORT, STRONG SHORT)
- days_since_crossover
- current_price
- macd
- signal_line
- histogram
- ema_200
- price_vs_ema200
- distance_from_ema200_pct
- volume_ratio
- price_change_5d_pct
- crossover_date
- current_date

## Trading Strategy Guidelines

### LONG Signals
- **Entry**: On STRONG LONG or LONG signals
- **Confirmation**: 
  - Price above EMA-200 (bullish trend)
  - MACD above zero line (strong momentum)
  - Increasing histogram (accelerating momentum)
- **Stop Loss**: Below recent swing low or EMA-200
- **Target**: Previous resistance or risk/reward ratio of 2:1+

### SHORT Signals
- **Entry**: On STRONG SHORT or SHORT signals
- **Confirmation**:
  - Price below EMA-200 (bearish trend)
  - MACD below zero line (strong momentum down)
  - Decreasing histogram (accelerating downward momentum)
- **Stop Loss**: Above recent swing high or EMA-200
- **Target**: Previous support or risk/reward ratio of 2:1+

### Risk Management
- Never risk more than 1-2% of account per trade
- Use proper position sizing
- Consider overall market conditions
- Fresh signals (0-1 days) may be more reliable than older signals
- Higher volume ratio (>1.5) indicates stronger conviction
- Large single-day moves (>3%) may indicate chasing - consider scaling in

## 🌐 Deploy to Web (Streamlit Cloud)

### Quick Deploy (FREE!)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `streamlit_app.py`
   - Click "Deploy"

3. **Done!** Your app is live at `https://your-app-name.streamlit.app`

📖 **Detailed Instructions**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 📁 Project Structure

```
MACD Model/
├── streamlit_app.py          # Streamlit web interface
├── macd_scanner.py            # Core scanner logic
├── MACD_Analysis.ipynb        # Jupyter notebook for analysis
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── DEPLOYMENT_GUIDE.md        # Detailed deployment steps
└── README.md                  # This file
```

## 🔧 Configuration Options

### Indicator Toggles
- **RSI**: Overbought/oversold detection
- **ADX**: Trend strength measurement (>25 = strong trend)
- **Bollinger Bands**: Volatility analysis
- **EMA-200**: Long-term trend filter

### MACD Parameters
- Fast EMA: 12 (default)
- Slow EMA: 26 (default)
- Signal Line: 9 (default)

### Signal Freshness
- Signals are filtered to 0-7 days since crossover
- Fresher signals generally more reliable

## 📊 Understanding the Signals

### Why Zero Line Position Matters
- **Crossover ABOVE zero**: Stronger bullish signal (12-EMA significantly above 26-EMA)
- **Crossover BELOW zero**: Stronger bearish signal (12-EMA significantly below 26-EMA)
- Crossovers at the zero line indicate the strongest momentum

### Signal Quality Indicators
- ✅ **Strong Trend** (ADX > 25)
- ✅ **Not Overbought/Oversold** (RSI 30-70)
- ✅ **Above Average Volume** (ratio > 1.2)
- ✅ **Fresh Signal** (0-2 days old)
- ✅ **Favorable BB Position** (not at extreme bands)

## 🤝 Contributing

Feel free to fork, modify, and submit pull requests. Suggestions for improvements are welcome!

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It does not constitute financial advice. Always do your own research and consider consulting with a financial advisor before making trading decisions. Past performance does not guarantee future results.

## 📄 License

Free to use and modify for personal use.

---

**Built with:** Python, Pandas, yFinance, Streamlit
**Data Source:** Yahoo Finance API
