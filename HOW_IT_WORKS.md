# How It Works — MACD Trading Model

## Overview

This system scans all S&P 500 stocks for fresh MACD crossover signals, confirms them with multiple technical indicators, and optionally filters them through a machine learning ensemble (XGBoost + LSTM) that predicts whether a signal will be profitable over a 5, 10, or 20-day holding period.

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌────────────┐
│  Yahoo       │ ──► │  MACD Crossover  │ ──► │  Multi-      │ ──► │  ML        │
│  Finance     │     │  Detection       │     │  Indicator   │     │  Ensemble  │
│  (1yr data)  │     │  (12, 26, 9)     │     │  Filtering   │     │  Filter    │
└──────────────┘     └──────────────────┘     └──────────────┘     └────────────┘
                                                                         │
                                                                         ▼
                                                                   ┌────────────┐
                                                                   │  Graded    │
                                                                   │  Signals   │
                                                                   │  (A+ to F) │
                                                                   └────────────┘
```

---

## Architecture

| File | Purpose |
|------|---------|
| `macd_scanner.py` | Core scanner — indicator calculations, crossover detection, signal generation, S&P 500 scanning |
| `data_preparation.py` | Collects historical crossover signals, computes features, and labels them with forward returns for ML training |
| `ml_model.py` | XGBoost classifier — learns which snapshot feature patterns predict profitable signals |
| `lstm_model.py` | LSTM neural network with attention — learns temporal patterns from 30-day indicator sequences |
| `ensemble_predictor.py` | Combines XGBoost (40%) and LSTM (60%) predictions into a weighted confidence score, grades signals A+ through F |
| `train_models.py` | End-to-end training pipeline — data collection → XGBoost → LSTM → ensemble creation |
| `streamlit_app.py` | Interactive web dashboard for running scans with toggleable settings |

---

## Stage 1: Data Ingestion

**File:** `macd_scanner.py` → `analyze_stock()`

For each ticker, the scanner downloads 1 year of daily OHLCV data from Yahoo Finance via `yfinance`. The data must contain at least 200 bars (for the EMA-200 calculation).

```python
stock = yf.Ticker(ticker)
data = stock.history(period='1y')
```

---

## Stage 2: Technical Indicator Calculations

**File:** `macd_scanner.py`

All indicators are calculated from raw OHLCV data. Every indicator can be toggled on/off.

### MACD (12, 26, 9)

The Moving Average Convergence Divergence measures momentum by comparing two exponential moving averages:

- **MACD Line** = EMA(Close, 12) − EMA(Close, 26)
- **Signal Line** = EMA(MACD Line, 9)
- **Histogram** = MACD Line − Signal Line

A **bullish crossover** occurs when the MACD line crosses above the signal line.
A **bearish crossover** occurs when the MACD line crosses below the signal line.

The position relative to the **zero line** matters:
- Crossovers **above zero** = stronger signals (the faster EMA is already above the slower EMA, confirming the prevailing trend)
- Crossovers **below zero** = contrarian/early signals

### EMA-200

A 200-period exponential moving average acting as a long-term trend filter:
- Price **above** EMA-200 → uptrend → favors long signals
- Price **below** EMA-200 → downtrend → favors short signals

### RSI (14-period)

Relative Strength Index measures the speed and magnitude of price changes on a 0–100 scale:

```
RSI = 100 − (100 / (1 + RS))
RS  = Avg Gain over 14 periods / Avg Loss over 14 periods
```

- RSI > 70 → overbought (caution for longs)
- RSI < 30 → oversold (caution for shorts)
- 30–70 → neutral zone (ideal)

### ADX (14-period)

Average Directional Index measures trend strength (regardless of direction):

- ADX > 25 → strong trend
- ADX > 20 → moderate trend
- ADX < 20 → weak/no trend (signals less reliable)

Also computes +DI and −DI (directional indicators) to confirm trend direction.

### Bollinger Bands (20, 2)

Measures volatility and price position within a statistical envelope:

- **Middle Band** = SMA(Close, 20)
- **Upper Band** = Middle + 2 × StdDev(Close, 20)
- **Lower Band** = Middle − 2 × StdDev(Close, 20)
- **%B** = (Close − Lower) / (Upper − Lower)

Key levels:
- %B > 0.8 → near upper band (overbought)
- %B < 0.2 → near lower band (oversold)
- 0.3 < %B < 0.7 → neutral zone

### Volume Analysis

Compares current volume to the 20-day moving average:

```
Volume Ratio = Today's Volume / SMA(Volume, 20)
```

A ratio > 1.0 confirms conviction behind the move.

---

## Stage 3: Crossover Detection

**File:** `macd_scanner.py` → `detect_crossover()`

Scans the entire price history for MACD/Signal line crossovers:

```python
# Bullish crossover: MACD crosses above Signal
bullish = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))

# Bearish crossover: MACD crosses below Signal
bearish = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
```

Only **fresh** crossovers (0–7 days old) are kept. Anything older is discarded.

---

## Stage 4: Signal Generation

**File:** `macd_scanner.py` → `generate_signal()`

Each fresh crossover is run through a multi-factor confirmation system. The signal strength depends on how many conditions align:

### STRONG LONG (Highest Confidence)

All of these must be true:
- Bullish MACD crossover
- Crossover occurred **above** the zero line
- Price is **above** EMA-200
- ADX > 25 (strong trend)
- RSI between 30–70 (not overbought)
- Bollinger %B not overbought (< 0.8)

*OR:*

- Bullish crossover above zero
- Price above EMA-200
- RSI < 35 or %B < 0.2 (oversold bounce within uptrend)
- ADX > 20

### LONG (Standard Confidence)

Relaxed version — requires crossover + price above EMA-200 + moderate trend, but allows fewer confirmations.

### STRONG SHORT / SHORT

Mirror logic: bearish crossover, price below EMA-200, crossover below zero line, with the same ADX/RSI/BB confirmation rules inverted.

### No Signal

If the crossover doesn't pass enough confirmation filters, it's discarded entirely. This is the first layer of quality control.

---

## Stage 5: Risk Management

**File:** `macd_scanner.py` → `analyze_stock()`

Every valid signal includes calculated trade levels:

| Level | Long | Short |
|-------|------|-------|
| **Entry** | Current close price | Current close price |
| **Stop Loss** | Entry × 0.95 (5% below) | Entry × 1.05 (5% above) |
| **Take Profit** | Entry + (Risk × 1.5) | Entry − (Risk × 1.5) |
| **Risk:Reward** | 1.5:1 | 1.5:1 |

The stop loss and take profit ensure a positive expected value even with a sub-50% win rate, as long as the average win is 1.5× the average loss.

---

## Stage 6: Machine Learning Filter (Optional)

When enabled, every signal that passes the rule-based filter is further evaluated by an ML ensemble that predicts whether the trade will be profitable.

### Feature Extraction

**File:** `data_preparation.py` → `extract_features_at_point()`

Two types of features are extracted at each crossover point:

#### Snapshot Features (for XGBoost) — 33 features

A flat dictionary of values at the moment of crossover:

| Category | Features |
|----------|----------|
| **MACD** | macd_value, signal, histogram, above_zero flag, histogram_slope (5-day), histogram_momentum (5-day avg) |
| **RSI** | rsi value, oversold/overbought/neutral flags, rsi_slope (5-day) |
| **ADX** | adx value, +DI, −DI, di_diff, adx_strong flag |
| **Price** | close price, distance from EMA-200 (%), above_ema200 flag, returns over 1/5/10/20 days |
| **Volume** | raw volume, volume_ratio_20d, volume_trend flag |
| **Bollinger** | %B position, bandwidth, bb_squeeze flag |
| **Volatility** | 10-day and 20-day price volatility |
| **Context** | crossover_type (bullish=1, bearish=−1) |

Plus 3 interaction features: `macd × rsi`, `adx × di_diff`, `volume_ratio × returns_1d`.

#### Sequence Features (for LSTM) — 30 × 8 matrix

A time-series of the last 30 days of 8 normalized indicators:

```
[Close, Volume, MACD, Signal, RSI, ADX, BB_%B, EMA_200]
```

Each column is z-score normalized (mean=0, std=1) within the 30-day window.

### Training Data Collection

**File:** `data_preparation.py` → `collect_training_data()`

1. Download 3–10 years of daily data for 30–500 stocks
2. Find every historical MACD crossover
3. At each crossover, extract snapshot + sequence features
4. Label each signal with forward returns at 5, 10, and 20 days
5. A signal is labeled **profitable** if the forward return exceeds 0.5% (covering estimated transaction costs)
6. Save snapshot features to `training_data.csv` and sequences to `training_data_sequences.npy`

### Model 1: XGBoost Classifier

**File:** `ml_model.py`

- **Algorithm:** Gradient-boosted decision trees (XGBoost)
- **Input:** 33+ snapshot features (flat dictionary)
- **Output:** Probability that the signal will be profitable (0.0–1.0)
- **Training:** 80/20 train-test split with stratification
- **Features:** StandardScaler normalization, class imbalance handled via `scale_pos_weight`
- **Evaluation:** AUC-ROC, accuracy, confusion matrix, 5-fold cross-validation
- **Hyperparameters:** 200 trees, depth 6, learning rate 0.05, subsample 0.9

**Why XGBoost:** Excels at learning non-linear relationships in tabular features. Tells you *which features matter most* (feature importance ranking). Fast to train and predict.

### Model 2: LSTM Neural Network

**File:** `lstm_model.py`

```
Input (30 × 8) → LSTM (2 layers, 128 hidden) → Attention → FC(64) → FC(32) → Sigmoid
```

- **Algorithm:** Long Short-Term Memory with attention mechanism
- **Input:** 30-day sequence of 8 normalized indicators
- **Output:** Probability that the signal will be profitable (0.0–1.0)
- **Architecture:**
  - 2-layer LSTM with 128 hidden units and 0.3 dropout
  - Attention mechanism that learns which days in the 30-day window matter most
  - Fully connected head: 128 → 64 → 32 → 1 with ReLU, dropout, and sigmoid output
- **Training:** Adam optimizer, BCELoss, ReduceLROnPlateau scheduler, gradient clipping (max norm 1.0), early stopping (patience 10)
- **GPU accelerated:** Automatically uses CUDA if available

**Why LSTM with Attention:** Captures temporal patterns — rising MACD momentum over several days, volume buildups, RSI divergences — that a snapshot-based model would miss. The attention mechanism lets it focus on the most relevant days (e.g., the 2–3 days leading up to the crossover).

### Ensemble Predictor

**File:** `ensemble_predictor.py`

Combines both models with weighted voting:

```
Ensemble Confidence = 0.4 × XGBoost_confidence + 0.6 × LSTM_confidence
```

LSTM gets higher weight because temporal patterns tend to be more predictive for crossover-type signals.

#### Signal Grading

| Grade | Confidence Range | Meaning |
|-------|-----------------|---------|
| A+ | 95–100% | Exceptional — highest probability of profit |
| A | 85–95% | Excellent |
| B | 75–85% | Good |
| C | 65–75% | Fair (default acceptance threshold) |
| D | 50–65% | Poor — below default threshold |
| F | < 50% | Reject — model predicts unprofitable |

The default **confidence threshold** is 65% — signals graded D or F are rejected. This threshold is adjustable:
- **Conservative (75–80%):** Fewer signals, higher quality
- **Balanced (65–70%):** Moderate signals, good quality
- **Aggressive (55–60%):** More signals, mixed quality

---

## Stage 7: Output

### Command Line (`macd_scanner.py`)

The scanner outputs a sorted table of all signals:

```
📈 LONG SIGNALS (4 stocks):
  ticker  signal       days_since  price   entry   stop_loss  take_profit  rsi    adx   ml_grade
  NVDA    STRONG LONG  1           875.32  875.32  831.55     941.00       52.3   28.1  A
  AAPL    LONG (B)     3           185.20  185.20  175.94     199.09       48.7   24.5  B
  ...
```

### Web App (`streamlit_app.py`)

Interactive Streamlit dashboard with:
- Toggle switches for each indicator
- Adjustable parameters (MACD periods, RSI levels, etc.)
- ML confidence slider
- Separate tabs for long/short signals
- Signal analytics and charts
- CSV export

---

## Training Pipeline

**File:** `train_models.py`

Run `python train_models.py` to launch an interactive menu:

| Option | What It Does | Time |
|--------|-------------|------|
| **1 — Quick Test** | 30 stocks, 10yr history, default hyperparameters | ~15–20 min |
| **2 — Full Training** | All S&P 500, 10yr history, hyperparameter optimization | ~3–4 hours |
| **3 — Custom** | Choose tickers, lookback, target period, epochs | Varies |
| **4 — Load Models** | Skip training, load existing `.pkl` and `.pth` files | Instant |

The pipeline runs 4 steps sequentially:

1. **Data Collection** → `training_data.csv` + `training_data_sequences.npy`
2. **XGBoost Training** → `xgboost_model_5d.pkl` + `xgboost_diagnostics_5d.png`
3. **LSTM Training** → `lstm_model_5d.pth` + `lstm_training_history_5d.png`
4. **Ensemble Creation** → Loads both models, ready for prediction

---

## File Dependencies

```
macd_scanner.py
├── yfinance (market data)
├── ensemble_predictor.py (optional ML filter)
│   ├── ml_model.py (XGBoost)
│   │   ├── xgboost_model_5d.pkl
│   │   └── sklearn StandardScaler
│   └── lstm_model.py (LSTM)
│       └── lstm_model_5d.pth
└── data_preparation.py (training data)
    └── macd_scanner.py (indicator calculations)

streamlit_app.py
└── macd_scanner.py (everything above)

train_models.py
├── data_preparation.py
├── ml_model.py
├── lstm_model.py
└── ensemble_predictor.py
```

---

## Key Design Decisions

**Why MACD crossovers as the base signal?**
MACD crossovers are one of the most widely followed momentum signals. By themselves they have a ~50% hit rate, but with multi-factor confirmation the hit rate improves significantly.

**Why require freshness (0–7 days)?**
Stale crossovers have already been priced in. Fresh ones (especially 0–2 days) have the most edge because the trend is just beginning.

**Why the zero-line matters:**
A bullish crossover above the zero line means the 12-EMA is already above the 26-EMA — the stock is in an established uptrend and the crossover confirms continuation. Below-zero crossovers are early/contrarian and less reliable.

**Why 5% stop loss with 1.5:1 R:R?**
At a 1.5:1 reward-to-risk ratio, you only need a ~40% win rate to break even. The multi-factor confirmation pushes the actual win rate well above that, creating positive expected value.

**Why an ensemble instead of a single model?**
XGBoost and LSTM capture fundamentally different patterns — XGBoost sees feature relationships (e.g., "when RSI is 45 and ADX is 28, the signal is usually profitable"), while LSTM sees temporal dynamics (e.g., "histogram has been expanding for 4 consecutive days"). Combining them reduces variance and improves robustness.

**Why 0.5% profitability threshold in labeling?**
This accounts for estimated round-trip transaction costs (commissions + slippage), ensuring that signals labeled "profitable" represent actual realized profit after friction.
