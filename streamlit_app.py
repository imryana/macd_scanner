"""
MACD Trading Signal Scanner - Streamlit Web App
Real-time scanning of S&P 500 stocks for MACD trading signals
"""

import streamlit as st
import pandas as pd
from macd_scanner import MACDScanner
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MACD Signal Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📈 MACD Trading Signal Scanner")
st.markdown("### Scan S&P 500 stocks for fresh MACD crossover signals with multiple confirmations")
st.markdown("---")

# Sidebar - Configuration
st.sidebar.header("🎛️ Configuration")

st.sidebar.subheader("Indicator Toggles")
use_rsi = st.sidebar.checkbox("RSI - Relative Strength Index", value=True, 
                               help="Identifies overbought/oversold conditions")
use_adx = st.sidebar.checkbox("ADX - Average Directional Index", value=True,
                               help="Measures trend strength")
use_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True,
                                    help="Volatility and price position indicator")
use_ema200 = st.sidebar.checkbox("EMA-200", value=True,
                                  help="Long-term trend filter")

st.sidebar.markdown("---")
st.sidebar.subheader("MACD Parameters")
macd_fast = st.sidebar.number_input("Fast EMA", min_value=5, max_value=50, value=12)
macd_slow = st.sidebar.number_input("Slow EMA", min_value=10, max_value=100, value=26)
macd_signal = st.sidebar.number_input("Signal Line", min_value=5, max_value=20, value=9)
ema_period = st.sidebar.number_input("EMA Period", min_value=50, max_value=300, value=200)

st.sidebar.markdown("---")

# Display active indicators
st.sidebar.markdown("---")
st.sidebar.markdown("### Active Indicators:")
st.sidebar.markdown(f"✅ MACD (Always ON)")
st.sidebar.markdown(f"{'✅' if use_rsi else '❌'} RSI")
st.sidebar.markdown(f"{'✅' if use_adx else '❌'} ADX")
st.sidebar.markdown(f"{'✅' if use_bollinger else '❌'} Bollinger Bands")
st.sidebar.markdown(f"{'✅' if use_ema200 else '❌'} EMA-200")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Scanner Controls")
    
    if st.button("🚀 Start Scan", type="primary", use_container_width=True):
        # Initialize scanner
        with st.spinner("Initializing scanner..."):
            scanner = MACDScanner(
                macd_fast=int(macd_fast),
                macd_slow=int(macd_slow),
                macd_signal=int(macd_signal),
                ema_period=int(ema_period),
                use_rsi=use_rsi,
                use_adx=use_adx,
                use_bollinger=use_bollinger,
                use_ema200=use_ema200
            )
        
        # Scan all S&P 500 stocks
        tickers = None
        st.warning("🔍 Scanning all S&P 500 stocks - This may take 5-10 minutes...")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Scan stocks
        with st.spinner("Scanning stocks..."):
            results = scanner.scan_all_stocks(tickers=tickers)
            progress_bar.progress(100)
            status_text.text("Scan complete!")
        
        # Store results in session state
        st.session_state['results'] = results
        st.session_state['scan_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if len(results) > 0:
            st.success(f"✅ Found {len(results)} fresh signals!")
        else:
            st.warning("No fresh signals found.")

with col2:
    st.subheader("ℹ️ About")
    st.markdown("""
    **Signal Types:**
    - 🟢 **STRONG LONG**: High probability bullish setup
    - 🔵 **LONG**: Bullish setup with confirmations
    - 🔴 **STRONG SHORT**: High probability bearish setup
    - 🟠 **SHORT**: Bearish setup with confirmations
    
    **Filters:**
    - Fresh signals (0-7 days since crossover)
    - Multiple indicator confirmations
    - Trend and momentum analysis
    """)

# Display results if they exist
if 'results' in st.session_state and len(st.session_state['results']) > 0:
    results = st.session_state['results']
    
    st.markdown("---")
    st.subheader("📊 Scan Results")
    st.caption(f"Last scan: {st.session_state.get('scan_time', 'N/A')}")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    long_signals = results[results['signal'].str.contains('LONG')]
    short_signals = results[results['signal'].str.contains('SHORT')]
    strong_long = results[results['signal'] == 'STRONG LONG']
    strong_short = results[results['signal'] == 'STRONG SHORT']
    
    with col1:
        st.metric("Total Signals", len(results))
    with col2:
        st.metric("Long Signals", len(long_signals), delta="📈")
    with col3:
        st.metric("Short Signals", len(short_signals), delta="📉")
    with col4:
        st.metric("Strong Signals", len(strong_long) + len(strong_short))
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 All Signals", "🟢 Long Signals", "🔴 Short Signals", "📈 Analytics"])
    
    with tab1:
        st.dataframe(
            results,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Download button
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"macd_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        if len(long_signals) > 0:
            st.markdown(f"### 🟢 {len(long_signals)} Long Signals")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                show_strong_only = st.checkbox("Show STRONG LONG only", key="long_strong")
            with col2:
                min_volume = st.slider("Min Volume Ratio", 0.5, 3.0, 1.0, 0.1, key="long_vol")
            
            filtered_long = long_signals
            if show_strong_only:
                filtered_long = filtered_long[filtered_long['signal'] == 'STRONG LONG']
            filtered_long = filtered_long[filtered_long['volume_ratio'] >= min_volume]
            
            st.dataframe(filtered_long, use_container_width=True, hide_index=True)
        else:
            st.info("No long signals found.")
    
    with tab3:
        if len(short_signals) > 0:
            st.markdown(f"### 🔴 {len(short_signals)} Short Signals")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                show_strong_only = st.checkbox("Show STRONG SHORT only", key="short_strong")
            with col2:
                min_volume = st.slider("Min Volume Ratio", 0.5, 3.0, 1.0, 0.1, key="short_vol")
            
            filtered_short = short_signals
            if show_strong_only:
                filtered_short = filtered_short[filtered_short['signal'] == 'STRONG SHORT']
            filtered_short = filtered_short[filtered_short['volume_ratio'] >= min_volume]
            
            st.dataframe(filtered_short, use_container_width=True, hide_index=True)
        else:
            st.info("No short signals found.")
    
    with tab4:
        st.markdown("### 📊 Signal Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Signal Distribution")
            signal_counts = results['signal'].value_counts()
            st.bar_chart(signal_counts)
        
        with col2:
            st.markdown("#### Statistics")
            st.write(f"**Average Days Since Crossover:** {results['days_since_crossover'].mean():.2f}")
            st.write(f"**Average Volume Ratio:** {results['volume_ratio'].mean():.2f}")
            st.write(f"**Average 5-Day Price Change:** {results['price_change_5d_pct'].mean():.2f}%")
            
            if use_rsi:
                st.write(f"**Average RSI:** {results['rsi'].mean():.2f}")
            if use_adx:
                st.write(f"**Average ADX:** {results['adx'].mean():.2f}")
        
        # Crossover position analysis
        if 'crossover_position' in results.columns:
            st.markdown("#### Crossover Position")
            crossover_counts = results['crossover_position'].value_counts()
            st.bar_chart(crossover_counts)
        
        # Very fresh signals
        very_fresh = results[results['days_since_crossover'] <= 1]
        if len(very_fresh) > 0:
            st.markdown(f"#### 🆕 Very Fresh Signals (0-1 days): {len(very_fresh)}")
            st.dataframe(
                very_fresh[['ticker', 'signal', 'days_since_crossover', 'current_price']],
                use_container_width=True,
                hide_index=True
            )

else:
    st.info("👆 Click 'Start Scan' to begin scanning for MACD signals")
    
    # Show example of what to expect
    with st.expander("ℹ️ What will I see after scanning?"):
        st.markdown("""
        After scanning, you'll see:
        - **Summary metrics** with total signals found
        - **Interactive tables** with all signal details
        - **Filtering options** by signal type and volume
        - **Analytics** showing signal distribution and statistics
        - **Download button** to export results as CSV
        
        Each signal includes:
        - Ticker symbol and current price
        - Signal type (STRONG LONG, LONG, SHORT, STRONG SHORT)
        - Days since crossover (fresher = better)
        - Crossover position (above/below zero line)
        - All indicator values (RSI, ADX, BB position, etc.)
        - Volume ratio and price momentum
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit | MACD Scanner v1.0 | Data from Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
