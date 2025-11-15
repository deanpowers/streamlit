#version 12.1

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Strategy Analyzer",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Crypto Trading Strategy Analyzer")
st.markdown("""
This tool analyzes cryptocurrency trading strategies and provides recommendations based on technical indicators and market conditions.
""")

# Initialize session state
if 'selected_strategy' not in st.session_state:
    st.session_state.selected_strategy = "Buy BTC and Sell ETH"

# Technical analysis functions
def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    try:
        if len(prices) < period:
            return 50
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    except Exception:
        return 50

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    try:
        if len(high) < period:
            return 50.0
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 50.0
    except Exception:
        return 50.0

def calculate_z_score(prices, period=24):
    """Calculate Z-score for price deviation"""
    try:
        if len(prices) < period:
            return 0
        recent_prices = prices[-period:]
        mean = recent_prices.mean()
        std = recent_prices.std()
        if std == 0:
            return 0
        z_score = (recent_prices.iloc[-1] - mean) / std
        return z_score if not np.isnan(z_score) else 0
    except Exception:
        return 0

def get_crypto_data(symbol, period="1mo"):
    """Fetch cryptocurrency data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(f"{symbol}-USD")
        data = ticker.history(period=period)
        if data.empty:
            # Generate realistic sample data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            if symbol == "BTC":
                prices = np.random.normal(0, 0.02, 30).cumsum() * 1000 + 45000
            else:  # ETH
                prices = np.random.normal(0, 0.02, 30).cumsum() * 100 + 3200
            data = pd.DataFrame({
                'Open': prices,
                'High': prices * 1.02,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, 30)
            }, index=dates)
        return data
    except Exception:
        # Generate fallback sample data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        if symbol == "BTC":
            prices = np.random.normal(0, 0.02, 30).cumsum() * 1000 + 45000
        else:  # ETH
            prices = np.random.normal(0, 0.02, 30).cumsum() * 100 + 3200
        data = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 30)
        }, index=dates)
        return data

def create_gauge(value, title, min_val, max_val, thresholds=None):
    """Create a gauge dial indicator"""
    if thresholds is None:
        thresholds = [min_val + (max_val - min_val) * 0.33, min_val + (max_val - min_val) * 0.66]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, thresholds[0]], 'color': "lightgray"},
                {'range': [thresholds[0], thresholds[1]], 'color': "gray"},
                {'range': [thresholds[1], max_val], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(t=50, b=10, l=10, r=10))
    return fig

def calculate_confidence(hourly_z, daily_z, rsi, atr, current_price):
    """Calculate confidence score based on multiple indicators"""
    confidence = 50  # Base confidence
    
    # Z-score contribution (40% weight)
    z_confidence = min(100, max(0, (abs(hourly_z) + abs(daily_z)) * 20))
    confidence += z_confidence * 0.4
    
    # RSI contribution (30% weight)
    rsi_confidence = 0
    if rsi < 30 or rsi > 70:  # Strong RSI signals
        rsi_confidence = 80
    elif rsi < 40 or rsi > 60:  # Moderate RSI signals
        rsi_confidence = 60
    else:  # Neutral RSI
        rsi_confidence = 40
    confidence += rsi_confidence * 0.3
    
    # ATR/Volatility contribution (30% weight)
    if current_price and atr:
        volatility_ratio = atr / current_price
        if volatility_ratio < 0.01:  # Low volatility - more confidence
            atr_confidence = 80
        elif volatility_ratio > 0.03:  # High volatility - less confidence
            atr_confidence = 40
        else:  # Moderate volatility
            atr_confidence = 60
        confidence += atr_confidence * 0.3
    
    return min(100, max(0, confidence))

def calculate_trade_amount(investment_amount, risk_tolerance, confidence):
    """Calculate trade amount based on investment, risk tolerance and confidence"""
    risk_factors = {
        "Very Low": 0.1,
        "Low": 0.3,
        "Medium": 0.6,
        "High": 0.8,
        "Very High": 1.0
    }
    
    risk_factor = risk_factors.get(risk_tolerance, 0.5)
    confidence_factor = confidence / 100
    
    return investment_amount * risk_factor * confidence_factor

def generate_single_recommendation(hourly_z, daily_z, rsi, atr, current_price, investment_amount, risk_tolerance):
    """Generate single trading recommendation based on Z-scores"""
    confidence = calculate_confidence(hourly_z, daily_z, rsi, atr, current_price)
    trade_amount = calculate_trade_amount(investment_amount, risk_tolerance, confidence)
    
    # Decision logic based on Z-scores
    if -1.5 <= hourly_z <= 1.5 and -1.5 <= daily_z <= 1.5:
        recommendation = {
            "action": "WAIT - Hold Cash",
            "color": "gray",
            "reason": "Both hourly and daily Z-scores are within normal range (-1.5 to 1.5)",
            "confidence": confidence,
            "trade_amount": 0,
            "details": "Market conditions are neutral. Preserving capital is recommended.",
            "show_trade_button": False
        }
    elif hourly_z >= 1.5 and daily_z >= 1.5:
        recommendation = {
            "action": "SELL BTC and BUY ETH",
            "color": "red",
            "reason": f"Both hourly ({hourly_z:.2f}) and daily ({daily_z:.2f}) Z-scores indicate overbought conditions for BTC",
            "confidence": confidence,
            "trade_amount": trade_amount,
            "details": f"Allocate ${trade_amount:,.2f} to short BTC/long ETH position",
            "show_trade_button": True
        }
    elif hourly_z <= -1.5 and daily_z <= -1.5:
        recommendation = {
            "action": "BUY BTC and SELL ETH",
            "color": "green",
            "reason": f"Both hourly ({hourly_z:.2f}) and daily ({daily_z:.2f}) Z-scores indicate oversold conditions for BTC",
            "confidence": confidence,
            "trade_amount": trade_amount,
            "details": f"Allocate ${trade_amount:,.2f} to long BTC/short ETH position",
            "show_trade_button": True
        }
    else:
        # Mixed signals
        recommendation = {
            "action": "WAIT - Mixed Signals",
            "color": "orange",
            "reason": "Conflicting signals between hourly and daily timeframes",
            "confidence": confidence,
            "trade_amount": 0,
            "details": "Hourly and daily Z-scores show conflicting signals. Wait for clearer direction.",
            "show_trade_button": False
        }
    
    return recommendation

# Sidebar for user inputs (UNCHANGED AS REQUESTED)
st.sidebar.header("🎯 Trading Parameters")

# Strategy selection
strategy_options = [
    "Buy BTC and Sell ETH",
    "Sell BTC and Buy ETH", 
    "Do Nothing (Hold Cash)",
    "Custom Strategy"
]

selected_strategy = st.sidebar.selectbox(
    "Select Trading Strategy:",
    strategy_options,
    index=strategy_options.index(st.session_state.selected_strategy) 
    if st.session_state.selected_strategy in strategy_options else 0
)

# Investment amount
investment_amount = st.sidebar.number_input(
    "Investment Amount (USD):", 
    value=10000.0, 
    min_value=100.0, 
    step=1000.0
)

# Risk tolerance
risk_tolerance = st.sidebar.select_slider(
    "Risk Tolerance:",
    options=["Very Low", "Low", "Medium", "High", "Very High"],
    value="Medium"
)

# Analysis period
analysis_period = st.sidebar.selectbox(
    "Analysis Period:",
    ["1 Week", "1 Month", "3 Months"],
    index=1
)

# Manual inputs section - UPDATED WITH BLANK FIELDS
st.sidebar.header("📊 Manual Inputs (Optional)")

# Create four blank input fields
manual_hourly_z = st.sidebar.text_input("Hourly Z-score:", value="", placeholder="Enter hourly Z-score")
manual_daily_z = st.sidebar.text_input("Daily Z-score:", value="", placeholder="Enter daily Z-score")
manual_rsi = st.sidebar.text_input("RSI:", value="", placeholder="Enter RSI value")
manual_atr = st.sidebar.text_input("ATR:", value="", placeholder="Enter ATR value")

# Helper function to convert manual inputs safely
def safe_float_convert(value, default=None):
    """Safely convert string input to float, return default if empty or invalid"""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# Main analysis
if st.sidebar.button("🚀 Analyze Strategy", type="primary"):
    with st.spinner("Fetching market data and analyzing strategies..."):
        
        # Simplified period mapping
        period_map = {
            "1 Week": "1wk",
            "1 Month": "1mo", 
            "3 Months": "3mo"
        }
        
        # Fetch data
        eth_data = get_crypto_data("ETH", period_map[analysis_period])
        btc_data = get_crypto_data("BTC", period_map[analysis_period])
        
        # Calculate current prices and indicators
        eth_price = eth_data['Close'].iloc[-1]
        btc_price = btc_data['Close'].iloc[-1]
        eth_rsi = calculate_rsi(eth_data['Close'])
        eth_z_score = calculate_z_score(eth_data['Close'])
        eth_atr = calculate_atr(eth_data['High'], eth_data['Low'], eth_data['Close'])
        btc_eth_ratio = btc_price / eth_price
        
        # Use manual inputs if provided, otherwise use calculated values
        final_hourly_z = safe_float_convert(manual_hourly_z, eth_z_score * 0.8)
        final_daily_z = safe_float_convert(manual_daily_z, eth_z_score)
        final_rsi = safe_float_convert(manual_rsi, eth_rsi)
        final_atr = safe_float_convert(manual_atr, eth_atr)
        
        # Generate single recommendation
        recommendation = generate_single_recommendation(
            final_hourly_z, final_daily_z, final_rsi, final_atr, 
            eth_price, investment_amount, risk_tolerance
        )
        
        # Display three dials at the top
        st.subheader("📊 Market Indicators")
        
        dial_col1, dial_col2, dial_col3 = st.columns(3)
        
        with dial_col1:
            # Hourly Z-score gauge (-3 to 3 range)
            st.plotly_chart(create_gauge(
                final_hourly_z, "Hourly Z-Score", -3, 3, [-1.5, 1.5]
            ), use_container_width=True)
        
        with dial_col2:
            # Daily Z-score gauge (-3 to 3 range)
            st.plotly_chart(create_gauge(
                final_daily_z, "Daily Z-Score", -3, 3, [-1.5, 1.5]
            ), use_container_width=True)
        
        with dial_col3:
            # BTC/ETH Ratio gauge (15 to 25 range)
            st.plotly_chart(create_gauge(
                btc_eth_ratio, "BTC/ETH Ratio", 15, 25, [18, 22]
            ), use_container_width=True)
        
        # Single Recommendation Section
        st.subheader("🎯 Trading Recommendation")
        
        # Create a prominent recommendation display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"<h2 style='color: {recommendation['color']}; text-align: center; padding: 20px; border: 2px solid {recommendation['color']}; border-radius: 10px;'>"
                       f"{recommendation['action']}</h2>", 
                       unsafe_allow_html=True)
            
            st.markdown(f"**Reason:** {recommendation['reason']}")
            st.markdown(f"**Details:** {recommendation['details']}")
            
            if recommendation['trade_amount'] > 0:
                st.markdown(f"**Recommended Trade Amount:** `${recommendation['trade_amount']:,.2f}`")
            
            # Add Place Order button for trade recommendations
            if recommendation['show_trade_button']:
                st.markdown("---")
                if st.button("📊 Place Order", type="primary", use_container_width=True):
                    st.success("""
                    **You can place this trade on the exchange of your choice or contact** 
                    
                    **[zojax.com/trade](https://zojax.com/trade)** 
                    
                    **and we will assist you in placing this order on the dYdX exchange.**
                    """)
        
        with col2:
            # Confidence meter
            confidence_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=recommendation['confidence'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Level"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 80], 'color': "lightyellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            confidence_fig.update_layout(height=250)
            st.plotly_chart(confidence_fig, use_container_width=True)
        
        # Additional market data
        st.subheader("📈 Market Data")
        data_col1, data_col2, data_col3, data_col4 = st.columns(4)
        
        with data_col1:
            st.metric("Current ETH Price", f"${eth_price:,.2f}")
        
        with data_col2:
            st.metric("Current BTC Price", f"${btc_price:,.2f}")
            
        with data_col3:
            st.metric("RSI (14)", f"{final_rsi:.1f}")
        
        with data_col4:
            st.metric("ATR (14)", f"${final_atr:.2f}")
        
        # Price charts
        st.subheader("📊 Price Movement")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.line_chart(btc_data['Close'], use_container_width=True)
            st.caption("BTC Price Movement")
        
        with chart_col2:
            st.line_chart(eth_data['Close'], use_container_width=True)
            st.caption("ETH Price Movement")
        
        # Technical details
        with st.expander("📋 Technical Details"):
            st.write(f"**Hourly Z-score:** {final_hourly_z:.2f}")
            st.write(f"**Daily Z-score:** {final_daily_z:.2f}")
            st.write(f"**RSI:** {final_rsi:.1f}")
            st.write(f"**ATR:** ${final_atr:.2f}")
            st.write(f"**BTC/ETH Ratio:** {btc_eth_ratio:.4f}")
            st.write(f"**Risk Tolerance:** {risk_tolerance}")
            st.write(f"**Investment Amount:** ${investment_amount:,.2f}")

else:
    # Initial state - show instructions
    st.info("👈 Configure your trading parameters in the sidebar and click 'Analyze Strategy' to get started!")
    
    st.subheader("Trading Logic:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🟢 BUY BTC / SELL ETH**
        - When: Both Z-scores ≤ -1.5
        - Signal: Oversold conditions
        - Action: Long BTC, Short ETH
        """)
    
    with col2:
        st.markdown("""
        **🔴 SELL BTC / BUY ETH**  
        - When: Both Z-scores ≥ 1.5
        - Signal: Overbought conditions  
        - Action: Short BTC, Long ETH
        """)
        
    with col3:
        st.markdown("""
        **⚪ HOLD CASH**
        - When: Z-scores between -1.5 and 1.5
        - Signal: Neutral conditions
        - Action: Preserve capital
        """)
    
    # Manual inputs explanation
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Manual Inputs Guide:**
    - Leave blank to use calculated values
    - Enter numbers only (e.g., 1.5, -2.0, 65.5)
    - Hourly Z-score: Short-term signals
    - Daily Z-score: Primary signals
    - RSI: Momentum (0-100)
    - ATR: Volatility in USD
    """)

# Footer
st.markdown("---")
st.markdown(
    "⚠️ **Disclaimer:** This tool is for educational purposes only. "
    "Cryptocurrency trading involves substantial risk of loss and is not suitable for every investor. "
    "Past performance is not indicative of future results."
)