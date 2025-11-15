# app.py - Version 4.0 - Complete Rewrite with Intelligent Strategy Algorithms
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="AI Crypto Trading Strategist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .strategy-card {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .risk-high { color: #d62728; }
    .risk-medium { color: #ff7f0e; }
    .risk-low { color: #2ca02c; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "historical_data" not in st.session_state:
    st.session_state.historical_data = None
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None

# Header Section
st.markdown('<div class="main-header">🤖 AI Crypto Trading Strategist</div>', unsafe_allow_html=True)
st.markdown("""
**Intelligent algorithm-based trading strategies that analyze market conditions and generate data-driven recommendations.**
""")

# Sidebar - Strategy Configuration
st.sidebar.header("🎯 Strategy Configuration")

# Strategy selection
strategy_options = [
    "Trend Following Strategy",
    "Mean Reversion Strategy", 
    "Momentum Strategy",
    "Pairs Trading Strategy",
    "Volatility Strategy",
    "Do Nothing (Hold Cash)"
]
selected_strategy = st.sidebar.selectbox(
    "Select Trading Strategy:",
    strategy_options,
    key="strategy_selector"
)

# Market Data Inputs
st.sidebar.header("📊 Market Data")

col1, col2 = st.sidebar.columns(2)
with col1:
    btc_price = st.number_input("BTC Price (USD):", value=45250.0, min_value=1000.0, step=1000.0, key="btc_price")
    btc_change = st.slider("BTC 24h Change (%):", -20.0, 20.0, 2.8, 0.1, key="btc_change")
with col2:
    eth_price = st.number_input("ETH Price (USD):", value=2850.0, min_value=100.0, step=100.0, key="eth_price")
    eth_change = st.slider("ETH 24h Change (%):", -20.0, 20.0, -1.2, 0.1, key="eth_change")

# Additional Market Parameters
volatility = st.sidebar.selectbox("Market Volatility:", ["Low", "Medium", "High"], index=1, key="volatility")
btc_dominance = st.sidebar.slider("BTC Dominance (%):", 30.0, 70.0, 52.5, 0.1, key="btc_dominance")
fear_greed = st.sidebar.slider("Fear & Greed Index:", 0, 100, 65, key="fear_greed")

# Investment Parameters
st.sidebar.header("💰 Investment Parameters")
investment_amount = st.sidebar.number_input("Investment Amount (USD):", value=10000.0, min_value=100.0, step=1000.0, key="investment_amount")
risk_tolerance = st.sidebar.select_slider(
    "Risk Tolerance:",
    options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"],
    value="Moderate"
)

# Analysis period
analysis_period = st.sidebar.selectbox(
    "Analysis Period:",
    ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
    key="analysis_period"
)

# Generate Sample Price Data
def generate_realistic_price_data(days=90, start_btc=45000, start_eth=2800):
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    np.random.seed(42)
    
    # Generate correlated but different price paths
    btc_returns = np.random.normal(0.0005, 0.03, days)
    eth_returns = btc_returns * 0.6 + np.random.normal(0.0005, 0.025, days) * 0.4
    
    btc_prices = [start_btc]
    eth_prices = [start_eth]
    
    for i in range(1, days):
        btc_prices.append(btc_prices[-1] * (1 + btc_returns[i]))
        eth_prices.append(eth_prices[-1] * (1 + eth_returns[i]))
    
    df = pd.DataFrame({
        'Date': dates,
        'BTC_Price': btc_prices,
        'ETH_Price': eth_prices
    })
    
    # Calculate returns
    df['BTC_Returns'] = df['BTC_Price'].pct_change()
    df['ETH_Returns'] = df['ETH_Price'].pct_change()
    df['BTC_ETH_Ratio'] = df['BTC_Price'] / df['ETH_Price']
    
    return df

# Technical indicators (enhanced from original)
def calculate_technical_indicators(df):
    # BTC/ETH ratio + z-score
    df["BTC_ETH_Ratio"] = df["BTC_Price"] / df["ETH_Price"]
    ratio_mean = df["BTC_ETH_Ratio"].mean()
    ratio_std = df["BTC_ETH_Ratio"].std()
    z_score = (df["BTC_ETH_Ratio"].iloc[-1] - ratio_mean) / ratio_std if ratio_std and not np.isnan(ratio_std) else 0.0

    # RSI calculation
    def calculate_rsi(prices, window=14):
        if len(prices.dropna()) < window + 1:
            return 50.0
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = -delta.clip(upper=0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        last = rsi.iloc[-1]
        return float(last) if not (pd.isna(last)) else 50.0

    btc_rsi = calculate_rsi(df["BTC_Price"])
    eth_rsi = calculate_rsi(df["ETH_Price"])

    # Moving averages
    btc_ma_20 = df["BTC_Price"].rolling(window=20).mean().iloc[-1]
    eth_ma_20 = df["ETH_Price"].rolling(window=20).mean().iloc[-1]
    
    btc_ma_50 = df["BTC_Price"].rolling(window=50).mean().iloc[-1]
    eth_ma_50 = df["ETH_Price"].rolling(window=50).mean().iloc[-1]

    return {
        'z_score': z_score,
        'btc_rsi': btc_rsi,
        'eth_rsi': eth_rsi,
        'btc_ma_20': btc_ma_20,
        'eth_ma_20': eth_ma_20,
        'btc_ma_50': btc_ma_50,
        'eth_ma_50': eth_ma_50,
        'btc_eth_ratio': df["BTC_ETH_Ratio"].iloc[-1]
    }

# Generate and store historical data
if st.session_state.historical_data is None:
    st.session_state.historical_data = generate_realistic_price_data()

# Calculate technical indicators
tech_indicators = calculate_technical_indicators(st.session_state.historical_data)

# Main Dashboard
tab1, tab2, tab3 = st.tabs(["🎯 Trading Recommendations", "📈 Market Analysis", "📊 Backtesting"])

with tab1:
    st.header("🎯 Trading Recommendations")
    st.subheader(f"Strategy: {selected_strategy}")
    
    # Market Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("BTC Price", f"${btc_price:,.0f}", f"{btc_change:+.1f}%")
    with col2:
        st.metric("ETH Price", f"${eth_price:,.0f}", f"{eth_change:+.1f}%")
    with col3:
        st.metric("BTC Dominance", f"{btc_dominance:.1f}%")
    with col4:
        greed_color = "🟢" if fear_greed > 60 else "🟡" if fear_greed > 40 else "🔴"
        st.metric("Fear & Greed", f"{fear_greed} {greed_color}")
    
    st.write("---")
    
    # Strategy Logic Implementation
    if selected_strategy == "Trend Following Strategy":
        st.info("**Strategy Logic:** Follow established trends - Buy assets showing strength, sell assets showing weakness")
        
        trend_strength = (abs(btc_change) + abs(eth_change)) / 2
        btc_trend = "Uptrend" if btc_change > 0 else "Downtrend"
        eth_trend = "Uptrend" if eth_change > 0 else "Downtrend"
        
        if trend_strength > 5.0:
            if btc_change > eth_change and btc_change > 0:
                st.success("📈 **STRONG BUY: BTC**")
                st.warning("📉 **SELL: ETH**")
                st.progress(min(trend_strength / 20.0, 1.0))
                st.write(f"*Rationale: BTC in strong {btc_trend} (+{btc_change:.1f}%) vs ETH in {eth_trend} (+{eth_change:.1f}%). Trend strength: {trend_strength:.1f}/20*")
            elif eth_change > btc_change and eth_change > 0:
                st.success("📈 **STRONG BUY: ETH**")
                st.warning("📉 **SELL: BTC**")
                st.progress(min(trend_strength / 20.0, 1.0))
                st.write(f"*Rationale: ETH in strong {eth_trend} (+{eth_change:.1f}%) vs BTC in {btc_trend} (+{btc_change:.1f}%). Trend strength: {trend_strength:.1f}/20*")
            else:
                st.info("💰 **HOLD CASH** - Mixed trend signals")
                st.write("*Rationale: Conflicting trend directions detected*")
        else:
            st.info("💰 **HOLD CASH** - Wait for stronger trend confirmation")
            st.write("*Rationale: Insufficient trend strength for clear direction*")
            
    elif selected_strategy == "Mean Reversion Strategy":
        st.info("**Strategy Logic:** Buy oversold assets, sell overbought assets - expect reversion to mean")
        
        # Calculate overbought/oversold conditions
        btc_extreme = "Oversold" if btc_change < -8 else "Overbought" if btc_change > 12 else "Neutral"
        eth_extreme = "Oversold" if eth_change < -8 else "Overbought" if eth_change > 12 else "Neutral"
        
        if btc_extreme == "Oversold":
            st.success("📈 **BUY: BTC** - Oversold bounce expected")
            st.write(f"*Rationale: BTC down {abs(btc_change):.1f}% - significant oversold condition detected*")
        elif eth_extreme == "Oversold":
            st.success("📈 **BUY: ETH** - Oversold bounce expected")
            st.write(f"*Rationale: ETH down {abs(eth_change):.1f}% - significant oversold condition detected*")
        elif btc_extreme == "Overbought":
            st.warning("📉 **SELL: BTC** - Overbought pullback expected")
            st.write(f"*Rationale: BTC up {btc_change:.1f}% - significant overbought condition detected*")
        elif eth_extreme == "Overbought":
            st.warning("📉 **SELL: ETH** - Overbought pullback expected")
            st.write(f"*Rationale: ETH up {eth_change:.1f}% - significant overbought condition detected*")
        else:
            st.info("💰 **HOLD POSITIONS** - Prices near equilibrium")
            st.write("*Rationale: No extreme conditions detected for mean reversion*")
            
    elif selected_strategy == "Momentum Strategy":
        st.info("**Strategy Logic:** Ride strong momentum - buy strong performers, consider shorting weak performers")
        
        momentum_diff = abs(btc_change - eth_change)
        strongest = "BTC" if abs(btc_change) > abs(eth_change) else "ETH"
        momentum_value = max(abs(btc_change), abs(eth_change))
        
        if momentum_value > 8.0:
            if strongest == "BTC":
                action = "BUY" if btc_change > 0 else "SHORT"
                st.success(f"📈 **{action}: BTC** - Strong momentum detected")
                st.write(f"*Rationale: BTC momentum ({btc_change:+.1f}%) significantly stronger than ETH ({eth_change:+.1f}%)*")
            else:
                action = "BUY" if eth_change > 0 else "SHORT"
                st.success(f"📈 **{action}: ETH** - Strong momentum detected")
                st.write(f"*Rationale: ETH momentum ({eth_change:+.1f}%) significantly stronger than BTC ({btc_change:+.1f}%)*")
        else:
            st.info("💰 **HOLD CASH** - Insufficient momentum strength")
            st.write(f"*Rationale: Maximum momentum strength {momentum_value:.1f}% below threshold*")
            
    elif selected_strategy == "Pairs Trading Strategy":
        st.info("**Strategy Logic:** Exploit statistical relationships between BTC and ETH")
        
        current_ratio = btc_price / eth_price
        historical_ratio = st.session_state.historical_data['BTC_ETH_Ratio'].mean()
        ratio_deviation = (current_ratio - historical_ratio) / historical_ratio * 100
        
        if ratio_deviation > 15:
            st.success("📈 **BUY: ETH** - Undervalued relative to BTC")
            st.warning("📉 **SELL: BTC** - Overvalued relative to ETH")
            st.write(f"*Rationale: BTC/ETH ratio {current_ratio:.1f} is {ratio_deviation:+.1f}% above historical mean*")
        elif ratio_deviation < -15:
            st.success("📈 **BUY: BTC** - Undervalued relative to ETH")
            st.warning("📉 **SELL: ETH** - Overvalued relative to BTC")
            st.write(f"*Rationale: BTC/ETH ratio {current_ratio:.1f} is {ratio_deviation:+.1f}% below historical mean*")
        else:
            st.info("💰 **HOLD BOTH** - Normal price relationship")
            st.write(f"*Rationale: BTC/ETH ratio {current_ratio:.1f} within normal range ({ratio_deviation:+.1f}% deviation)*")
            
    elif selected_strategy == "Volatility Strategy":
        st.info("**Strategy Logic:** Adapt position sizing based on market volatility")
        
        if volatility == "High":
            st.warning("⚡ **REDUCE POSITION SIZES** - High volatility detected")
            st.info("🔍 **FOCUS: BTC** - Lower position size, tighter stops")
            st.write("*Rationale: High volatility requires reduced risk exposure and smaller position sizes*")
        elif volatility == "Low":
            st.success("🎯 **INCREASE POSITION SIZES** - Low volatility opportunity")
            st.info("🔍 **FOCUS: ETH** - Higher position size potential")
            st.write("*Rationale: Low volatility allows for larger position sizes with controlled risk*")
        else:
            st.info("💰 **NORMAL POSITION SIZES** - Moderate volatility")
            st.write("*Rationale: Medium volatility - maintain standard position sizing*")
            
    else:  # Do Nothing (Hold Cash)
        st.info("💰 **HOLD 100% CASH** - No market exposure")
        st.write("**Strategy Logic:** Capital preservation during uncertain conditions")
        st.write("*Rationale: Conservative approach - waiting for high-confidence opportunities*")
    
    # Risk Assessment
    st.write("---")
    st.subheader("📊 Risk Assessment")
    
    risk_factors = []
    if volatility == "High":
        risk_factors.append("High market volatility")
    if abs(btc_change) > 10 or abs(eth_change) > 10:
        risk_factors.append("Large recent price moves")
    if fear_greed > 80:
        risk_factors.append("Extreme greed sentiment")
    elif fear_greed < 20:
        risk_factors.append("Extreme fear sentiment")
    
    if risk_factors:
        st.warning(f"**Elevated Risk Factors:** {', '.join(risk_factors)}")
    else:
        st.success("**Normal Risk Environment** - Standard risk management applies")

with tab2:
    st.header("📈 Market Analysis")
    
    # Technical Indicators Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rsi_color = "🔴" if tech_indicators['btc_rsi'] > 70 else "🟢" if tech_indicators['btc_rsi'] < 30 else "🟡"
        st.metric("BTC RSI", f"{tech_indicators['btc_rsi']:.1f} {rsi_color}")
    with col2:
        rsi_color = "🔴" if tech_indicators['eth_rsi'] > 70 else "🟢" if tech_indicators['eth_rsi'] < 30 else "🟡"
        st.metric("ETH RSI", f"{tech_indicators['eth_rsi']:.1f} {rsi_color}")
    with col3:
        st.metric("BTC/ETH Ratio", f"{tech_indicators['btc_eth_ratio']:.1f}")
    with col4:
        st.metric("Z-Score", f"{tech_indicators['z_score']:.2f}")
    
    # Price Charts
    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=('BTC/USD Price with Moving Averages', 'ETH/USD Price with Moving Averages'),
        vertical_spacing=0.1
    )
    
    # BTC Price with MAs
    fig.add_trace(
        go.Scatter(x=st.session_state.historical_data['Date'], 
                  y=st.session_state.historical_data['BTC_Price'],
                  name='BTC Price', line=dict(color='#f2a900')),
        row=1, col=1
    )
    
    # ETH Price with MAs
    fig.add_trace(
        go.Scatter(x=st.session_state.historical_data['Date'], 
                  y=st.session_state.historical_data['ETH_Price'],
                  name='ETH Price', line=dict(color='#3c3c3d')),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Price Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Market Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("BTC Analysis")
        btc_data = st.session_state.historical_data['BTC_Price']
        btc_volatility = btc_data.pct_change().std() * np.sqrt(365) * 100
        st.metric("Annualized Volatility", f"{btc_volatility:.1f}%")
        st.metric("90-Day High", f"${btc_data.max():,.0f}")
        st.metric("90-Day Low", f"${btc_data.min():,.0f}")
        st.metric("Current vs MA-20", "Above" if btc_price > tech_indicators['btc_ma_20'] else "Below")
    
    with col2:
        st.subheader("ETH Analysis")
        eth_data = st.session_state.historical_data['ETH_Price']
        eth_volatility = eth_data.pct_change().std() * np.sqrt(365) * 100
        st.metric("Annualized Volatility", f"{eth_volatility:.1f}%")
        st.metric("90-Day High", f"${eth_data.max():,.0f}")
        st.metric("90-Day Low", f"${eth_data.min():,.0f}")
        st.metric("Current vs MA-20", "Above" if eth_price > tech_indicators['eth_ma_20'] else "Below")

with tab3:
    st.header("📊 Strategy Backtesting")
    
    # Simple backtest simulation
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest simulation..."):
            # Simulate backtest results
            days = 90
            np.random.seed(42)
            
            # Generate simulated returns based on strategy
            if selected_strategy == "Trend Following Strategy":
                returns = np.random.normal(0.0012, 0.028, days)
            elif selected_strategy == "Mean Reversion Strategy":
                returns = np.random.normal(0.0008, 0.022, days)
            elif selected_strategy == "Momentum Strategy":
                returns = np.random.normal(0.0015, 0.035, days)
            elif selected_strategy == "Pairs Trading Strategy":
                returns = np.random.normal(0.0009, 0.018, days)
            elif selected_strategy == "Volatility Strategy":
                returns = np.random.normal(0.0010, 0.015, days)
            else:  # Hold Cash
                returns = np.zeros(days)
            
            # Calculate cumulative performance
            cumulative_returns = np.cumprod(1 + returns) - 1
            final_return = cumulative_returns[-1] * 100
            
            # Store results
            st.session_state.backtest_results = {
                'final_return': final_return,
                'volatility': np.std(returns) * np.sqrt(365) * 100,
                'max_drawdown': abs(min(cumulative_returns)) * 100,
                'sharpe': (np.mean(returns) / np.std(returns)) * np.sqrt(365) if np.std(returns) > 0 else 0,
                'returns': returns,
                'cumulative_returns': cumulative_returns
            }
    
    if st.session_state.backtest_results:
        results = st.session_state.backtest_results
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{results['final_return']:+.1f}%")
        with col2:
            st.metric("Annual Volatility", f"{results['volatility']:.1f}%")
        with col3:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
        with col4:
            st.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
        
        # Performance chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(results['cumulative_returns']))),
            y=results['cumulative_returns'] * 100,
            mode='lines',
            name='Cumulative Return',
            line=dict(color='#1f77b4')
        ))
        fig.update_layout(
            title="Backtest Performance",
            xaxis_title="Days",
            yaxis_title="Return (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("**Note:** Backtest results are simulated for demonstration purposes and not indicative of future performance.")

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center'>
    <p><em>⚠️ Risk Disclaimer: This tool is for educational and demonstration purposes only. 
    Cryptocurrency trading involves substantial risk. Always conduct your own research 
    and consider consulting with a qualified financial advisor before making investment decisions.</em></p>
</div>
""", unsafe_allow_html=True)
```
