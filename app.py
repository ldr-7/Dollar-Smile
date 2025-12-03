import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="MS Dollar Smile Tracker", layout="wide")
st.title("💵 Live Morgan Stanley 'Dollar Smile' Tracker")

# --- 1. DATA FETCHING (Cached 5 min) ---
@st.cache_data(ttl=300) 
def fetch_smile_data():
    """
    Fetches live data for:
    - USD Index (DX-Y.NYB) -> Y-Axis (USD Strength)
    - SPY (US Stocks) & ACWX (Global ex-US) -> X-Axis Growth Proxy
    - VIX -> X-Axis Fear Proxy
    """
    end_date = datetime.now()
    # Fetch 2 years to ensure we have enough data for a 1-year rolling Z-score
    start_date = end_date - timedelta(days=730) 

    tickers = {
        'USD': 'DX-Y.NYB',
        'US_Eq': 'SPY',
        'Global_Eq': 'ACWX',
        'VIX': '^VIX'
    }

    try:
        # Fetch data
        raw_data = yf.download(list(tickers.values()), start=start_date, end=end_date, progress=False)
        
        # Handle yfinance structure (Access 'Close' price)
        if 'Close' in raw_data.columns:
            data = raw_data['Close']
        else:
            data = raw_data

        # Flatten MultiIndex columns if necessary
        if isinstance(data.columns, pd.MultiIndex):
            # Map ticker symbols to friendly names
            # Note: yfinance might return column names as the Ticker string
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        # Rename columns to friendly names
        # Invert the dictionary to map Symbol -> Friendly Name
        inv_map = {v: k for k, v in tickers.items()}
        data = data.rename(columns=inv_map)
        
        # Ensure all required columns exist (handling potential API failures for specific tickers)
        required = ['USD', 'US_Eq', 'Global_Eq', 'VIX']
        if not all(col in data.columns for col in required):
            st.error("Missing data columns from API. Please try again later.")
            return pd.DataFrame()

        return data.dropna()
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()

df = fetch_smile_data()

if df.empty:
    st.stop()

# --- 2. CALCULATIONS ---

# Rolling window for Z-Scores (1 year / 252 trading days)
window = 252

# A. Y-Axis: USD Strength (Z-Score)
# We compare current USD price to its 1-year average
df['USD_Mean'] = df['USD'].rolling(window).mean()
df['USD_Std'] = df['USD'].rolling(window).std()
df['Y_Coord'] = (df['USD'] - df['USD_Mean']) / df['USD_Std']

# B. X-Axis: Economic Conditions
# Step 1: Calculate Relative Growth (US vs Global)
df['Rel_Growth'] = df['US_Eq'] / df['Global_Eq']
df['Growth_Mean'] = df['Rel_Growth'].rolling(window).mean()
df['Growth_Std'] = df['Rel_Growth'].rolling(window).std()
df['Growth_Z'] = (df['Rel_Growth'] - df['Growth_Mean']) / df['Growth_Std']

# Step 2: Determine Active X-Coordinate based on Regime
def get_smile_x(row):
    # FEAR REGIME (Left Side)
    # If VIX is high (>25), logic shifts to "Safety".
    # We ignore growth metrics and force the X coordinate negative.
    if row['VIX'] > 25:
        # Calculate how severe the fear is
        severity = (row['VIX'] - 25) / 5
        # Cap the shift so it doesn't go off chart, but ensure it's on the left (< -1)
        return -1.5 - severity 
    
    # NORMAL/GROWTH REGIME (Middle to Right)
    # Follow the US vs Global growth ratio Z-score
    return row['Growth_Z']

df['X_Coord'] = df.apply(get_smile_x, axis=1)

# Get most recent data points
current = df.iloc[-1]
prev_week = df.iloc[-5] if len(df) > 5 else df.iloc[0]

# --- 3. DASHBOARD LAYOUT ---

# Create two columns: Main Chart (Left) and Metrics Sidebar (Right)
col_chart, col_metrics = st.columns([3, 1])

with col_metrics:
    st.subheader("Current Metrics")
    
    # METRIC 1: Regime Classification
    regime = "Unknown"
    regime_color = "gray"
    if current['VIX'] > 25:
        regime = "FEAR / RISK OFF"
        regime_color = "red"
        st.error(f"⚠️ {regime}")
        st.caption("Dollar strengthening due to safety seeking.")
    elif current['X_Coord'] > 0.5:
        regime = "US EXCEPTIONALISM"
        regime_color = "green"
        st.success(f"🇺🇸 {regime}")
        st.caption("Dollar strengthening due to US growth.")
    else:
        regime = "GLOBAL SYNC / MUDDLING"
        regime_color = "orange"
        st.warning(f"⚖️ {regime}")
        st.caption("Dollar weakening as capital flows global.")

    st.divider()

    # METRIC 2: Data Points
    st.metric("VIX (Fear)", f"{current['VIX']:.2f}", delta=f"{current['VIX'] - prev_week['VIX']:.2f}", delta_color="inverse")
    st.metric("DXY (USD Index)", f"{current['USD']:.2f}", delta=f"{current['USD'] - prev_week['USD']:.2f}")
    st.metric("US/Global Ratio", f"{current['Rel_Growth']:.3f}", delta=f"{current['Growth_Z']:.2f} σ")
    
    st.divider()
    st.caption(f"Active Coordinates:\nX: {current['X_Coord']:.2f} | Y: {current['Y_Coord']:.2f}")


with col_chart:
    # --- VISUALIZATION ---
    fig = go.Figure()

    # 1. Theoretical Smile Curve (Background)
    x_theory = np.linspace(-4, 4, 100)
    # Parabola: y = x^2 (roughly normalized)
    y_theory = 0.5 * (x_theory**2) - 1.0 
    
    fig.add_trace(go.Scatter(
        x=x_theory, y=y_theory,
        mode='lines',
        name='Theoretical Smile',
        line=dict(color='lightgrey', dash='dash'),
        hoverinfo='skip'
    ))

    # 2. Historical Trail (Last 60 Days)
    trail_df = df.tail(60)
    fig.add_trace(go.Scatter(
        x=trail_df['X_Coord'],
        y=trail_df['Y_Coord'],
        mode='lines+markers',
        name='Last 60 Days',
        marker=dict(size=4, opacity=0.5, color='gray'),
        line=dict(width=1, color='gray'),
        text=trail_df.index.strftime('%Y-%m-%d'),
        hovertemplate='<b>Date</b>: %{text}<br><b>X</b>: %{x:.2f}<br><b>Y</b>: %{y:.2f}'
    ))

    # 3. Current Position (Red Diamond)
    fig.add_trace(go.Scatter(
        x=[current['X_Coord']],
        y=[current['Y_Coord']],
        mode='markers+text',
        name='CURRENT',
        marker=dict(size=18, color='red', symbol='diamond', line=dict(width=2, color='black')),
        text=["📍 YOU ARE HERE"],
        textposition="top center"
    ))

    # Annotations for the "Smile" Zones
    fig.add_annotation(x=-3, y=3, text="Recession / Fear<br>(Strong USD)", showarrow=False, font=dict(color="red", size=12))
    fig.add_annotation(x=3, y=3, text="US Stronger Growth<br>(Strong USD)", showarrow=False, font=dict(color="green", size=12))
    fig.add_annotation(x=0, y=-2, text="Global Sync Growth<br>(Weak USD)", showarrow=False, font=dict(color="orange", size=12))

    fig.update_layout(
        title="Active Dollar Smile Positioning",
        xaxis_title="<-- Risk Aversion (VIX) | US vs Global Growth (Rel. Strength) -->",
        yaxis_title="USD Strength (Z-Score)",
        height=600,
        showlegend=False,
        xaxis=dict(range=[-4, 4], zeroline=True), 
        yaxis=dict(range=[-3, 4], zeroline=True)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- 4. DATA TABLE ---
with st.expander("📊 View Historical Data Table"):
    st.dataframe(df[['USD', 'US_Eq', 'Global_Eq', 'VIX', 'X_Coord', 'Y_Coord']].sort_index(ascending=False))
