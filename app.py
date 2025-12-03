import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Dollar Smile Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Morgan Stanley Dollar Smile Theory Dashboard")
st.markdown("Visualizing the Dollar Smile theory using live market data")

# Sidebar for metrics
st.sidebar.header("Current Metrics")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_market_data():
    """Fetch live market data for DXY, SPY, ACWX, and VIX"""
    try:
        # Fetch data for the last 90 days to calculate trailing metrics
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        # Fetch tickers
        dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date, progress=False)
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
        acwx = yf.download("ACWX", start=start_date, end=end_date, progress=False)
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        
        return {
            'dxy': dxy,
            'spy': spy,
            'acwx': acwx,
            'vix': vix
        }
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def get_latest_value(df, column='Adj Close'):
    """Get the latest non-null value from a dataframe"""
    if df.empty:
        return None
    if column in df.columns:
        series = df[column].dropna()
    else:
        series = df.dropna()
    if series.empty:
        return None
    return series.iloc[-1]

def calculate_zscore(series, window=60):
    """Calculate rolling Z-score"""
    if len(series) < window:
        window = len(series)
    if window < 2:
        return pd.Series([0] * len(series), index=series.index)
    
    rolling_mean = series.rolling(window=window, min_periods=min(10, window//3)).mean()
    rolling_std = series.rolling(window=window, min_periods=min(10, window//3)).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    zscore = (series - rolling_mean) / rolling_std
    return zscore.fillna(0)

def calculate_smile_coordinates(dxy_data, spy_data, acwx_data, vix_data):
    """Calculate X and Y coordinates for the Dollar Smile"""
    
    # Get adjusted close prices
    dxy_close = dxy_data['Adj Close'] if 'Adj Close' in dxy_data.columns else dxy_data.iloc[:, -1]
    spy_close = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns else spy_data.iloc[:, -1]
    acwx_close = acwx_data['Adj Close'] if 'Adj Close' in acwx_data.columns else acwx_data.iloc[:, -1]
    vix_close = vix_data['Adj Close'] if 'Adj Close' in vix_data.columns else vix_data.iloc[:, -1]
    
    # Align all series by date
    aligned = pd.DataFrame({
        'DXY': dxy_close,
        'SPY': spy_close,
        'ACWX': acwx_close,
        'VIX': vix_close
    }).dropna()
    
    if len(aligned) < 10:
        return None, None, None
    
    # Calculate SPY/ACWX ratio
    growth_ratio = aligned['SPY'] / aligned['ACWX']
    
    # Calculate Z-scores
    dxy_zscore = calculate_zscore(aligned['DXY'])
    growth_ratio_zscore = calculate_zscore(growth_ratio)
    
    # Calculate X coordinate
    # If VIX > 25, shift X to negative (Left Side/Fear)
    x_coords = []
    for idx, vix_val in enumerate(aligned['VIX']):
        if pd.notna(vix_val) and vix_val > 25:
            # Shift to negative (fear side)
            x_coords.append(-abs(growth_ratio_zscore.iloc[idx]))
        else:
            x_coords.append(growth_ratio_zscore.iloc[idx])
    
    x_coords = pd.Series(x_coords, index=aligned.index)
    
    # Y coordinate is Z-score of DXY
    y_coords = dxy_zscore
    
    # Create result dataframe
    result_df = pd.DataFrame({
        'Date': aligned.index,
        'X': x_coords,
        'Y': y_coords,
        'VIX': aligned['VIX'],
        'DXY': aligned['DXY'],
        'Growth_Ratio': growth_ratio
    })
    
    return result_df, aligned['VIX'].iloc[-1], aligned['DXY'].iloc[-1], growth_ratio.iloc[-1]

def create_smile_plot(df, current_vix, current_dxy, current_growth_ratio):
    """Create the Dollar Smile visualization"""
    
    if df is None or len(df) < 2:
        st.error("Insufficient data to create visualization")
        return None
    
    # Get last 60 days for trail
    df_trail = df.tail(60).copy()
    current_point = df.iloc[-1]
    
    # Create static parabola background (theoretical smile)
    # Parabola: y = a*x^2 + b, where a > 0 creates a smile
    x_range = np.linspace(df['X'].min() - 0.5, df['X'].max() + 0.5, 200)
    # Standard smile: y = x^2 (shifted and scaled to fit data)
    # Adjust parabola to fit the data range
    x_center = (df['X'].min() + df['X'].max()) / 2
    y_min = df['Y'].min()
    y_max = df['Y'].max()
    
    # Create a smile curve: y = a*(x - x_center)^2 + y_offset
    # Scale to fit the data
    a = 0.3  # Curvature parameter
    y_offset = (y_min + y_max) / 2
    y_smile = a * (x_range - x_center)**2 + y_offset
    
    # Create figure
    fig = go.Figure()
    
    # Add theoretical smile curve (parabola)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_smile,
        mode='lines',
        name='Theoretical Smile',
        line=dict(color='gray', width=2, dash='dash'),
        opacity=0.7
    ))
    
    # Add trail of last 60 days
    fig.add_trace(go.Scatter(
        x=df_trail['X'],
        y=df_trail['Y'],
        mode='lines+markers',
        name='60-Day Trail',
        line=dict(color='blue', width=1),
        marker=dict(size=4, color='blue', opacity=0.6),
        hovertemplate='Date: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
        text=df_trail['Date'].dt.strftime('%Y-%m-%d')
    ))
    
    # Add current position (highlighted)
    fig.add_trace(go.Scatter(
        x=[current_point['X']],
        y=[current_point['Y']],
        mode='markers+text',
        name='Current Position',
        marker=dict(size=20, color='red', symbol='diamond'),
        text=['Current'],
        textposition='top center',
        hovertemplate=f'Date: {current_point["Date"].strftime("%Y-%m-%d")}<br>X: {current_point["X"]:.3f}<br>Y: {current_point["Y"]:.3f}<br>VIX: {current_point["VIX"]:.2f}<br>DXY: {current_point["DXY"]:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Dollar Smile Visualization',
        xaxis_title='Smile X-Coordinate (Growth/Risk Sentiment)',
        yaxis_title='Smile Y-Coordinate (USD Strength - DXY Z-score)',
        hovermode='closest',
        template='plotly_white',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Add zero lines
    fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.3)
    fig.add_vline(x=0, line_dash="dot", line_color="black", opacity=0.3)
    
    return fig

# Main app logic
data = fetch_market_data()

if data is not None:
    # Calculate coordinates
    result_df, current_vix, current_dxy, current_growth_ratio = calculate_smile_coordinates(
        data['dxy'], data['spy'], data['acwx'], data['vix']
    )
    
    if result_df is not None:
        # Display metrics in sidebar
        st.sidebar.metric("VIX", f"{current_vix:.2f}")
        st.sidebar.metric("USD Level (DXY)", f"{current_dxy:.2f}")
        st.sidebar.metric("Growth Ratio (SPY/ACWX)", f"{current_growth_ratio:.4f}")
        
        # Display current coordinates
        current_x = result_df.iloc[-1]['X']
        current_y = result_df.iloc[-1]['Y']
        st.sidebar.markdown("---")
        st.sidebar.subheader("Current Position")
        st.sidebar.metric("X-Coordinate", f"{current_x:.3f}")
        st.sidebar.metric("Y-Coordinate", f"{current_y:.3f}")
        
        # Determine regime
        if current_vix > 25:
            regime = "Left Side (Fear)"
            st.sidebar.warning(f"⚠️ {regime}")
        elif current_x > 0.5 and current_y > 0.3:
            regime = "Right Side (US Exceptionalism)"
            st.sidebar.success(f"✅ {regime}")
        elif current_x < -0.2 and current_y > 0.3:
            regime = "Left Side (Crisis USD Bid)"
            st.sidebar.error(f"🔴 {regime}")
        else:
            regime = "Transitional/Mixed"
            st.sidebar.info(f"ℹ️ {regime}")
        
        # Create and display plot
        fig = create_smile_plot(result_df, current_vix, current_dxy, current_growth_ratio)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display data table (optional, collapsible)
        with st.expander("View Historical Data"):
            st.dataframe(
                result_df[['Date', 'X', 'Y', 'VIX', 'DXY', 'Growth_Ratio']].tail(30),
                use_container_width=True
            )
    else:
        st.error("Could not calculate coordinates. Please check data availability.")
else:
    st.error("Failed to fetch market data. Please try again later.")

# Footer
st.markdown("---")
st.markdown("**Data Source:** Yahoo Finance via yfinance | **Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
