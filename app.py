from __future__ import annotations
import os
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# ----------------------------
# Utilities
# ----------------------------

@dataclass
class TickerSpec:
    label: str
    tickers: List[str]
    invert: bool = False  # use for FX where we want USD strength (e.g., EURUSD inverted)


def fetch_first_working(tickers: List[str], start: str) -> pd.Series:
    """Try a list of tickers until one works; return Adj Close series."""
    for t in tickers:
        try:
            df = yf.download(t, start=start, progress=False)["Adj Close"].dropna()
            if df.size:
                df.name = t
                return df
        except Exception:
            pass
    return pd.Series(dtype=float)


def pct_change_period(s: pd.Series, days: int) -> pd.Series:
    return s.pct_change(days)


def zscore(s: pd.Series, lookback: int) -> pd.Series:
    roll = s.rolling(lookback, min_periods=max(10, lookback // 5))
    return (s - roll.mean()) / (roll.std(ddof=0) + 1e-9)


def standardize(series_list: List[pd.Series], lookback: int, invert_flags: List[bool]) -> List[pd.Series]:
    out = []
    for s, inv in zip(series_list, invert_flags):
        zs = zscore(s, lookback)
        out.append(-zs if inv else zs)
    return out


def safe_align(cols: Dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.concat(cols, axis=1).dropna(how="all")
    return df.dropna()  # require full data for composites


# ----------------------------
# Data plumbing
# ----------------------------

START_DEFAULT = (dt.date.today() - dt.timedelta(days=365 * 6)).strftime("%Y-%m-%d")

GROWTH_SPECS: List[TickerSpec] = [
    TickerSpec("US Equities (SPY)", ["SPY"]),
    TickerSpec("Global ex-US (ACWX)", ["ACWX"]),
    TickerSpec("Credit vs UST (HYG/IEF)", ["HYG"], invert=False),  # we'll divide by IEF later
    TickerSpec("Copper/Gold (HG/GC)", ["HG=F"], invert=False),     # will divide by GC later
    TickerSpec("Volatility (VIX) — inverted", ["^VIX"], invert=True),
]

USD_SPECS: List[TickerSpec] = [
    TickerSpec("DXY (or fallback UUP)", ["DX-Y.NYB", "^DXY", "UUP"]),
    TickerSpec("EURUSD (invert)", ["EURUSD=X"], invert=True),
    TickerSpec("JPYUSD (invert; use USDJPY and invert appropriately)", ["JPY=X"], invert=False),
]

# Special cases we build from pairs
PAIR_HELPERS = {
    "HYG/IEF": ("HYG", "IEF"),
    "HG/GC": ("HG=F", "GC=F"),
    "USDJPY to JPYUSD": ("USDJPY=X",),  # if needed
}


def get_market_frame(start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (growth_df, usd_df) daily levels for each component."""
    cols_g: Dict[str, pd.Series] = {}

    # Base series
    spy = fetch_first_working(["SPY"], start)
    acwx = fetch_first_working(["ACWX"], start)
    hyg = fetch_first_working(["HYG"], start)
    ief = fetch_first_working(["IEF"], start)
    hg = fetch_first_working(["HG=F"], start)
    gc = fetch_first_working(["GC=F"], start)
    vix = fetch_first_working(["^VIX"], start)

    # Ratios
    hyg_ief = (hyg / ief).dropna()
    hg_gc = (hg / gc).dropna()

    cols_g["SPY"] = spy
    cols_g["ACWX"] = acwx
    cols_g["HYG/IEF"] = hyg_ief
    cols_g["HG/GC"] = hg_gc
    cols_g["VIX_inv"] = vix  # we'll invert via z-score later

    growth_df = safe_align(cols_g)

    # USD block
    dxy = fetch_first_working(["DX-Y.NYB", "^DXY"], start)
    if dxy.empty:
        dxy = fetch_first_working(["UUP"], start)  # ETF fallback

    eurusd = fetch_first_working(["EURUSD=X"], start)
    usdjpy = fetch_first_working(["JPY=X"], start)  # this is USD/JPY typically; name is confusing on yfinance
    # yfinance "JPY=X" = USDJPY (units JPY per USD). To get JPYUSD, invert.
    jpyusd = 1.0 / usdjpy if not usdjpy.empty else pd.Series(dtype=float)

    cols_u: Dict[str, pd.Series] = {
        "DXY_or_UUP": dxy,
        "EURUSD_inv": eurusd,  # will invert in z-score stage
        "JPYUSD_inv": jpyusd,  # will invert in z-score stage (so ends up aligned with USD strength)
    }
    usd_df = safe_align(cols_u)

    return growth_df, usd_df


# ----------------------------
# Composite construction
# ----------------------------

def composite_scores(growth_df: pd.DataFrame,
                     usd_df: pd.DataFrame,
                     z_lookback_days: int = 252,
                     perf_window_days: int = 63,
                     weights_growth: Dict[str, float] | None = None,
                     weights_usd: Dict[str, float] | None = None) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Compute X (growth) and Y (USD) composite scores.
    Returns (x_series, y_series, joined_frame)
    """
    if weights_growth is None:
        weights_growth = {c: 1.0 for c in growth_df.columns}
    if weights_usd is None:
        weights_usd = {c: 1.0 for c in usd_df.columns}

    # Compute performance windows first (e.g., 3-month returns), then z-score
    g_feats = {}
    for c in growth_df.columns:
        perf = pct_change_period(growth_df[c], perf_window_days)
        inv = ("VIX" in c)  # invert VIX behaviour
        g_feats[c] = (-zscore(perf, z_lookback_days) if inv else zscore(perf, z_lookback_days)) * weights_growth.get(c, 1.0)

    u_feats = {}
    for c in usd_df.columns:
        perf = pct_change_period(usd_df[c], perf_window_days)
        inv = ("inv" in c)  # invert the inverses so higher = stronger USD
        u_feats[c] = (-zscore(perf, z_lookback_days) if inv else zscore(perf, z_lookback_days)) * weights_usd.get(c, 1.0)

    g_mat = pd.concat(g_feats, axis=1).dropna()
    u_mat = pd.concat(u_feats, axis=1).dropna()
    both = g_mat.join(u_mat, how="inner")

    x = g_mat.sum(axis=1) / max(1e-9, sum(abs(weights_growth.get(c, 1.0)) for c in g_mat.columns))
    y = u_mat.sum(axis=1) / max(1e-9, sum(abs(weights_usd.get(c, 1.0)) for c in u_mat.columns))

    x, y = x.align(y, join="inner")
    frame = pd.DataFrame({"GrowthX": x, "USDY": y})
    return x, y, frame


def fit_smile_curve(x: pd.Series, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Fit y ~ a*x^2 + b using polyfit and return curve points (xs, ys)."""
    if len(x) < 30:
        # not enough data, return simple parabola
        xs = np.linspace(-2, 2, 200)
        ys = xs ** 2
        return xs, ys
    try:
        coeffs = np.polyfit(x, y, deg=2)
        xs = np.linspace(x.min(), x.max(), 400)
        ys = coeffs[0] * xs ** 2 + coeffs[1] * xs + coeffs[2]
        return xs, ys
    except Exception:
        xs = np.linspace(-2, 2, 200)
        ys = xs ** 2
        return xs, ys


def classify_regime(x_now: float, y_now: float) -> str:
    # Simple regime map
    if x_now <= -0.4 and y_now >= 0.4:
        return "Crisis USD bid (left tail)"
    if x_now >= 0.5 and y_now >= 0.3:
        return "US Exceptionalism (right tail)"
    if x_now >= 0.2 and y_now <= -0.2:
        return "Global reflation / USD soft (smile trough)"
    return "Transitional / Mixed"


# ----------------------------
# Dash app
# ----------------------------

app = Dash(__name__)
app.title = "Interactive Dollar Smile"

app.layout = html.Div([
    html.H1("Interactive Dollar Smile"),
    html.Div("Proxies via yfinance. Adjust lookback and windows to taste."),
    html.Br(),

    html.Div([
        html.Div([
            html.Label("Lookback (years)"),
            dcc.Slider(id="lookback_years", min=2, max=10, step=1, value=6,
                       marks={i: str(i) for i in range(2, 11)}),
            html.Br(),
            html.Label("Performance window (days)"),
            dcc.Slider(id="perf_days", min=21, max=126, step=7, value=63,
                       marks={21: "1m", 42: "2m", 63: "3m", 84: "4m", 105: "5m", 126: "6m"}),
            html.Br(),
            html.Label("Z-score lookback (days)"),
            dcc.Slider(id="z_lookback", min=126, max=756, step=21, value=252,
                       marks={126: "~6m", 252: "~1y", 504: "~2y", 756: "~3y"}),
        ], style={"flex": 1, "minWidth": 280, "paddingRight": "24px"}),

        html.Div([
            html.Div(id="regime_box", style={"fontSize": "20px", "marginBottom": "10px", "fontWeight": "600"}),
            dcc.Graph(id="smile_chart"),
        ], style={"flex": 3})
    ], style={"display": "flex", "flexWrap": "wrap"}),

    html.Br(),
    html.Div([
        html.Div([
            html.H3("Growth / Risk Sentiment Composite (X)"),
            dcc.Graph(id="ts_growth"),
        ], style={"flex": 1, "minWidth": 360, "paddingRight": "12px"}),
        html.Div([
            html.H3("USD Strength Composite (Y)"),
            dcc.Graph(id="ts_usd"),
        ], style={"flex": 1, "minWidth": 360, "paddingLeft": "12px"}),
    ], style={"display": "flex", "flexWrap": "wrap"}),

    html.Br(),
    html.Details([
        html.Summary("Data & Methodology"),
        html.Ul([
            html.Li("Growth (X): 3m perf z-scores averaged for SPY, ACWX, HYG/IEF, Copper/Gold, and inverted VIX."),
            html.Li("USD (Y): 3m perf z-scores averaged for DXY/UUP, inverted EURUSD, inverted JPYUSD."),
            html.Li("Smile curve: quadratic fit over historical X vs Y."),
            html.Li("Regimes: rule-based bands on current (X, Y)."),
        ])
    ])
])


@app.callback(
    [Output("smile_chart", "figure"),
     Output("ts_growth", "figure"),
     Output("ts_usd", "figure"),
     Output("regime_box", "children")],
    [Input("lookback_years", "value"),
     Input("perf_days", "value"),
     Input("z_lookback", "value")]
)

def update(lookback_years: int, perf_days: int, z_lb: int):
    start = (dt.date.today() - dt.timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

    growth_df, usd_df = get_market_frame(start)
    if growth_df.empty or usd_df.empty:
        msg = "One or more required series failed to download. Try increasing lookback or check tickers."
        empty_fig = go.Figure().add_annotation(text=msg, showarrow=False, x=0.5, y=0.5)
        return empty_fig, empty_fig, empty_fig, "Data error"

    x, y, frame = composite_scores(growth_df, usd_df, z_lookback_days=z_lb, perf_window_days=perf_days)

    xs, ys = fit_smile_curve(frame["GrowthX"], frame["USDY"])

    # Build scatter with color by time
    dates = frame.index
    colors = (dates - dates.min()).days

    scatter = go.Scatter(
        x=frame["GrowthX"], y=frame["USDY"], mode="markers",
        marker=dict(size=6, color=colors, showscale=True, colorbar=dict(title="days")),
        name="History"
    )

    curve = go.Scatter(x=xs, y=ys, mode="lines", name="Fitted smile (quad)")

    # Today's point
    x_now = frame["GrowthX"].iloc[-1]
    y_now = frame["USDY"].iloc[-1]
    today_pt = go.Scatter(x=[x_now], y=[y_now], mode="markers+text",
                          marker=dict(size=14, symbol="diamond"),
                          text=["Today"], textposition="top center",
                          name="Today")

    fig_smile = go.Figure(data=[scatter, curve, today_pt])
    fig_smile.update_layout(
        xaxis_title="Growth / Risk Sentiment (X)",
        yaxis_title="USD Strength (Y)",
        title="Dollar Smile: composites vs fitted curve",
        template="plotly_white",
        hovermode="closest"
    )

    # Time series
    ts_g = go.Figure(data=[go.Scatter(x=frame.index, y=frame["GrowthX"], mode="lines", name="GrowthX")])
    ts_g.add_hline(y=0, line=dict(dash="dot"))
    ts_g.update_layout(title="Growth / Risk Sentiment (X)", template="plotly_white")

    ts_u = go.Figure(data=[go.Scatter(x=frame.index, y=frame["USDY"], mode="lines", name="USDY")])
    ts_u.add_hline(y=0, line=dict(dash="dot"))
    ts_u.update_layout(title="USD Strength (Y)", template="plotly_white")

    regime = classify_regime(x_now, y_now)
    regime_text = f"Current regime: {regime} — X={x_now:.2f}, Y={y_now:.2f}"

    return fig_smile, ts_g, ts_u, regime_text


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(debug=True, host=host, port=port)

