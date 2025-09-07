from __future__ import annotations
import os
import sys
import argparse
import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Reuse core functions from the Dash app
from app import get_market_frame, composite_scores, fit_smile_curve, classify_regime


def build_figures(lookback_years: int, perf_days: int, z_lb: int, offline: bool = False) -> Tuple[go.Figure, go.Figure, go.Figure, str]:
    if offline:
        # Generate synthetic composites for offline viewing
        rng = np.random.default_rng(42)
        n = 420
        dates = pd.bdate_range(dt.date.today() - dt.timedelta(days=int(n * 1.6)), periods=n)
        x = pd.Series(rng.normal(0, 0.09, size=n)).cumsum()
        noise = rng.normal(0, 0.18, size=n)
        y = 0.55 * (x ** 2) + 0.12 * x + noise
        frame = pd.DataFrame({"GrowthX": x, "USDY": y}, index=dates)
    else:
        start = (dt.date.today() - dt.timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

        growth_df, usd_df = get_market_frame(start)
        if growth_df.empty and usd_df.empty:
            raise RuntimeError("Data download failed. Try increasing lookback or check tickers/network.")

        _, _, frame = composite_scores(growth_df, usd_df, z_lookback_days=z_lb, perf_window_days=perf_days)
        if frame.empty:
            raise RuntimeError("Composite frame is empty after alignment. Try a shorter lookback or different perf window.")

    xs, ys = fit_smile_curve(frame["GrowthX"], frame["USDY"])

    dates = frame.index
    colors = (dates - dates.min()).days

    scatter = go.Scatter(
        x=frame["GrowthX"], y=frame["USDY"], mode="markers",
        marker=dict(size=6, color=colors, showscale=True, colorbar=dict(title="days")),
        name="History"
    )
    curve = go.Scatter(x=xs, y=ys, mode="lines", name="Fitted smile (quad)")

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

    ts_g = go.Figure(data=[go.Scatter(x=frame.index, y=frame["GrowthX"], mode="lines", name="GrowthX")])
    ts_g.add_hline(y=0, line=dict(dash="dot"))
    ts_g.update_layout(title="Growth / Risk Sentiment (X)", template="plotly_white")

    ts_u = go.Figure(data=[go.Scatter(x=frame.index, y=frame["USDY"], mode="lines", name="USDY")])
    ts_u.add_hline(y=0, line=dict(dash="dot"))
    ts_u.update_layout(title="USD Strength (Y)", template="plotly_white")

    regime = classify_regime(x_now, y_now)
    regime_text = f"Current regime: {regime} — X={x_now:.2f}, Y={y_now:.2f}"

    return fig_smile, ts_g, ts_u, regime_text


def export_html(out_path: str, fig_smile: go.Figure, ts_g: go.Figure, ts_u: go.Figure, regime_text: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frag1 = pio.to_html(fig_smile, include_plotlyjs="cdn", full_html=False)
    frag2 = pio.to_html(ts_g, include_plotlyjs=False, full_html=False)
    frag3 = pio.to_html(ts_u, include_plotlyjs=False, full_html=False)

    html = f"""
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Dollar Smile Snapshot</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', Arial, sans-serif; margin: 16px; }}
      .regime {{ font-size: 18px; font-weight: 600; margin: 8px 0 16px; }}
      .row {{ display: flex; flex-wrap: wrap; gap: 16px; }}
      .col {{ flex: 1 1 480px; min-width: 360px; }}
    </style>
  </head>
  <body>
    <h1>Dollar Smile Snapshot</h1>
    <div class=\"regime\">{regime_text}</div>
    <div class=\"col\">{frag1}</div>
    <div class=\"row\">
      <div class=\"col\">{frag2}</div>
      <div class=\"col\">{frag3}</div>
    </div>
    <p style=\"margin-top:24px;color:#666\">Data via yfinance; charts generated offline.</p>
  </body>
  </html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Export Dollar Smile snapshot to HTML")
    parser.add_argument("--lookback-years", type=int, default=6)
    parser.add_argument("--perf-days", type=int, default=63)
    parser.add_argument("--z-lookback", type=int, default=252)
    parser.add_argument("--out", type=str, default="/workspace/output/dollar_smile.html")
    parser.add_argument("--offline", action="store_true", help="Generate synthetic composites offline (no data downloads)")
    args = parser.parse_args(argv)

    fig_smile, ts_g, ts_u, regime_text = build_figures(args.lookback_years, args.perf_days, args.z_lookback, offline=args.offline)
    export_html(args.out, fig_smile, ts_g, ts_u, regime_text)
    print(f"Wrote snapshot to: {args.out}\n{regime_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

