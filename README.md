# Interactive Dollar Smile — Dash + yfinance

What this app does
- Pulls market proxies via yfinance.
- Builds two composite scores:
  - X (Growth/Risk Sentiment): SPY, ACWX, HYG/IEF, Copper/Gold, VIX (inverted)
  - Y (USD Strength): DXY (or UUP fallback), EURUSD (inverted), JPYUSD (inverted)
- Normalizes each component with rolling z-scores and averages to a composite.
- Displays an interactive "dollar smile" scatter (X vs Y), highlights "today",
  fits a quadratic smile curve y = a*x^2 + b, and labels the current regime.
- Includes time-series panels for both composites and configurable lookback.

How to run
1) Install deps
```bash
python3 -m pip install -U -r requirements.txt
```
2) Start server
```bash
python3 app.py
```
3) Open http://127.0.0.1:8050 (or the forwarded port in your IDE).

Notes
- yfinance uses liquid proxies; true macro series would need other APIs.
- You can tweak lookbacks in the UI sliders.
# Dollar-Smile
Interactive Dollar Smile
