import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model

# Title
st.title("📈 Stock Analyzer App")

# Sidebar input
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Download data
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data found. Please check your ticker symbol or date range.")
    st.stop()

# Show raw data
st.subheader(f"Raw Data for {ticker}")
st.dataframe(data.tail())

# Plot closing price
st.subheader("Closing Price Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close Price"))
st.plotly_chart(fig, use_container_width=True)

# Moving Averages
st.subheader("Moving Averages")
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
fig_ma.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA 20", line=dict(color='orange')))
fig_ma.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA 50", line=dict(color='green')))
st.plotly_chart(fig_ma, use_container_width=True)

# Daily returns and volatility (GARCH)
st.subheader("Volatility Modeling (GARCH)")
returns = 100 * data["Close"].pct_change().dropna()

if len(returns) < 50:
    st.warning("Not enough data for GARCH model. Select a longer date range.")
else:
    model = arch_model(returns, vol='Garch', p=1, q=1)
    result = model.fit(disp='off')
    forecast = result.forecast(horizon=5)
    st.write("Forecasted Variance (next 5 days):")
    st.dataframe(forecast.variance[-1:])

# Value at Risk (VaR)
st.subheader("Value at Risk (VaR)")
confidence_level = 0.95
VaR = -np.percentile(returns, (1 - confidence_level) * 100)
st.write(f"One-day 95% Value at Risk: {VaR:.2f}%")
