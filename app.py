import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model

# -----------------------------
# Title
# -----------------------------
st.title("📈 Stock Analyzer App")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data found. Please check your ticker or date range.")
    st.stop()

# -----------------------------
# Show Raw Data
# -----------------------------
st.subheader(f"Raw Data for {ticker}")
st.dataframe(data.tail())

# -----------------------------
# Closing Price Chart
# -----------------------------
st.subheader("📉 Closing Price Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close Price"))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Moving Averages
# -----------------------------
st.subheader("📊 Moving Averages")
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
fig_ma.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA 20", line=dict(color='orange')))
fig_ma.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA 50", line=dict(color='green')))
st.plotly_chart(fig_ma, use_container_width=True)

# -----------------------------
# Technical Indicators
# -----------------------------
st.subheader("📐 Technical Indicators")

# RSI
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
data["RSI"] = rsi

# MACD
exp1 = data["Close"].ewm(span=12, adjust=False).mean()
exp2 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = exp1 - exp2
data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

# Bollinger Bands
data["BB_MA"] = data["Close"].rolling(window=20).mean()
data["BB_Upper"] = data["BB_MA"] + 2 * data["Close"].rolling(window=20).std()
data["BB_Lower"] = data["BB_MA"] - 2 * data["Close"].rolling(window=20).std()

# Plot RSI
st.write("**RSI (Relative Strength Index)**")
st.line_chart(data["RSI"])

# Plot MACD
st.write("**MACD (Moving Average Convergence Divergence)**")
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"))
fig_macd.add_trace(go.Scatter(x=data.index, y=data["Signal"], name="Signal", line=dict(dash="dot")))
st.plotly_chart(fig_macd, use_container_width=True)

# Plot Bollinger Bands
st.write("**Bollinger Bands**")
fig_bb = go.Figure()
fig_bb.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
fig_bb.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], name="Upper Band", line=dict(color='red')))
fig_bb.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], name="Lower Band", line=dict(color='blue')))
st.plotly_chart(fig_bb, use_container_width=True)

# -----------------------------
# Volatility Modeling (GARCH)
# -----------------------------
st.subheader("📉 Volatility Modeling (GARCH)")
returns = 100 * data["Close"].pct_change().dropna()

if len(returns) < 50:
    st.warning("Not enough data for GARCH model. Select a longer date range.")
else:
    model = arch_model(returns, vol='Garch', p=1, q=1)
    result = model.fit(disp='off')
    forecast = result.forecast(horizon=5)
    st.write("Forecasted Variance (next 5 days):")
    st.dataframe(forecast.variance[-1:])

# -----------------------------
# Value at Risk (VaR)
# -----------------------------
st.subheader("📌 Value at Risk (VaR)")
confidence_level = 0.95
VaR = -np.percentile(returns, (1 - confidence_level) * 100)
st.write(f"One-day 95% Value at Risk: {VaR:.2f}%")