# 📊 Stock Analyzer App

This is an interactive Streamlit web application that allows users to analyze stock data using financial models, technical indicators, and visualizations. The app provides tools for understanding historical stock trends, computing moving averages, modeling volatility using GARCH, and estimating risk through Value at Risk (VaR).

## 🚀 Features

- Fetches historical stock data using `yfinance`
- Interactive UI for stock ticker and date range selection
- Visualizations using Plotly for:
  - Closing price chart
  - Moving Averages (MA20, MA50)
  - Volatility modeling using GARCH(1,1)
- Daily returns calculation
- 5-day GARCH forecast of variance
- Value at Risk (VaR) calculation

## 🛠️ Technologies Used

- Python
- Streamlit
- yfinance
- pandas & numpy
- plotly
- arch (for GARCH modeling)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RohithR1996/stock-analyzer-app.git
   cd stock-analyzer-app
   Create and activate a virtual environment:
   python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
Install the required packages:
pip install -r requirements.txt
Run the app:
streamlit run app.py
