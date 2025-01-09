# :chart_with_upwards_trend: Stock Price Prediction Using Deep Learning

A Streamlit application for predicting stock prices using deep learning models. This application allows users to visualize stock data, analyze trends with moving averages, and predict stock prices based on historical data.

### :bulb: Features
1. Stock Data Retrieval: 
- Fetch stock data for user-defined tickers using Yahoo Finance.
  
2. Data Visualization:
- Visualize stock closing prices over time.
- Plot moving averages (100-day and 200-day).
- Compare original and predicted stock prices.
  
3. Deep Learning:
- Uses LSTM (Long Short-Term Memory) for stock price prediction.
- Scalable and optimized prediction models.
  
4. User-Friendly Interface:
- Simple, interactive interface with Streamlit.

### :gear: Installation and Usage

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```
   
2. Run the app

   ```
   $ streamlit run app.py
   ```

 
### :rocket: How to Use
Input Stock Ticker:
- Enter the stock ticker (e.g., AAPL, MSFT, TSLA) in the text box.

View Stock Data:
- Explore descriptive statistics and charts for stock performance.

Analyze Moving Averages:
- See trends using 100-day and 200-day moving averages.
- Compare performance using visualizations.
  
Predict Stock Prices:
- Leverage deep learning predictions for future prices.
- Compare original vs. predicted prices.
  
### :notebook_with_decorative_cover: Example Tickers
- US Stocks: AMZN, AAPL, TSLA, MSFT
- Indices: ^GSPC (S&P 500), ^DJI (Dow Jones)
- Crypto: BTC-USD, ETH-USD
- Global Stocks: 7203.T (Toyota), RELIANCE.NS

### :wrench: Dependencies
- Python
- Streamlit
- yfinance
- NumPy
- Pandas
- Matplotlib
- Keras
- scikit-learn
