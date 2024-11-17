import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the date range
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)

st.title('Stock Market Price Predictor')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker (e.g., AMZN, MSFT, AAPL)', 'AMZN')
df = yf.download(user_input, start=start, end=end)

if df.empty:
    st.error("No data found for the given stock ticker. Please check the ticker symbol and try again.")
else:
    st.subheader('Stock Data')
    st.write(df.describe())

    # Calculating Moving Averages and Additional Features
    df['MA_for_100_days'] = df['Close'].rolling(window=100).mean()
    df['MA_for_200_days'] = df['Close'].rolling(window=200).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Plotting Closing Price and Moving Averages
    st.subheader('Closing Price and Moving Averages')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.plot(df['MA_for_100_days'], label='100-day MA', color='orange')
    plt.plot(df['MA_for_200_days'], label='200-day MA', color='green')
    plt.title('Closing Price and Moving Averages')
    plt.xlabel('Years')
    plt.ylabel('Price Values')
    plt.legend()
    st.pyplot(fig)

    # Plotting the Closing Price vs Time Chart
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.title('Closing Price vs Time')
    plt.xlabel('Years')
    plt.ylabel('Price Values')
    plt.legend()
    st.pyplot(fig)

    # Plotting Closing Price vs Time Chart with 100-day MA
    st.subheader('Closing Price vs Time Chart with 100-day MA')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['MA_for_100_days'], label='MA for 100 days', color='orange')
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.title('Closing Price vs Time with 100-day MA')
    plt.xlabel('Years')
    plt.ylabel('Price Values')
    plt.legend()
    st.pyplot(fig)

    # Plotting Closing Price vs Time Chart with 200-day MA
    st.subheader('Closing Price vs Time Chart with 200-day MA')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['MA_for_200_days'], label='MA for 200 days', color='green')
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.title('Closing Price vs Time with 200-day MA')
    plt.xlabel('Years')
    plt.ylabel('Price Values')
    plt.legend()
    st.pyplot(fig)

    # Plotting Closing Price vs Time Chart with 100-day and 200-day MA
    st.subheader('Closing Price vs Time Chart with 100-day and 200-day MA')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['MA_for_100_days'], label='MA for 100 days', color='orange')
    plt.plot(df['MA_for_200_days'], label='MA for 200 days', color='green')
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.title('Closing Price vs Time with 100-day and 200-day MA')
    plt.xlabel('Years')
    plt.ylabel('Price Values')
    plt.legend()
    st.pyplot(fig)

    # Preprocess Data
    df.dropna(inplace=True)
    x_test = df[['Close', 'SMA_20', 'SMA_50']]

    if x_test.empty or len(x_test) < 100:
        st.error("Not enough data available after preprocessing. Please try with a different stock ticker.")
    else:
        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test)

        # Creating training data
        x_data, y_data = [], []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i])
            y_data.append(scaled_data[i, 0])  # Predicting the Close price

        x_data, y_data = np.array(x_data), np.array(y_data)

        # Debugging Data Shapes
        st.text(f"Data shapes: x_data={x_data.shape}, y_data={y_data.shape}")

        # Defining the LSTM Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_data.shape[1], x_data.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_data, y_data, epochs=25, batch_size=32)

        # Save Model
        model.save("Latest_Stock_Price_model.keras")

        # Predictions
        predictions = model.predict(x_data)
        inv_pre = scaler.inverse_transform(
            np.concatenate((predictions, np.zeros((predictions.shape[0], x_test.shape[1] - 1))), axis=1))[:, 0]
        inv_y_test = scaler.inverse_transform(
            np.concatenate((y_data.reshape(-1, 1), np.zeros((y_data.shape[0], x_test.shape[1] - 1))), axis=1))[:, 0]

        # Debugging Predictions
        st.text(f"Predictions shape: {predictions.shape}")

        # Prepare Plotting Data
        plotting_data = pd.DataFrame({
            'Original': inv_y_test,
            'Predicted': inv_pre
        }, index=df.index[-len(inv_y_test):])

        st.subheader("Original vs Predicted Values")
        st.write(plotting_data)

        # Plotting Predictions
        st.subheader("Closing Price: Original vs Predicted")
        fig = plt.figure(figsize=(15, 6))
        plt.plot(df['Close'], label="Original Close Price", color='blue')
        plt.plot(plotting_data['Predicted'], label="Predicted Close Price", color='orange')
        plt.xlabel('Years')
        plt.ylabel('Price Values')
        plt.legend()
        st.pyplot(fig)
