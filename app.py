import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
# Add regularization and early stopping
# from tensorflow.keras.callbacks import EarlyStopping
# (Need to update for previous how many days data we are using to predict the next day, here 90 days)
# create a input box to get the number of previous days data to be used to predict the next day
# 90 days is the default value
No_prev_days = st.number_input("Enter the number of previous days data to be used to predict the next day", 30)
#No_prev_days = 90 # Hyperparameter


# Function to fetch stock data with error handling
def fetch_stock_data(symbol, start, end):
    try:
        stock_data = yf.download(symbol, start=start, end=end)
        if stock_data.empty:
            st.error(f"Error: The stock symbol '{symbol}' is incorrect or the stock may be delisted.")
            return None
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for symbol '{symbol}': {str(e)}")
        return None
    
# Function to create LSTM model
def create_model(input_shape):
    # tf.keras.backend.clear_session()  # Clear any previous model
    model = Sequential([
        LSTM(30, return_sequences=True, input_shape=input_shape), 
        Dropout(0.2),
        LSTM(30, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to prepare data for LSTM 
def prepare_data(data, time_step=No_prev_days):   
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    x_data, y_data = [], []
    for i in range(time_step, len(scaled_data)):
        x_data.append(scaled_data[i-time_step:i, 0])
        y_data.append(scaled_data[i, 0])
    return np.array(x_data), np.array(y_data), scaler

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(values, 'orange', label='MA')
    ax.plot(full_data.Close, 'b', label='Close Price')
    if extra_data:
        ax.plot(extra_dataset, 'g', label='Extra Data')
    ax.legend()
    return fig

# Streamlit app
st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock ID", "GOOG")

# Fetch data
end = datetime.now()
start = datetime(end.year-7, end.month, end.day) # 7 years data
# data shape dataframe : Date are the index and columns are the open, high, low , close, Adj close, volume
data = fetch_stock_data(stock, start, end)
if data is None:
    st.stop()  # Stop execution if there's an error
st.subheader("7 years Stock Data ")
st.write(data)

# Prepare data
close_prices = data['Close'].values 
x, y, scaler = prepare_data(close_prices) 

# Split data
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Reshape data for LSTM Chatgpt
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Create and train model
model = create_model((x_train.shape[1], 1))
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                    epochs=70, batch_size=32, verbose=0)
st.success("Model trained successfully.")


# Plot training error vs epochs
st.subheader('Training Error vs Epochs')
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
st.pyplot(fig)

# Make predictions
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)


# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict) 
test_predict = scaler.inverse_transform(test_predict)  
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)) 
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)) 

# Calculate RMSE and MAE
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))
train_mae = mean_absolute_error(y_train_inv, train_predict)
test_mae = mean_absolute_error(y_test_inv, test_predict)

st.subheader("Model Performance")
st.write(f"Train RMSE: {train_rmse:.2f}")
st.write(f"Test RMSE: {test_rmse:.2f}")
st.write(f"Train MAE: {train_mae:.2f}")
st.write(f"Test MAE: {test_mae:.2f}")

# Plot predictions 
st.subheader('Original Close Price vs Predicted Close Price')
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(data.index[No_prev_days:train_size+No_prev_days], y_train_inv, label='Original Train Data')
ax.plot(data.index[train_size+No_prev_days:], y_test_inv, label='Original Test Data')
ax.plot(data.index[No_prev_days:train_size+No_prev_days], train_predict, label='Predicted Train Data')
ax.plot(data.index[train_size+No_prev_days:], test_predict, label='Predicted Test Data')
ax.legend()
st.pyplot(fig)

# now train the model with the remaining test data, so that it gets trained on the entire data
# complete the code for it here
# Train the model with the remaining test data

# model.fit(x_test, y_test, epochs=50, batch_size=32, verbose=0)
# st.success("Model retrained with all available data!")


# Predict future trend for next 60 days (update 2)
# (Need to update for previous how many days data we are using to predict the next day, here 90 days)
last_90_days = close_prices[-No_prev_days:] 
last_90_days_scaled = scaler.transform(last_90_days.reshape(-1, 1)) 
X_future = np.array([last_90_days_scaled]) 
X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1)) 

future_days = 60  # 2 months
future_pred = []

try:
    for i in range(future_days):
        future_price = model.predict(X_future)
        future_pred.append(future_price[0])
        X_future = np.append(X_future[:, 1:, :], future_price.reshape(1, 1, 1), axis=1)

    future_pred = scaler.inverse_transform(np.array(future_pred).reshape(-1, 1)) 
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=future_days) 


    st.subheader(f'Future Trend Prediction using previous {No_prev_days} days data')
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(data.index, close_prices, label='Historical Data')
    ax.plot(future_dates, future_pred, label='Future Prediction')
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error predicting future trend: {str(e)}")

# Display current/todays stock price 
st.subheader("Current Stock Price")
st.write(f"The current stock price of {stock} is {close_prices[-1]}")


# Display next week's predictions with variance
next_week_pred = future_pred[:7]
next_week_dates = future_dates[:7]
rolling_std = pd.Series(close_prices).rolling(window=7).std().iloc[-1]
upper_band = next_week_pred + (2 * rolling_std)
lower_band = next_week_pred - (2 * rolling_std)

next_week_df = pd.DataFrame({
    'Date': next_week_dates,
    'Predicted Close': next_week_pred.flatten(),
    'Upper Band': upper_band.flatten(),
    'Lower Band': lower_band.flatten()
})

st.subheader("Next Week's Predictions with Variance")
st.write(next_week_df.set_index('Date'))


try:
    st.subheader("Future return predictions with variance")
    current_price = close_prices[-1]
    rolling_std = pd.Series(close_prices[-60:]).std()  # Calculate rolling std over the last 60 days
    prediction_days = [20, 30, 45, 60]
    #rolling_stds = [pd.Series(close_prices[-day:]).std() for day in prediction_days]  # Calculate std() for each prediction day
    
    prediction_df = pd.DataFrame({
        'Days': prediction_days,
        'Predicted Price': [future_pred[i-1][0] for i in prediction_days],
        'Upper Band': [future_pred[i-1][0] + (2 * rolling_std) for i in prediction_days],
        'Lower Band': [future_pred[i-1][0] - (2 * rolling_std) for i in prediction_days],
        'Percentage Change': [((future_pred[i-1][0] - current_price) / current_price) * 100 for i in prediction_days]
    })

    prediction_df['Percentage Change'] = prediction_df['Percentage Change'].apply(lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
    st.table(prediction_df.set_index('Days'))
except Exception as e:
    st.error(f"Error in creating prediction table: {str(e)}")
