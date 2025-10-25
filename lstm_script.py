# lstm_script.py (Updated for Performance)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import mlflow

def plot_time_series(timesteps, values, format='-', start=0, end=None, label=None, title="Forecast"):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(timesteps[start:end], values[start:end], format, label=label)
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Forecasted Values of Sales")
    ax.set_title(title)
    if label: ax.legend(fontsize=10)
    ax.grid(True)
    return fig

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def forecast_lstm(df, timesteps, look_back=28, epochs=100): # Increased look_back and epochs
    with mlflow.start_run(run_name="LSTM"):
        st.write("Preprocessing data for LSTM...")
        mlflow.log_param("model_type", "LSTM_Stacked")
        mlflow.log_param("look_back_window", look_back)
        mlflow.log_param("epochs", epochs)

        dataset = df['y'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        
        train_size = int(len(dataset) * 0.8)
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        
        X_train, y_train = create_dataset(train, look_back)
        X_test, y_test = create_dataset(test, look_back)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        st.write("Training Stacked LSTM model...")
        # --- PERFORMANCE UPGRADE: Stacked LSTM with Dropout ---
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1))) # return_sequences=True to stack
        model.add(Dropout(0.2)) # Dropout layer to prevent overfitting
        model.add(LSTM(50, return_sequences=False)) # Second LSTM layer
        model.add(Dropout(0.2))
        model.add(Dense(1)) # Output layer
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        # Implement early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_train, y_train, 
                  epochs=epochs, 
                  batch_size=32, 
                  validation_data=(X_test, y_test), 
                  verbose=0,
                  callbacks=[early_stopping])
        
        # Evaluate on test set
        test_predict = model.predict(X_test)
        test_predict_inv = scaler.inverse_transform(test_predict)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
        mae = mean_absolute_error(y_test_inv, test_predict_inv)
        mlflow.log_metric("RMSE_test", rmse)
        mlflow.log_metric("MAE_test", mae)

        train_predict = model.predict(X_train)
        train_predict_inv = scaler.inverse_transform(train_predict)
        y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
        rmse_train = np.sqrt(mean_squared_error(y_train_inv, train_predict_inv))
        mae_train = mean_absolute_error(y_train_inv, train_predict_inv)
        mlflow.log_metric("RMSE_train", rmse_train)
        mlflow.log_metric("MAE_train", mae_train)

        st.write("Performing autoregressive forecast...")
        future_preds = []
        last_sequence = dataset[-look_back:]

        for _ in range(timesteps):
            current_input = np.reshape(last_sequence, (1, look_back, 1))
            prediction = model.predict(current_input, verbose=0)
            future_preds.append(prediction[0,0])
            last_sequence = np.append(last_sequence[1:], prediction, axis=0)

        future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        last_date = df['ds'].iloc[-1]
        future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i+1) for i in range(timesteps)])

        fig = plot_time_series(timesteps=df['ds'], values=df['y'], label="Historical Data")
        ax = fig.gca()
        ax.plot(future_dates, future_preds_inv.flatten(), label="LSTM Forecast")
        ax.legend()
        
        return_df = pd.DataFrame({'ds': future_dates, 'yhat': future_preds_inv.flatten()})
        st.success("LSTM forecast complete.")
        return return_df, None, fig