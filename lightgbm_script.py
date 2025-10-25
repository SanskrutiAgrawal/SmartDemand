# lightgbm_script.py 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

def create_features_for_lgbm(df):
    """Creates time series features from a datetime index."""
    df = df.copy()
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    
    # --- PERFORMANCE UPGRADE: Add a time index to capture trend ---
    df['time_idx'] = np.arange(len(df))

    # Lag features
    for lag in [7, 14, 21, 28]:
        df[f'lag_{lag}'] = df['y'].shift(lag)
        
    # Rolling window features
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df['y'].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['y'].shift(1).rolling(window=window).std()
        
    return df.drop('date', axis=1)

def forecast_lightgbm(df, timesteps):
    with mlflow.start_run(run_name="LightGBM"):
        st.write("Engineering features for LightGBM model...")
        mlflow.log_param("model_type", "LightGBM")
        
        # --- PERFORMANCE REFACTOR: Engineer features on the whole dataset first ---
        df_lgbm = df.copy().set_index('ds')
        full_features = create_features_for_lgbm(df_lgbm)
        
        # Split data after feature creation
        train_size = int(len(full_features) * 0.8)
        train_df = full_features.iloc[:train_size]
        test_df = full_features.iloc[train_size:]
        
        FEATURES = [col for col in full_features.columns if col != 'y']
        TARGET = 'y'
        
        # Drop rows with NaNs from lags/windows in training data
        train_df = train_df.dropna()
        X_train, y_train = train_df[FEATURES], train_df[TARGET]
        X_test, y_test = test_df[FEATURES], test_df[TARGET]

        params = {'objective': 'regression', 'metric': 'rmse', 'n_estimators': 1000, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1}
        mlflow.log_params(params)
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        mlflow.log_metric("RMSE_test", rmse)
        mlflow.log_metric("MAE_test", mae)
        
        train_preds = model.predict(X_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, train_preds))
        mae_train = mean_absolute_error(y_train, train_preds)
        mlflow.log_metric("RMSE_train", rmse_train)
        mlflow.log_metric("MAE_train", mae_train)
        
        st.write("Performing autoregressive forecast...")
        future_preds = []
        history = full_features.copy() # Use full features with history
        
        for i in range(timesteps):
            next_step_features = history[FEATURES].tail(1)
            prediction = model.predict(next_step_features)[0]
            future_preds.append(prediction)
            
            # Create next row for history and append
            last_row = history.iloc[-1].name
            next_date = last_row + pd.DateOffset(days=1)
            # This is a simplified way; a more robust way would recreate features for the new row
            # But for forecasting, we mainly need to update the time-based features and lags
            history.loc[next_date] = {'y': prediction} # Append prediction
            history = create_features_for_lgbm(history[['y']]) # Re-create features for the new full history

        last_date = df['ds'].iloc[-1]
        future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i+1) for i in range(timesteps)])
        fig = plot_time_series(timesteps=df['ds'], values=df['y'], label="Historical Data")
        ax = fig.gca()
        ax.plot(future_dates, future_preds, label="LightGBM Forecast")
        ax.legend()
        
        return_df = pd.DataFrame({'ds': future_dates, 'yhat': future_preds})
        st.success("LightGBM forecast complete.")
        return return_df, None, fig