# prophet_script.py 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import streamlit as st

def plot_time_series(timesteps, values, format='-', start=0, end=None, label=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(timesteps[start:end], values[start:end], format, label=label)
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Forecasted Values of Sales")
    if label: ax.legend(fontsize=10)
    ax.grid(True)
    return fig

def read_process(file):
    df = pd.read_csv(file)
    date_formats = ["%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%b %d, %Y", "%d %b %Y", "%d %B %Y"]
    for date_format in date_formats:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=date_format)
            break
        except ValueError:
            continue
    if df['Date'].isna().any(): raise ValueError("Date parsing failed for all formats")
    
    data = pd.DataFrame()
    data["ds"] = df["Date"]
    data["y"] = df["Sales"]
    return data

def forecast(model, df, timesteps, f):
    # --- PERFORMANCE UPGRADE ---
    # 1. Add country-specific holidays. This is a massive driver of performance.
    # Change 'US' to your country's code (e.g., 'IN' for India, 'UK' for United Kingdom)
    model.add_country_holidays(country_name='US')

    model.fit(df)
    
    freq_str = 'D' if f == 1 else 'W'
    future_df = model.make_future_dataframe(periods=timesteps, freq=freq_str)
    
    full_forecast = model.predict(future_df)
    forecast_df = full_forecast[['ds', 'yhat']].iloc[len(df):].reset_index(drop=True)

    fig = model.plot(full_forecast)
    # Add changepoints to visualize trend flexibility
    a = add_changepoints_to_plot(fig.gca(), model, full_forecast)
    ax = fig.gca()
    ax.set_title("Prophet Forecast with Trend Changepoints")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    return forecast_df, fig

# We need a new function for the main script to call that uses the tuned parameters
def forecast_prophet_tuned(df, timesteps, f):
    # --- PERFORMANCE UPGRADE ---
    # 2. Tune prior scales. This makes the model more flexible.
    # changepoint_prior_scale: How flexibly the trend can change. Higher is more flexible.
    # seasonality_prior_scale: How much weight to give to seasonality. Higher fits seasonality harder.
    tuned_model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    return forecast(tuned_model, df, timesteps, f)