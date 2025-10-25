# main.py 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime, date, timedelta
import tensorflow as tf
from io import BytesIO
import time
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Importing the required model scripts ---
from prophet_script import read_process as prophet_read_process, forecast as prophet_forecast
from lightgbm_script import forecast_lightgbm
from lstm_script import forecast_lstm

# Helper functions remain the same
def save_fig_to_bytes(fig):
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    return img_bytes

def plot_forecast_only(forecast_df, model_name):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(forecast_df['ds'], forecast_df['yhat'], label=f'{model_name} Forecast')
    ax.set_title(f"{model_name} Forecast Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True)
    return fig

# --- Generator Functions remain the same ---
def generate_prophet_files(df, timesteps, freq):
    model = Prophet()
    forecast_df, _ = prophet_forecast(model, df, timesteps, freq)
    fig_new = plot_forecast_only(forecast_df, "Prophet")
    st.pyplot(fig_new)
    csv = forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Sales'}).to_csv(index=False).encode('utf-8')
    img_bytes = save_fig_to_bytes(fig_new)
    return csv, img_bytes, forecast_df

def generate_lightgbm_files(df, timesteps):
    forecast_df, _, _ = forecast_lightgbm(df, timesteps)
    fig_new = plot_forecast_only(forecast_df, "LightGBM")
    st.pyplot(fig_new)
    csv = forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Sales'}).to_csv(index=False).encode('utf-8')
    img_bytes = save_fig_to_bytes(fig_new)
    return csv, img_bytes, forecast_df
    
def generate_lstm_files(df, timesteps):
    forecast_df, _, _ = forecast_lstm(df, timesteps)
    fig_new = plot_forecast_only(forecast_df, "LSTM")
    st.pyplot(fig_new)
    csv = forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Sales'}).to_csv(index=False).encode('utf-8')
    img_bytes = save_fig_to_bytes(fig_new)
    return csv, img_bytes, forecast_df

# --- Comparison Dashboard (Updated to use Tabs for results) ---
def run_comparison_dashboard(df, timesteps, freq):
    st.header("Model Comparison Dashboard")
    
    # Model execution logic remains the same
    all_forecasts = {'Date': pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=timesteps)}
    with st.spinner("Running Prophet..."):
        # ... (rest of the model running code is unchanged)
        with mlflow.start_run(run_name="Prophet"):
            model = Prophet()
            prophet_df, _ = prophet_forecast(model, df.copy(), timesteps, freq)
            all_forecasts['Prophet'] = prophet_df['yhat'].values
            fitted_vals = model.predict(df[['ds']])['yhat']
            rmse_train = np.sqrt(mean_squared_error(df['y'], fitted_vals))
            mae_train = mean_absolute_error(df['y'], fitted_vals)
            mlflow.log_metric("RMSE_train", rmse_train)
            mlflow.log_metric("MAE_train", mae_train)
            mlflow.log_param("model_type", "Prophet")

    with st.spinner("Running LightGBM..."):
        lgbm_df, _, _ = forecast_lightgbm(df.copy(), timesteps)
        all_forecasts['LightGBM'] = lgbm_df['yhat'].values
        
    with st.spinner("Running LSTM..."):
        lstm_df, _, _ = forecast_lstm(df.copy(), timesteps)
        all_forecasts['LSTM'] = lstm_df['yhat'].values

    forecast_df = pd.DataFrame(all_forecasts)

    # --- UI IMPROVEMENT: Use Tabs for a cleaner result display ---
    tab1, tab2 = st.tabs(["📊 Forecast Chart", "📈 Metrics Summary"])

    with tab1:
        st.subheader("All Model Forecasts")
        fig, ax = plt.subplots(figsize=(15, 10))
        for col in forecast_df.columns:
            if col != 'Date':
                ax.plot(forecast_df['Date'], forecast_df[col], label=col)
        ax.legend()
        ax.set_title("Model Forecast Comparison")
        st.pyplot(fig)

    with tab2:
        st.subheader("Experiment Metrics")
        st.info("The following metrics were logged to MLflow. Run `mlflow ui` for more details.")
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=3)
        desired_cols = ['tags.mlflow.runName', 'metrics.RMSE_test', 'metrics.MAE_test', 'metrics.RMSE_train', 'metrics.MAE_train']
        cols_to_show = [col for col in desired_cols if col in runs.columns]
        st.dataframe(runs[cols_to_show].rename(columns={
            'tags.mlflow.runName': 'Model', 'metrics.RMSE_test': 'Test RMSE',
            'metrics.MAE_test': 'Test MAE', 'metrics.RMSE_train': 'Train RMSE', 'metrics.MAE_train': 'Train MAE',
        }), use_container_width=True)

    st.session_state.csv_data = forecast_df.to_csv(index=False).encode('utf-8')
    st.session_state.img_data = save_fig_to_bytes(fig)
    st.success("Comparison complete!")

# --- Main application function (Completely Re-designed UI) ---
def app():
    # Page Title
    st.header("Generate Forecasts")
    st.markdown("Upload your time-series data to generate and compare forecasts from multiple models.")
    
    # --- UI IMPROVEMENT: Use a container with a border for all inputs ---
    with st.container(border=True):
        st.subheader("1. Configure Your Forecast")
        
        # --- UI IMPROVEMENT: Use columns for a cleaner layout ---
        col1, col2 = st.columns([2, 1]) # Give the file uploader more space
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload your sales CSV file",
                type="csv",
                help="The CSV file must contain 'Date' and 'Sales' columns."
            )
        with col2:
            end_date = st.date_input(
                "Select Forecast End Date",
                datetime.now().date() + timedelta(days=90)
            )
            
        st.subheader("2. Select Your Mode")
        run_mode = st.radio(
            "Choose an option:",
            ["Single Model Forecast", "Compare All Models"],
            horizontal=True,
            label_visibility="collapsed"
        )

    # --- Main Logic ---
    if uploaded_file is not None:
        try:
            df_prophet_style = prophet_read_process(uploaded_file)
            start_date = df_prophet_style["ds"].iloc[-1]
            d1 = df_prophet_style["ds"].iloc[0]
            d2 = df_prophet_style["ds"].iloc[1]
            if isinstance(end_date, date): end_date = pd.Timestamp(end_date)
            freq_days = (d2 - d1).days
            if freq_days == 1: timesteps = (end_date - start_date).days
            elif freq_days == 7: timesteps = (end_date - start_date).days // 7
            else:
                st.error("Data frequency not recognized as daily or weekly.")
                return
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return
        
        # --- UI IMPROVEMENT: Add a clear call-to-action button and container for results ---
        st.markdown("---")
        st.subheader("3. Generate & View Results")
        
        if run_mode == "Single Model Forecast":
            model_selection = st.selectbox("Select Model", ["Prophet", "LightGBM", "LSTM"])
            if st.button("🚀 Generate Forecast", use_container_width=True):
                with st.spinner(f"Running {model_selection}..."):
                    # Logic is unchanged
                    if model_selection == "Prophet": st.session_state.csv_data, st.session_state.img_data, _ = generate_prophet_files(df_prophet_style, timesteps, freq_days)
                    elif model_selection == "LightGBM": st.session_state.csv_data, st.session_state.img_data, _ = generate_lightgbm_files(df_prophet_style, timesteps)
                    elif model_selection == "LSTM": st.session_state.csv_data, st.session_state.img_data, _ = generate_lstm_files(df_prophet_style, timesteps)
                st.success("Forecast Generated!")
        
        elif run_mode == "Compare All Models":
            if st.button("🚀 Run Comparison", use_container_width=True):
                run_comparison_dashboard(df_prophet_style, timesteps, freq_days)

        # --- UI IMPROVEMENT: Organize download buttons into columns ---
        if 'csv_data' in st.session_state and 'img_data' in st.session_state:
            st.markdown("---")
            st.subheader("4. Download Your Results")
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button("Download Forecast Data (.csv)", st.session_state.csv_data, 'forecasts.csv', 'text/csv', use_container_width=True)
            with d_col2:
                st.download_button("Download Chart (.png)", st.session_state.img_data, 'forecast_chart.png', 'image/png', use_container_width=True)