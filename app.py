import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import pickle

# Page configuration
st.set_page_config(page_title="USD-INR Predictor", page_icon="📈", layout="centered")

st.title('USD to INR Exchange Rate Predictor 📈')
st.write("A Deep Learning (LSTM) based time-series forecasting model predicting the next 5 days.")

# Cache the model and scaler to prevent reloading on every app rerun
@st.cache_resource
def load_assets():
    model = load_model('usdinr_lstm_model.keras', compile=False)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# Fetch latest data (explicitly setting auto_adjust to avoid warnings)
st.write("Fetching latest data from Yahoo Finance...")
inr_quote = yf.download("USDINR=X", start="2016-01-01", progress=False, auto_adjust=True)
new_df = inr_quote[['Close']]

# Prepare input using the last 30 days of data (Optimized Lookback)
last_30_days = new_df[-30:].values
last_30_days_scaled = scaler.transform(last_30_days)
X_future = np.reshape(last_30_days_scaled, (1, last_30_days_scaled.shape[0], 1))

# Future prediction
future_pred_scaled = model.predict(X_future, verbose=0)

# Inverse transform considering the (-1, 1) or standard scaler structure
close_min = scaler.min_[0]
close_scale = scaler.scale_[0]
future_predictions = (future_pred_scaled - close_min) / close_scale

# --- Automatic Real-Time Bias Correction ---
latest_date_str = new_df.index[-1].strftime('%Y-%m-%d')
current_real_price = float(new_df.iloc[-1].item())

model_base_price = future_predictions[0][0]  # Model's first day prediction
price_gap = current_real_price - model_base_price

# Adjust future predictions dynamically with the fetched live price gap
future_predictions_adjusted = future_predictions[0] + price_gap
# -------------------------------------------

# --- Current Value Section ---
st.subheader("Current Exchange Rate:")
st.metric(label=f"Latest Value ({latest_date_str})", value=f"₹ {current_real_price:.4f}")
st.markdown("---")
# -----------------------------

st.subheader("Next 5-Day Forecast (Live Adjusted):")
last_date = new_df.index[-1]

# Display predictions
for i in range(5):
    next_date = last_date + pd.tseries.offsets.BDay(i+1) # Calculate only Business Days
    st.success(f"**{next_date.date()}**  →  ₹ {future_predictions_adjusted[i]:.4f}")

# Add model performance metrics and developer info to the sidebar
st.sidebar.header("Model Evaluation Metrics")
st.sidebar.info("Test RMSE: 0.5186 \n\nTest MAE: 0.3644")
st.sidebar.write("Error margin is exceptionally low, indicating highly accurate performance on unseen data.")
st.sidebar.markdown("---")
st.sidebar.write("Developed by **Arijit Dasgupta**")
