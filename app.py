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

# Fetch latest data
st.write("Fetching latest data from Yahoo Finance...")
inr_quote = yf.download("USDINR=X", start="2016-01-01", auto_adjust=False)
new_df = inr_quote[['Close']]

# Prepare input using the last 60 days of data
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_future = np.reshape(last_60_days_scaled, (1, last_60_days_scaled.shape[0], 1))

# Future prediction
future_pred_scaled = model.predict(X_future)
future_predictions = scaler.inverse_transform(future_pred_scaled)

# --- Newly added section (Current Value) ---
latest_date_str = new_df.index[-1].strftime('%Y-%m-%d')
latest_price = float(new_df.iloc[-1, 0])

st.subheader("Current Exchange Rate:")
st.metric(label=f"Latest Value ({latest_date_str})", value=f"₹ {latest_price:.4f}")
st.markdown("---")
# ----------------------------------------

st.subheader("Next 5-Day Forecast:")
last_date = new_df.index[-1]


# Display predictions
for i in range(5):
    next_date = last_date + pd.tseries.offsets.BDay(i+1) # Calculate only Business Days
    st.success(f"**{next_date.date()}**  →  ₹ {future_predictions[0][i]:.4f}")

# Add model performance metrics and developer info to the sidebar
st.sidebar.header("Model Evaluation Metrics")
st.sidebar.info("Test RMSE: 0.5786 \n\nTest MAE: 0.4341")
st.sidebar.write("Error margin is approximately ~0.5%-0.6%, indicating highly stable performance on unseen data.")
st.sidebar.markdown("---")
st.sidebar.write("Developed by **Arijit Dasgupta**")

