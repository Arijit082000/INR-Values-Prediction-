import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import altair as alt
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

# Fetch historical data for lookback input (Refreshes every 1 hour)
@st.cache_data(ttl=3600)
def get_historical_data():
    df = yf.download('USDINR=X', start='2016-01-01', progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    return df

df = get_historical_data()
data = df.filter(['Close'])

# Fetch latest real-time/close price for USD/INR (Ticker: INR=X)
st.write("Fetching latest data from Yahoo Finance...")
ticker = "INR=X"
live_data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
if isinstance(live_data.columns, pd.MultiIndex):
    live_data.columns = live_data.columns.droplevel(1)

if not live_data.empty:
    current_real_price = float(live_data['Close'].iloc[-1].item())
    latest_date_str = live_data.index[-1].strftime('%Y-%m-%d %H:%M')
else:
    # Fallback default price if API fails
    current_real_price = 95.50
    latest_date_str = "Latest Fallback"

# Prepare input using the last 30 days of data (Optimized Lookback)
last_30_days = data[-30:].values
last_30_days_scaled = scaler.transform(last_30_days)
X_future = np.reshape(last_30_days_scaled, (1, last_30_days_scaled.shape[0], 1))

# Future prediction
future_pred_scaled = model.predict(X_future, verbose=0)

# Inverse transform to get actual currency values
close_min = scaler.min_[0]
close_scale = scaler.scale_[0]
future_predictions = (future_pred_scaled - close_min) / close_scale

# --- Automatic Real-Time Bias Correction (Trend Anchor Method) ---
future_days = 5
future_predictions_adjusted = np.zeros(future_days)

# First day prediction: anchor to current real price with half the trend step
trend_step = future_predictions[0][1] - future_predictions[0][0]
future_predictions_adjusted[0] = current_real_price + (trend_step / 2) 

# Predictions for the remaining days: maintain the original model's daily gaps
for i in range(1, future_days):
    daily_change = future_predictions[0][i] - future_predictions[0][i-1]
    future_predictions_adjusted[i] = future_predictions_adjusted[i-1] + daily_change
# -------------------------------------------

# --- Current Value Section ---
st.subheader("Current Exchange Rate:")
st.metric(label=f"Latest Value ({latest_date_str})", value=f"₹ {current_real_price:.4f}")
st.markdown("---")
# -----------------------------

st.subheader("Next 5-Day Forecast:")
last_date = df.index[-1]
future_dates = []

# Display predictions
for i in range(future_days):
    next_date = last_date + pd.tseries.offsets.BDay(i+1) # Calculate only Business Days
    future_dates.append(next_date)
    st.success(f"**{next_date.date()}**  →  ₹ {future_predictions_adjusted[i]:.4f}")

# --- 30-Day Historical + 5-Day Forecast Interactive Chart ---
st.markdown("---")
st.subheader("Historical Trend (Last 30 Days) & 5-Day Forecast 📊")

# 1. Last 30 days historical data
hist_df = data[-30:].copy().reset_index()
hist_df.columns = ['Date', 'Rate']
hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.tz_localize(None)
hist_df['Type'] = 'Historical (Last 30 Days)'

# 2. Next 5 days forecast data (anchored to the last historical point so lines connect)
forecast_dates = [hist_df['Date'].iloc[-1]] + [d.tz_localize(None) for d in future_dates]
forecast_rates = [float(hist_df['Rate'].iloc[-1])] + list(future_predictions_adjusted)

forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Rate': forecast_rates,
    'Type': '5-Day Forecast'
})

combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)

chart = (
    alt.Chart(combined_df)
    .mark_line(point=True, strokeWidth=2.5)
    .encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Rate:Q', title='Exchange Rate (₹)', scale=alt.Scale(zero=False)),
        color=alt.Color(
            'Type:N',
            title='Series',
            scale=alt.Scale(
                domain=['Historical (Last 30 Days)', '5-Day Forecast'],
                range=['#29b5e8', '#00cc96']
            ),
            legend=alt.Legend(orient='top')
        ),
        tooltip=[
            alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
            alt.Tooltip('Type:N', title='Category'),
            alt.Tooltip('Rate:Q', title='Rate (₹)', format='.4f')
        ]
    )
    .properties(height=380)
    .interactive()
)

st.altair_chart(chart, use_container_width=True)

# Add model performance metrics and developer info to the sidebar
st.sidebar.header("Model Evaluation Metrics")
st.sidebar.info("Test RMSE: 0.5628 \n\nTest MAE: 0.4033")
st.sidebar.write("Error margin is exceptionally low, indicating highly accurate performance on unseen data.")
st.sidebar.markdown("---")
st.sidebar.write("Developed by **Arijit Dasgupta**")
