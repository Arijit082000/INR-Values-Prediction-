# USD to INR Exchange Rate Predictor 

An end-to-end Deep Learning web application that uses a Long Short-Term Memory (LSTM) neural network to forecast future USD to INR exchange rates. The application fetches real-time financial data, preprocesses it, and generates multi-day predictions through an interactive Streamlit interface.

##  Live Demo
* **App Link:** https://8mthgdgjrrjtqyra3rcbmy.streamlit.app/



##  Key Features
* **Real-Time Data Integration:** Automatically fetches recent historical currency data using `yfinance`.
* **Deep Learning Forecasting:** Utilizes an LSTM time-series model trained on historical exchange rates to predict upcoming trends.
* **Data Scaling & Processing:** Employs `scikit-learn` scalers to normalize and inverse-transform data for accurate numerical outputs.
* **Interactive Dashboard:** Built using Streamlit to provide a clean, fast, and user-friendly web interface.

---

## Tech Stack & Libraries
* **Language:** Python
* **Web Framework:** Streamlit
* **Deep Learning:** TensorFlow / Keras (LSTM Model)
* **Data Manipulation & Finance:** Pandas, NumPy, yFinance
* **Machine Learning Preprocessing:** Scikit-Learn (MinMaxScaler)

---

##  Project Structure
```text
 INR-Values-Prediction
 ┣  app.py                  # Main Streamlit application script
 ┣  requirements.txt        # Required Python packages & dependencies
 ┣  runtime.txt             # Python version configuration for deployment
 ┣  scaler.pkl              # Saved Scikit-Learn scaler object
 ┗  usdinr_lstm_model.keras # Trained LSTM deep learning model

git clone [https://github.com/Arijit082000/INR-Values-Prediction-.git](https://github.com/Arijit082000/INR-Values-Prediction-.git)
cd INR-Values-Prediction-


pip install -r requirements.txt


streamlit run app.py
