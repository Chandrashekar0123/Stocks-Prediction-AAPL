import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Load model and scaler (updated filenames)
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

# Streamlit page setup
st.set_page_config(page_title="Stock Forecast", layout="centered")
st.title("📈 Stock Price Prediction App")

# Disclaimer
st.warning("⚠️ Disclaimer: This is just a prediction. Please do not rely solely on it for investment decisions. Invest wisely and at your own risk.")

# Inputs
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL").upper()
start_date = st.date_input("Select Start Date", min_value=datetime.date.today())
end_date = st.date_input("Select End Date", min_value=start_date, max_value=datetime.date.today() + datetime.timedelta(days=365))
interval_choice = st.selectbox("View Interval", ["Daily", "Weekly", "Monthly", "Yearly"], index=0)

# Load stock data
@st.cache_data
def get_data(symbol):
    return yf.download(symbol, start="2010-01-01", end=datetime.date.today().strftime('%Y-%m-%d'))

# Preprocessing
def prepare_input(data):
    close = data['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(close)
    return scaled[-60:].reshape(1, 60, 1)

def predict_future(input_data, days):
    predictions = []
    current_input = input_data.copy()
    for _ in range(days):
        pred = model.predict(current_input)[0][0]
        predictions.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Resample for interval view
def resample_predictions(dates, prices, interval):
    df = pd.DataFrame({'Date': dates, 'Predicted_Price': prices.flatten()})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    if interval == "Weekly":
        df = df.resample('W').mean()
    elif interval == "Monthly":
        df = df.resample('M').mean()
    elif interval == "Yearly":
        df = df.resample('Y').mean()

    return df

# Prediction button
if st.button("Predict Future Stock Prices"):
    try:
        # Load and prepare data
        data = get_data(stock_symbol)
        model_input = prepare_input(data)

        # Generate predictions
        future_days = (end_date - start_date).days + 1
        future_dates = pd.date_range(start=start_date, periods=future_days, freq='B')
        predictions = predict_future(model_input, future_days)

        # Resample if needed
        prediction_df = resample_predictions(future_dates, predictions, interval_choice)

        # Display table
        st.subheader(f"📄 Predicted Prices for {stock_symbol}")
        st.dataframe(prediction_df.rename(columns={"Predicted_Price": "Predicted Price (USD)"}))

        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Predicted_Price'],
                                 mode='lines+markers', name='Predicted Price'))
        fig.update_layout(title=f"{stock_symbol} Forecast ({interval_choice})",
                          xaxis_title="Date", yaxis_title="Price (USD)",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
