import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Yes Bank Stock Predictor", page_icon="📈", layout="centered")

st.title("📈 Yes Bank Monthly Close Price Predictor")
st.write("This app predicts the monthly closing stock price for Yes Bank based on the Open, High, and Low prices for that month.")

from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('best_model_rf.pkl')
    # The Random Forest was trained on scaled data, so we must recreate the scaler
    df = pd.read_csv('data_YesBank_StockPrices.csv')
    scaler = StandardScaler()
    scaler.fit(df[['Open', 'High', 'Low']])
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading the model or data: {e}")
    st.stop()

st.sidebar.header("Input Features")
st.sidebar.write("Please enter the stock prices for the month:")

open_price = st.sidebar.number_input("Open Price (₹)", min_value=0.0, value=100.0, step=1.0)
high_price = st.sidebar.number_input("High Price (₹)", min_value=0.0, value=110.0, step=1.0)
low_price = st.sidebar.number_input("Low Price (₹)", min_value=0.0, value=90.0, step=1.0)

if high_price < open_price or high_price < low_price:
    st.sidebar.warning("Note: High price should generally be greater than or equal to Open and Low prices.")
if low_price > open_price or low_price > high_price:
    st.sidebar.warning("Note: Low price should generally be less than or equal to Open and High prices.")

if st.sidebar.button("Predict Close Price"):
    input_df = pd.DataFrame({
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price]
    })
    
    with st.spinner("Predicting..."):
        # Transform inputs using the fitted scaler
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
    
    st.success(f"### Predicted Close Price: ₹{prediction:.2f}")
    
    st.write("### Provided Inputs")
    st.write(f"- **Open**: ₹{open_price}")
    st.write(f"- **High**: ₹{high_price}")
    st.write(f"- **Low**: ₹{low_price}")
    
    st.info("This prediction is powered by a Random Forest Regressor trained on historical monthly stock data of Yes Bank.")
