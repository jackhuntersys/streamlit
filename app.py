import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("ML Model Predictor")
user_input = st.number_input("Enter a value:")
prediction = model.predict([[user_input]])
st.write("Prediction:", prediction[0])