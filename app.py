import streamlit as st
import pickle
import numpy as np
import pandas as pd

import joblib

# Load model and features
model = joblib.load('linear_regression_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')


st.title("🚕 NYC Green Taxi Fare Predictor")
st.markdown("Input your trip details to predict the total fare (including tips & surcharges).")

# Build input dictionary
user_input = {}

# Manual mapping for dummy columns
dummy_categories = {
    'store_and_fwd_flag': ['Y'],
    'weekday': ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
}

# Basic numeric & categorical fields
for feature in feature_columns:
    if feature.startswith('store_and_fwd_flag_') or feature.startswith('weekday_'):
        continue
    elif feature in ['passenger_count', 'hourofday']:
        user_input[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0, step=1)
    else:
        user_input[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0)

# Dummy fields: store_and_fwd_flag
flag_input = st.selectbox("Store and Forward Flag", ['N', 'Y'])
for val in dummy_categories['store_and_fwd_flag']:
    col = f"store_and_fwd_flag_{val}"
    user_input[col] = 1 if flag_input == val and col in feature_columns else 0

# Dummy fields: weekday
weekday_input = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
for val in dummy_categories['weekday']:
    col = f"weekday_{val}"
    user_input[col] = 1 if weekday_input == val and col in feature_columns else 0

# Build DataFrame in correct order
input_df = pd.DataFrame([user_input])[feature_columns]

# Predict button
if st.button("Predict Total Fare"):
    pred = model.predict(input_df)[0]
    st.success(f"💵 Estimated Total Amount: ${pred:.2f}")
