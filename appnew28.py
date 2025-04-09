
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and preprocessing objects
model = joblib.load('Suicide_Rates_Prediction.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoders.pkl')

# Load the original dataset (for country list)
df = pd.read_csv(r"C:\Users\shivam srivastava\Downloads\suicide-rate-by-country-2024.csv")

st.title("Suicide Rate Prediction for 2021")

# Collect user input
input_country = st.selectbox("Select Country", df['country'].unique())
input_2020_both = st.number_input("Both Sexes Rate (2020)", min_value=0.0)
input_2020_male = st.number_input("Male Rate (2020)", min_value=0.0)
input_2020_female = st.number_input("Female Rate (2020)", min_value=0.0)
input_2019_both = st.number_input("Both Sexes Rate (2019)", min_value=0.0)
input_2019_male = st.number_input("Male Rate (2019)", min_value=0.0)
input_2019_female = st.number_input("Female Rate (2019)", min_value=0.0)

if st.button("Predict Suicide Rate for 2021"):
    # Step 1: Build dataframe
    input_df = pd.DataFrame([{
        'country': input_country,
        'SuicideRate_BothSexes_RatePer100k_2020': input_2020_both,
        'SuicideRate_Male_RatePer100k_2020': input_2020_male,
        'SuicideRate_Female_RatePer100k_2020': input_2020_female,
        'SuicideRate_BothSexes_RatePer100k_2019': input_2019_both,
        'SuicideRate_Male_RatePer100k_2019': input_2019_male,
        'SuicideRate_Female_RatePer100k_2019': input_2019_female
    }])

    # Step 2: Feature Engineering
    input_df['YoY Change 2020'] = input_df['SuicideRate_BothSexes_RatePer100k_2020'] - input_df['SuicideRate_BothSexes_RatePer100k_2019']
    input_df['YoY Change 2021'] = 0  # Placeholder (you can drop if not needed)

    # Step 3: Transform columns
    cat_cols = ['country']
    num_cols = [col for col in input_df.columns if col not in cat_cols]

    # Preprocess input
    encoded_cat = encoder.transform(input_df[cat_cols])  # Remove .toarray()
    scaled_num = scaler.transform(input_df[num_cols])

    final_input = np.concatenate([scaled_num, encoded_cat], axis=1)

    # Step 4: Predict
    prediction = model.predict(final_input)
    st.success(f"Predicted Suicide Rate for 2021: {prediction[0]:.2f} per 100k population")
