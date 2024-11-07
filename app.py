import streamlit as st
import joblib
import numpy as np
import pandas as pd

def main():
    st.title('Breast Cancer Prediction App')
    st.write("Enter the values for the features to get a prediction.")

    # Define input fields for user to enter all 30 feature values
    features = []
    for i in range(1, 31):
        feature_value = st.number_input(f'Feature {i} (Feature {i} Description)', min_value=0.0)
        features.append(feature_value)

    # Handle cases where input features may need categorical encoding
    def preprocess_input(features):
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Ensure that categorical columns are handled correctly
        # If using categorical features, convert to appropriate format
        # Example: encoding unknown categories with 'unknown' handling

        # Scale the features
        features_scaled = scaler.transform(np.array(features).reshape(1, -1))
        return features_scaled

    # Button to make a prediction
    if st.button('Predict'):
        try:
            features_scaled = preprocess_input(features)
            model = joblib.load('model.pkl')
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)[0, 1]

            # Display the result
            st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
            st.write(f"Probability of Positive: {probability:.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == '__main__':
    main()
