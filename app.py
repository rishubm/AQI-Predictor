import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load("xgboost_aqi_model.pkl")

# Define pollutant ranges
ranges = {
    'PM2.5': (0, 500),
    'PM10': (0, 600),
    'NO2': (0, 300),
    'SO2': (0, 200)
}

# Streamlit interface
st.title("🌍 Real-Time AQI Prediction in India")
st.write("Enter the current pollutant levels to predict the AQI category.")

# Sidebar for pollutant input
st.sidebar.header("Enter Pollutant Levels")

# User inputs
features = []
for pollutant, (min_val, max_val) in ranges.items():
    value = st.sidebar.slider(f"{pollutant} (µg/m³)", min_value=min_val, max_value=max_val, value=(min_val + max_val) // 2)
    features.append(value)

# Predict button
if st.sidebar.button("Predict AQI"):
    # Reshape input for the model
    input_features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)[0]

    # AQI categories and suggestions
    aqi_labels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    suggestions = {
        'Good': "✅ Enjoy the fresh air!",
        'Moderate': "😐 It's fine but be cautious if you have respiratory issues.",
        'Unhealthy for Sensitive Groups': "⚠️ Sensitive groups should limit outdoor activities.",
        'Unhealthy': "❗ Everyone should reduce outdoor activities.",
        'Very Unhealthy': "🚫 Avoid outdoor activities. Use an air purifier if possible.",
        'Hazardous': "☠️ Stay indoors! Wear a mask if you must go out."
    }

    # Display AQI category and suggestion
    aqi_category = aqi_labels[prediction]
    st.success(f"🌟 **AQI Category:** {aqi_category}")
    st.warning(suggestions[aqi_category])

    # Display feature importance
    st.subheader("🔍 Feature Importance")
    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.bar(ranges.keys(), importances, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('XGBoost Feature Importance')
    plt.xticks(rotation=45)
    st.pyplot(fig)
