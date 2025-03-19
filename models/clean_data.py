import pandas as pd

# Load the dataset
data = pd.read_csv('/kaggle/input/air-quality-data-in-india/city_hour.csv')

threshold = len(data) * 0.5
data = data.loc[:, data.isnull().sum() < threshold]

# Fill missing values with column mean
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Convert 'Datetime' to pandas datetime object
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Extract temporal features
data['Year'] = data['Datetime'].dt.year
data['Month'] = data['Datetime'].dt.month
data['Day'] = data['Datetime'].dt.day
data['Hour'] = data['Datetime'].dt.hour
data['Weekday'] = data['Datetime'].dt.weekday  # 0 = Monday, 6 = Sunday


relevant_columns = ['Datetime', 'City', 'PM2.5', 'PM10', 'NO2', 'SO2', 'AQI']
data = data[relevant_columns]

# Save the cleaned data
data.to_csv('/kaggle/working/processed_air_quality_data.csv')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Create a new 'AQI_Category' column
data['AQI_Category'] = data['AQI'].apply(categorize_aqi)
