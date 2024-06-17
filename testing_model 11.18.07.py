import pandas as pd
import numpy as np
import joblib

# Libraries 

# Load data
weather_data_test = pd.read_csv('weather_data_test/weather_data_test.csv')

# Set up the index and deleting unnecessary columns
weather_data_test.columns = weather_data_test.iloc[8]
weather_data_test = weather_data_test.drop(index=range(9)).reset_index(drop=True)

# Rename columns
weather_data_test.columns = [
    'date', 
    'temp. [2m elevation corrected](°C)', 
    'temperature [850mb](°C)', 
    'temperature [500mb](°C)', 
    'precipitation_total (mm)', 
    'relative_humidity [2m](%)', 
    'snowfall_amount (cm)', 
    'wind_gusts', 
    'wind_speed [10m](km/h)', 
    'wind_direction [10m](°)', 
    'wind_speed [850mb](km/h)', 
    'wind_direction [850mb](°)', 
    'wind_speed [500mb](km/h)', 
    'wind_direction [500mb](°)', 
    'cloud_cover_total (%)',
    'cloud_cover_high [high cld lay](%)',
    'cloud_cover_medium [mid cld lay](%)',
    'cloud_cover_low [low cld lay](%)',
    'shortwave_radiation (W/m²)',
    'longwave_radiation (W/m²)',
    'mean_sea_level_pressure (hPa)',
    'geopotential_height [800mb](Gpm)',  
    'temperature (°C)',
    'soil_temperature [0-7cm down](°C)'
]

# Convert numeric columns to the correct data type
numeric_columns = [
    'temp. [2m elevation corrected](°C)', 'temperature [850mb](°C)', 'temperature [500mb](°C)', 
    'precipitation_total (mm)', 'relative_humidity [2m](%)', 'snowfall_amount (cm)', 
    'wind_speed [10m](km/h)', 'wind_direction [10m](°)', 'wind_speed [850mb](km/h)', 
    'wind_direction [850mb](°)', 'wind_speed [500mb](km/h)', 'wind_direction [500mb](°)', 
    'cloud_cover_total (%)', 'cloud_cover_high [high cld lay](%)', 'cloud_cover_medium [mid cld lay](%)', 
    'cloud_cover_low [low cld lay](%)', 'shortwave_radiation (W/m²)', 'longwave_radiation (W/m²)', 
    'mean_sea_level_pressure (hPa)', 'geopotential_height [800mb](Gpm)', 'temperature (°C)', 
    'soil_temperature [0-7cm down](°C)'
]

# Convert columns to numeric, coercing errors to NaN
for column in numeric_columns:
    weather_data_test[column] = pd.to_numeric(weather_data_test[column], errors='coerce')

# Format date column
weather_data_test['date'] = pd.to_datetime(weather_data_test['date'])

# Extract year, month, day and hour from date column
weather_data_test['year'] = weather_data_test['date'].dt.year
weather_data_test['month'] = weather_data_test['date'].dt.month
weather_data_test['day'] = weather_data_test['date'].dt.day
weather_data_test['hour'] = weather_data_test['date'].dt.hour

# Establish dayofweek number and name column
weather_data_test['dayofweek'] = weather_data_test['date'].dt.dayofweek
weather_data_test['dayofweek_name'] = weather_data_test['date'].dt.day_name()

# Delete null values and duplicates
weather_data_test = weather_data_test.dropna()
weather_data_test = weather_data_test.drop_duplicates()

# Weekday and weekend
weather_data_test['is_weekend'] = weather_data_test['dayofweek'].isin([5, 6]) 

# Organize index columns
weather_data_test = weather_data_test[['date', 'year', 'month', 'day', 'hour', 'dayofweek', 'dayofweek_name', 'is_weekend',
                                     'temp. [2m elevation corrected](°C)', 'temperature [850mb](°C)', 'temperature [500mb](°C)', 
                                     'precipitation_total (mm)', 'relative_humidity [2m](%)', 'snowfall_amount (cm)', 
                                     'wind_speed [10m](km/h)', 'wind_direction [10m](°)', 'wind_speed [850mb](km/h)', 
                                     'wind_direction [850mb](°)', 'wind_speed [500mb](km/h)', 'wind_direction [500mb](°)', 
                                     'cloud_cover_total (%)', 'cloud_cover_high [high cld lay](%)', 'cloud_cover_medium [mid cld lay](%)', 
                                     'cloud_cover_low [low cld lay](%)', 'shortwave_radiation (W/m²)', 'longwave_radiation (W/m²)', 
                                     'mean_sea_level_pressure (hPa)', 'geopotential_height [800mb](Gpm)', 'temperature (°C)', 
                                     'soil_temperature [0-7cm down](°C)']]

# Function to calculate dew point
def calculate_dew_point(temp_celsius, relative_humidity):
    temp_fahrenheit = temp_celsius * 9/5 + 32
    vapor_pressure = 6.112 * np.exp((17.67 * temp_fahrenheit) / (temp_fahrenheit + 243.5)) * relative_humidity / 100
    dew_point_celsius = (243.5 * np.log(vapor_pressure / 6.112)) / (17.67 - np.log(vapor_pressure / 6.112))
    
    return dew_point_celsius

# Calculate dew point for each row
weather_data_test['dew_point_C'] = weather_data_test.apply(
    lambda row: calculate_dew_point(row['temp. [2m elevation corrected](°C)'], row['relative_humidity [2m](%)']), axis=1)

print(weather_data_test)

# Hourly Prediction

columns_to_load = ['year', 'month', 'day', 'hour', 
            'temp. [2m elevation corrected](°C)', 'temperature [850mb](°C)', 
            'temperature [500mb](°C)', 'precipitation_total (mm)', 
            'relative_humidity [2m](%)', 'snowfall_amount (cm)', 
            'wind_speed [10m](km/h)', 'wind_direction [10m](°)', 
            'wind_speed [850mb](km/h)', 'wind_direction [850mb](°)', 
            'wind_speed [500mb](km/h)', 'wind_direction [500mb](°)', 
            'cloud_cover_total (%)', 'cloud_cover_high [high cld lay](%)', 
            'cloud_cover_medium [mid cld lay](%)', 'cloud_cover_low [low cld lay](%)', 
            'shortwave_radiation (W/m²)', 'longwave_radiation (W/m²)', 
            'mean_sea_level_pressure (hPa)', 'geopotential_height [800mb](Gpm)', 
            'temperature (°C)', 'soil_temperature [0-7cm down](°C)', 'dew_point_C']

# Load the trained model
model = joblib.load('models/weather_data.pkl')

# Standardize the features using the scaler trained on the training data
scaler = joblib.load('scaler_folder/scaler.pkl')
X_test = weather_data_test[columns_to_load]
X_test_scaled = scaler.transform(X_test)

# Make predictions
predictions = model.predict(X_test_scaled)

# Define the weather conditions dictionary
weather_conditions = {
    'Rainy': 'Rainy',
    'Overcast': 'Overcast',
    'Partly Cloudy': 'Partly Cloudy',
    'Snowy': 'Snowy',
    'Foggy': 'Foggy',
    'Windy': 'Windy',
    'Sunny': 'Sunny',
    'Clear': 'Clear'
}

# Map predictions to weather conditions
predicted_weather = [weather_conditions.get(pred, 'Unknown') for pred in predictions]

# Concatenate predicted_weather list with selected columns from weather_data_test into a DataFrame
weather_prediction = pd.concat([
    weather_data_test[['year', 'month', 'day', 'hour', 'temperature [850mb](°C)', 'wind_speed [10m](km/h)', 'cloud_cover_total (%)', 'relative_humidity [2m](%)']], 
    pd.DataFrame({'predicted_weather': predicted_weather})
], axis=1)

# Display the DataFrame with predictions
print(weather_prediction.head(50))
