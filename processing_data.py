import pandas as pd
import os
import numpy as np

# Libraries

# Load data
weather_city = pd.read_csv('original_data/weather-city.csv')

# Set up the index and deleting unnecessary columns
weather_city.columns = weather_city.iloc[8]
weather_city = weather_city.drop(index=range(9)).reset_index(drop=True)
#print(weather_city.head(10))

# Dataset info
#print(weather_city.info())

# Rename columns
weather_city.columns = [
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

# Display the DataFrame to verify the changes
#print(weather_city.head())

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
    weather_city[column] = pd.to_numeric(weather_city[column], errors='coerce')

# Format date column
weather_city['date'] = pd.to_datetime(weather_city['date'])

# Extract year, month, day and hour from date column
weather_city['year'] = weather_city['date'].dt.year
weather_city['month'] = weather_city['date'].dt.month
weather_city['day'] = weather_city['date'].dt.day
weather_city['hour'] = weather_city['date'].dt.hour

# Establish dayofweek number and name column
weather_city['dayofweek'] = weather_city['date'].dt.dayofweek
weather_city['dayofweek_name'] = weather_city['date'].dt.day_name()
##print(weather_city.head())

# Null values and duplicates
#print('Null Values', weather_city.isnull().sum())
#print('Duplicate Values', weather_city.duplicated().sum())

# Delete null values and duplicates
weather_city = weather_city.dropna()
weather_city = weather_city.drop_duplicates()
##print('Null Values', weather_city.isnull().sum())
##print('Duplicate Values', weather_city.duplicated().sum())

# Weekday and weekend
weather_city['is_weekend'] = weather_city['dayofweek'].isin([5, 6]) 
#print(weather_city.head(10))

# Organize index columns
weather_city = weather_city[['date', 'year', 'month', 'day', 'hour', 'dayofweek', 'dayofweek_name', 'is_weekend',
                                     'temp. [2m elevation corrected](°C)', 'temperature [850mb](°C)', 'temperature [500mb](°C)', 
                                     'precipitation_total (mm)', 'relative_humidity [2m](%)', 'snowfall_amount (cm)', 
                                     'wind_speed [10m](km/h)', 'wind_direction [10m](°)', 'wind_speed [850mb](km/h)', 
                                     'wind_direction [850mb](°)', 'wind_speed [500mb](km/h)', 'wind_direction [500mb](°)', 
                                     'cloud_cover_total (%)', 'cloud_cover_high [high cld lay](%)', 'cloud_cover_medium [mid cld lay](%)', 
                                     'cloud_cover_low [low cld lay](%)', 'shortwave_radiation (W/m²)', 'longwave_radiation (W/m²)', 
                                     'mean_sea_level_pressure (hPa)', 'geopotential_height [800mb](Gpm)', 'temperature (°C)', 
                                     'soil_temperature [0-7cm down](°C)']]
#print(weather_city.head())

# Function to calculate dew point
def calculate_dew_point(temp_celsius, relative_humidity):
    temp_fahrenheit = temp_celsius * 9/5 + 32
    vapor_pressure = 6.112 * np.exp((17.67 * temp_fahrenheit) / (temp_fahrenheit + 243.5)) * relative_humidity / 100
    dew_point_celsius = (243.5 * np.log(vapor_pressure / 6.112)) / (17.67 - np.log(vapor_pressure / 6.112))
    
    return dew_point_celsius

# Calculate dew point for each row
weather_city['dew_point_C'] = weather_city.apply(
    lambda row: calculate_dew_point(row['temp. [2m elevation corrected](°C)'], row['relative_humidity [2m](%)']), axis=1)


def classify_weather(row):
    temp = row['temp. [2m elevation corrected](°C)']
    precip = row['precipitation_total (mm)']
    cloud_cover = row['cloud_cover_total (%)']
    snowfall = row['snowfall_amount (cm)']
    wind_speed = row['wind_speed [10m](km/h)']
    humidity = row['relative_humidity [2m](%)']

    if snowfall > 0.1:
        return 'Snowy'
    elif precip > 0.1:
        return 'Rainy'
    elif cloud_cover >= 80 and precip > 0.1:
        return 'Rainy'
    elif cloud_cover >= 80 and precip == 0:
        return 'Overcast'
    elif cloud_cover >= 40 and cloud_cover < 80 and precip > 0.1:
        return 'Showers'
    elif cloud_cover >= 40 and cloud_cover < 80:
        return 'Partly Cloudy'
    elif cloud_cover < 40 and humidity > 80:
        return 'Foggy'
    elif wind_speed > 20:
        return 'Windy'
    elif cloud_cover < 20 and temp > 20:
        return 'Sunny'
    elif cloud_cover < 20:
        return 'Clear'
    else:
        return 'Clear'

# Apply the function to each row in the DataFrame
weather_city['weather_label'] = weather_city.apply(classify_weather, axis=1)

# Display a sample of the DataFrame to verify the new labels
print(weather_city.head(1000))

weather_data_train = 'weather_data_train'
if not os.path.exists(weather_data_train):
    os.makedirs(weather_data_train) 

file_path = os.path.join(weather_data_train, 'train_weather_data.csv')
weather_city.to_csv(file_path, index=False)
print('Data processed and saved in', file_path)
