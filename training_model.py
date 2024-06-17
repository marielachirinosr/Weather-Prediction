import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib
import seaborn as sns

# Libraries
import matplotlib.pyplot as plt

# Load data
weather_data = pd.read_csv('weather_data_train/train_weather_data.csv')

# Feature Engineering
features = ['year', 'month', 'day', 'hour',
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

# Split the data into features and target variable
X = weather_data[features]
y = weather_data['weather_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler to a file
folder_name = 'scaler_folder'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

scaler_filename = os.path.join(folder_name, 'scaler.pkl')
joblib.dump(scaler, scaler_filename)

# Model Selection - RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)  
y_test_pred = model.predict(X_test_scaled)   

# Evaluate the model
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred)}")
print(f"Testing Accuracy: {accuracy_score(y_test, y_test_pred)}")
print("Classification Report on Test Data:")
print(classification_report(y_test, y_test_pred))

# Generate classification report
report = classification_report(y_test, y_test_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Plot the classification report
plt.figure(figsize=(10, 6))
sns.heatmap(df_report.iloc[:-1, :].T, annot=True, cmap='Blues')

plt.title('Classification Report')
plt.ylabel('Metrics')
plt.xlabel('Classes')

# Save the plot as a PNG file
plt.savefig('classification_report.png')

# Show the plot
plt.show()

# Create DataFrames for y_train and y_train_pred
train_results = pd.DataFrame({'y_train_true': y_train, 'y_train_pred': y_train_pred})

# Create DataFrames for y_test and y_test_pred
test_results = pd.DataFrame({'y_test_true': y_test, 'y_test_pred': y_test_pred})

# Print the DataFrames
print("Training DataFrame:")
print(train_results.head())
print("\nTesting DataFrame:")
print(test_results.head())

# Save the model
os.makedirs('models', exist_ok=True) 
joblib.dump(model, 'models/weather_data.pkl')
