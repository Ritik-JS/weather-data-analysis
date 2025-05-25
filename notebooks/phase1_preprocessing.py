# 📦 Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 📂 Create Plots Directory if Not Exists
os.makedirs("../plots", exist_ok=True)

# 📥 Load Dataset
df = pd.read_csv("../data/weather_data.csv")

# 🧹 Initial Cleaning and DateTime Conversion
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df.sort_values('timestamp')

# 🧠 Feature Engineering
df['Month'] = df['timestamp'].dt.month
df['Day'] = df['timestamp'].dt.day
df['Hour'] = df['timestamp'].dt.hour
df['Weekday'] = df['timestamp'].dt.day_name()

# Feels Like Temperature (Simplified formula)
df['Feels_Like'] = df['temperature'] - ((100 - df['humidity']) / 5) - (df['wind_speed'] * 0.7)

# 📊 Summary Statistics
print("=== Summary Statistics ===")
print(df.describe())

# 📉 Visualization 1: Temperature Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['temperature'], bins=30, kde=True, color='skyblue')
plt.title('Temperature Distribution')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.savefig('../plots/temp_distribution.png')
plt.close()

# 📅 Visualization 2: Monthly Average Temperature
plt.figure(figsize=(8, 5))
monthly_avg = df.groupby('Month')['temperature'].mean()
monthly_avg.plot(kind='bar', color='orange')
plt.title('Average Monthly Temperature')
plt.ylabel('Temperature (°C)')
plt.xlabel('Month')
plt.savefig('../plots/monthly_avg_temp.png')
plt.close()

# 🔥 Visualization 3: Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df[['temperature', 'humidity', 'wind_speed', 'Feels_Like']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('../plots/feature_correlation_heatmap.png')
plt.close()

# 💧 Visualization 4: Average Humidity by Hour
plt.figure(figsize=(10, 5))
avg_humidity_hourly = df.groupby('Hour')['humidity'].mean()
avg_humidity_hourly.plot(kind='line', marker='o')
plt.title('Average Humidity by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Humidity (%)')
plt.grid(True)
plt.savefig('../plots/hourly_humidity_trend.png')
plt.close()

print("✅ Phase 1 preprocessing and visualization completed.")
