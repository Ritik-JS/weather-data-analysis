import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
os.makedirs("../plots", exist_ok=True)

# Load the data
df = pd.read_csv("../data/weather_data.csv.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Drop rows with invalid timestamps
df = df.dropna(subset=['timestamp'])

# Extract year, month, and decade
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['decade'] = (df['year'] // 10) * 10

# Drop rows with null temperature
df = df.dropna(subset=['temperature'])

# Plot 1: Monthly temperature trends over time
monthly_avg = df.groupby(['year', 'month'])['temperature'].mean().reset_index()
monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))
plt.figure(figsize=(12,6))
sns.lineplot(x='date', y='temperature', data=monthly_avg)
plt.title("Monthly Average Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.tight_layout()
plt.savefig("../plots/monthly_trends.png")
plt.close()

# Plot 2: Heatmap of average temperature per month per decade
heatmap_data = df.groupby(['decade', 'month'])['temperature'].mean().unstack()
plt.figure(figsize=(10,6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".1f")
plt.title("Average Monthly Temperature per Decade")
plt.xlabel("Month")
plt.ylabel("Decade")
plt.tight_layout()
plt.savefig("../plots/heatmap_decade_avg.png")
plt.close()

# Plot 3: Correlation between monthly average temperatures
month_temp = df.pivot_table(index='year', columns='month', values='temperature', aggfunc='mean')
corr_matrix = month_temp.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
plt.title("Correlation Between Monthly Temperatures")
plt.tight_layout()
plt.savefig("../plots/monthly_correlation.png")
plt.close()

print("Phase 1 preprocessing complete. Plots saved in ../plots/")
