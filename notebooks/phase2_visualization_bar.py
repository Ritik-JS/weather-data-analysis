import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("../data/weather_data.csv.csv")

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "YEAR"])

# Calculate average temperature for each month
monthly_avg = df.mean()

# Plot pie chart
plt.figure(figsize=(10, 8))
colors = plt.cm.viridis_r(monthly_avg / max(monthly_avg))  # color gradient
plt.pie(monthly_avg, labels=monthly_avg.index, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title("Average Temperature Contribution by Month (1901 onwards)", fontsize=14)
plt.tight_layout()

# Save the pie chart
plt.savefig("../plots/monthly_avg_pie_chart.png")
plt.show()
