import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'vtec_data_year_2020.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Convert the 'DATE AND TIME' column to datetime format
data['DATE AND TIME'] = pd.to_datetime(data['DATE AND TIME'], format='%d-%m-%Y %H:%M')

# Extract the month from the datetime
data['Month'] = data['DATE AND TIME'].dt.strftime('%Y-%m')

# Group the data by month and calculate the mean of observed and predicted VTEC values
monthly_data = data.groupby('Month').mean()

# Plot the line chart for observed and predicted values month-wise
plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['VTEC OBSERVED'], label='Observed VTEC', color='blue', marker='o')
plt.plot(monthly_data.index, monthly_data['VTEC PREDICTED'], label='Predicted VTEC', color='red', marker='x')

# Customize the chart
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('VTEC Values')
plt.title('Month-wise Observed vs Predicted VTEC Values')
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()