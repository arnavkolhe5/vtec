import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('vtec_data_year_2020.csv')

# Convert "DATE AND TIME" to datetime format and extract month
data['DATE AND TIME'] = pd.to_datetime(data['DATE AND TIME'], format='%d-%m-%Y %H:%M')
data['Month'] = data['DATE AND TIME'].dt.month

# Group by month and take the mean of observed and predicted values
monthly_data = data.groupby('Month').mean()

# Perform linear regression
X = monthly_data['VTEC OBSERVED'].values.reshape(-1, 1)  # Observed values (X-axis)
y = monthly_data['VTEC PREDICTED'].values  # Predicted values (Y-axis)

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Plot the graph
plt.figure(figsize=(8, 6))
sns.scatterplot(x=monthly_data['VTEC OBSERVED'], y=monthly_data['VTEC PREDICTED'], label='Actual Data')
plt.plot(monthly_data['VTEC OBSERVED'], y_pred, color='red', label='Regression Line')

# Labels and title
plt.xlabel('Observed VTEC')
plt.ylabel('Predicted VTEC')
plt.title('Observed vs Predicted VTEC (Monthly Average)')
plt.legend()

# Show plot
plt.show()