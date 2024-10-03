import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'vtec_data_year_2020.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Extract observed and predicted values
observed_values = data['VTEC OBSERVED'].values.reshape(-1, 1)
predicted_values = data['VTEC PREDICTED'].values.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(observed_values, predicted_values)

# Generate predictions for the regression line
predicted_line = model.predict(observed_values)

# Plot the observed vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(observed_values, predicted_values, color='blue', label='Actual Data', alpha=0.6)
plt.plot(observed_values, predicted_line, color='red', label='Regression Line')

# Labels and title
plt.xlabel('Observed VTEC')
plt.ylabel('Predicted VTEC')
plt.title('Observed vs Predicted VTEC Regression')

# Show legend
plt.legend()

# Display the plot
plt.show()