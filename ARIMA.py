"""
Implementing the ARIMA Mathematical Model to forecast Life Expectancy using the WHO dataset
"""
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Life Expectancy Data.csv')

# Filter data for the period 2000-2015
data = data[(data['Year'] >= 2000) & (data['Year'] <= 2015)]

# Prepare training and testing sets
train_data = data[data['Year'] <= 2013]
test_data = data[data['Year'] >= 2013]

train_life_expectancy = train_data.groupby('Year')['Life expectancy'].mean()
test_life_expectancy = test_data.groupby('Year')['Life expectancy'].mean()

# Plot ACF and PACF for Life Expectancy
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(train_life_expectancy, ax=plt.gca(), title="ACF for Life Expectancy")
plt.subplot(122)
plot_pacf(train_life_expectancy, ax=plt.gca(), title="PACF for Life Expectancy")
plt.show()

# Define ARIMA parameters based on ACF and PACF plots
p, d, q = 1, 1, 0  

# Fit the ARIMA model on life expectancy
model = ARIMA(train_life_expectancy, order=(p, d, q))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test_life_expectancy))

# Plotting the results
plt.figure()
plt.plot(train_life_expectancy.index, train_life_expectancy, label='Train Life Expectancy')
plt.plot(test_life_expectancy.index, test_life_expectancy, label='Actual Life Expectancy (Test)')
plt.plot(test_life_expectancy.index, forecast, label='Forecasted Life Expectancy')
plt.title('Forecast vs Actual Life Expectancy')
plt.legend()
plt.show()

# Calculate error
mse = mean_squared_error(test_life_expectancy, forecast)
print(f"Mean Mean Squared Error for Life Expectancy: {mse}")

# Align data by taking only the corresponding years
test_years = test_life_expectancy.index
forecast_values = forecast.values[:len(test_years)]  # Extract forecasted values for test years
squared_errors = (test_life_expectancy - forecast_values) ** 2

# Convert years to integers
test_years = test_life_expectancy.index.astype(int)

# Plot MSE as a bar graph
plt.figure()
plt.bar(test_years, squared_errors, color='skyblue')
plt.title('Mean Squared Error (MSE) for Life Expectancy Forecast')
plt.xlabel('Year')
plt.ylabel('Mean Squared Error')
plt.xticks(test_years)  # Set integer ticks
plt.legend()
plt.show()

