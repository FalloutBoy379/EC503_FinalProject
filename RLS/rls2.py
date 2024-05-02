import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def recursive_least_squares(X, y, delta=1.0, lam=0.99):
    n_samples, n_features = X.shape
    P = delta * np.eye(n_features)  # Initialize the inverse covariance matrix
    theta = np.zeros(n_features)    # Initialize the coefficient vector
    predictions = np.zeros(n_samples)  # Array to store predictions
    mse_history = np.zeros(n_samples)  # Array to store MSE

    for i in range(n_samples):
        x_i = X[i, :]
        y_i = y[i]
        prediction = x_i.dot(theta)
        predictions[i] = prediction
        error = y_i - prediction
        mse_history[i] = error**2

        K = (P @ x_i) / (lam + x_i.T @ P @ x_i)  # Gain vector
        theta += K * error  # Update coefficients
        P = (P - np.outer(K, x_i.T @ P)) / lam  # Update inverse covariance matrix

    return theta, predictions, mse_history

# Data preparation
data_path = 'Life_Expectancy_Data_Cleaned.csv'
data = pd.read_csv(data_path)
data = data.sort_values(by=['Year'])  # Ensure data is sorted by Year
data_subset = data[["Year", "Life expectancy", "HIV/AIDS", "Income composition of resources", "Adult Mortality", "BMI", "Schooling"]].copy()
data_subset.fillna(0, inplace=True)

# Prepare the data matrix X and target vector y
X = data_subset[["HIV/AIDS", "Income composition of resources", "Adult Mortality", "BMI", "Schooling"]].values
y = data_subset["Life expectancy"].values
years = data_subset["Year"].values  # Extract year information
X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add constant for intercept

# Run Recursive Least Squares
theta, predictions, mse_history = recursive_least_squares(X, y)

mse_history = mse_history / mse_history.max()  # Normalize MSE values

# Plotting Time Series of Actual vs Predicted Life Expectancy with Year
plt.figure(figsize=(12, 6))
plt.plot(years, y, label='Actual Life Expectancy', marker='o', linestyle='-')
plt.plot(years, predictions, label='Predicted Life Expectancy', marker='x', linestyle='--')
plt.title('Actual vs Predicted Life Expectancy Over Years')
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.legend()
plt.grid(True)
plt.show()

specific_years = [2013, 2014, 2015]

# Find indices corresponding to these years
indices = [i for i, year in enumerate(years) if year in specific_years]
specific_mse = mse_history[indices]

# Create a dictionary to aggregate MSE by year
year_mse = {}
for year, mse in zip(years[indices], specific_mse):
    if year not in year_mse:
        year_mse[year] = []
    year_mse[year].append(mse)

# Average MSE for these years (assuming multiple entries per year)
avg_mse_by_year = {year: np.mean(mse) for year, mse in year_mse.items()}

# Convert to lists for plotting
years_to_plot = list(avg_mse_by_year.keys())
mse_values = list(avg_mse_by_year.values())

# Plotting MSE for the last three years
plt.figure(figsize=(10, 6))
plt.bar([str(year) for year in years_to_plot], mse_values, color='blue')
plt.title('MSE for Specific Years: 2013, 2014, 2015')
plt.xlabel('Year')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

# Plotting Predicted vs Actual Life Expectancy
plt.figure(figsize=(12, 6))
plt.plot(y, predictions, 'o', label='Predicted vs Actual')
plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', label='Perfect Prediction')  # Line for perfect prediction
plt.title('Predicted vs Actual Life Expectancy')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.legend()
plt.grid(True)
plt.show()
