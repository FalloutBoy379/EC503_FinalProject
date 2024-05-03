import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

file_path = 'C:/Users/anshm/OneDrive/Desktop/EC503/EC503_FinalProject/Life Expectancy Data.csv'
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

data_cleaned = data.dropna(subset=['Life expectancy']).copy()

population = data_cleaned['Population']
population_std = (population - population.mean()) / population.std()
data_cleaned['Population'] = population_std

data_cleaned = pd.get_dummies(data_cleaned, columns=['Status'])

X = data_cleaned.drop(['Life expectancy', 'Country'], axis=1)
y = data_cleaned['Life expectancy']

# Generate a new csv file with the feature columns remaining in X
# Normalize all the columns
X_normalized = (X - X.mean()) / X.std()
X_normalized.to_csv('new_data.csv', index=False)
y = (y - y.mean()) / y.std()

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

feature_importances = rf_model.feature_importances_
features = X.columns
feature_importances = feature_importances / feature_importances.max()

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance')
plt.show()

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
plt.figure(figsize=(10, 6))
plt.bar(['Train MSE', 'Test MSE'], [mse_train, mse_test], color=['blue', 'green'])
plt.title('Mean Squared Error for Training and Test Sets')
plt.ylabel('MSE')
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred, alpha=0.5, label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Actual vs Predicted Life Expectancy')
plt.legend()
plt.show()

residuals = y_test - y_test_pred
plt.figure(figsize=(10, 8))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Life Expectancy')
plt.ylabel('Residuals')
plt.title('Residuals of Predictions')
plt.show()

# Export the cleaned dataset to a CSV file named 'Life_Expectancy_Data_Cleaned.csv'
data_cleaned.to_csv('C:/Users/anshm/OneDrive/Desktop/EC503/EC503_FinalProject/Life_Expectancy_Data_Cleaned.csv', index=False)
