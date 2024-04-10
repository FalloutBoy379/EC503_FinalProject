import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and prepare data
file_path = 'Life Expectancy Data.csv'
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

data_cleaned = data.dropna(subset=['Life expectancy']).copy()

# Standardize 'Population'
scaler = StandardScaler()
data_cleaned.loc[:, 'Population_std'] = scaler.fit_transform(data_cleaned[['Population']])

# Encode categorical 'Status' column
data_cleaned = pd.get_dummies(data_cleaned, columns=['Status'])

# Prepare features and target
X = data_cleaned.drop(['Life expectancy', 'Country', 'Population', 'Population_std', 'Year'], axis=1)
y = data_cleaned['Life expectancy']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importances
feature_importances = rf_model.feature_importances_
features = X.columns
feature_importances = feature_importances / feature_importances.max()

# Plotting feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance')
plt.show()

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate and plot MSE
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

# Residuals plot
residuals = y_test - y_test_pred
plt.figure(figsize=(10, 8))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Life Expectancy')
plt.ylabel('Residuals')
plt.title('Residuals of Predictions')
plt.show()
