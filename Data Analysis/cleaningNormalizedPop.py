import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'Life Expectancy Data.csv'  # Adjust this to your file location
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()
data_cleaned = data.dropna(subset=['Life expectancy']).copy()

# Standardize the 'Population' feature
scaler = StandardScaler()
data_cleaned.loc[:, 'Population_std'] = scaler.fit_transform(data_cleaned[['Population']])

# One-hot encode categorical variables
data_cleaned = pd.get_dummies(data_cleaned, columns=['Status'])

# Prepare features and target variable for modeling
X = data_cleaned.drop(['Life expectancy', 'Country', 'Population', 'Population_std'], axis=1)  # Also exclude 'Population_std'
y = data_cleaned['Life expectancy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importance analysis
feature_importances = rf_model.feature_importances_
features = X.columns

# Plotting feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance')
plt.show()
