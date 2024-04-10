import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

file_path = 'Life Expectancy Data.csv'
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()
data_cleaned = data.dropna(subset=['Life expectancy']).copy()

scaler = StandardScaler()
data_cleaned.loc[:, 'Population_std'] = scaler.fit_transform(data_cleaned[['Population']])

data_cleaned = pd.get_dummies(data_cleaned, columns=['Status'])

X = data_cleaned.drop(['Life expectancy', 'Country', 'Population', 'Population_std', 'Year'], axis=1)
y = data_cleaned['Life expectancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

feature_importances = rf_model.feature_importances_
features = X.columns
feature_importances = feature_importances / max(feature_importances)

top_10_features = sorted(zip(feature_importances, features), reverse=True)[:10]
for importance, feature in top_10_features:
    if feature != 'Year':
        print(f"{feature}: {importance}")

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance')
plt.show()
