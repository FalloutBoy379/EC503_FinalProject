import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and clean the dataset
file_path = './Life Expectancy Data.csv'  # Update to your file path
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

# Imputation function to fill missing values with column mean
def impute_with_mean(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_cols:
        if column != 'Year':
            df[column] = df[column].fillna(df[column].mean())
    return df

# Apply imputation grouped by 'Country'
data_cleaned = data.groupby('Country', as_index=False).apply(impute_with_mean).reset_index(drop=True)

# Encode 'Status' and handle potential NaNs in 'Life expectancy'
data_cleaned['Status'] = data_cleaned['Status'].map({'Developed': 1, 'Developing': 0})
data_cleaned = data_cleaned.dropna(subset=['Life expectancy'])  # Drop rows where 'Life expectancy' is NaN

# Exploratory Data Analysis: Distribution of 'Life expectancy'
plt.figure(figsize=(10, 6))
sns.histplot(data_cleaned['Life expectancy'], kde=True, bins=30)
plt.title('Distribution of Life Expectancy')
plt.show()

# Save the cleaned data
cleaned_file_path = './Life_Expectancy_Data_Cleaned.csv'  # Update to your desired path
data_cleaned.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data exported to {cleaned_file_path}")

# Prepare data for modeling
X = data_cleaned.drop(['Life expectancy', 'Country'], axis=1)
y = data_cleaned['Life expectancy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)

# Calculate feature importances
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(10, 8))
features = X.columns
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance')
plt.show()
