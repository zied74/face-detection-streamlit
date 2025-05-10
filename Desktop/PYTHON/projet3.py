import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import streamlit as st
import joblib
import pandas as pd

# Load your dataset (replace 'your_data.csv' with your actual file)
df = pd.read_csv (r'C:\\Users\PC\Desktop\PYTHON\Financial_inclusion_dataset.csv')

# Display first few rows
print("First 5 rows:")
print(df.head())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Data shape
print("\nData shape:", df.shape)
# Display dataset information
print("\nDataset information:")
print(df.info())

# Column data types
print("\nData types:")
print(df.dtypes)

# Count of unique values in each column
print("\nUnique values per column:")
print(df.nunique())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values (choose appropriate strategy)
# For numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())  # or mean()

# For categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Alternatively, drop rows with missing values if appropriate
# df = df.dropna()

# Handle corrupted values (example: replace negative values with median)
for col in num_cols:
    df[col] = np.where(df[col] < 0, df[col].median(), df[col])
    # Check for duplicates
print("\nNumber of duplicates:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Function to detect and handle outliers using IQR
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Apply to all numerical columns
for col in num_cols:
    df = handle_outliers(df, col)

# Alternative visualization approach to check outliers
plt.figure(figsize=(15, 10))
df[num_cols].boxplot()
plt.title('Boxplot of Numerical Features After Outlier Handling')
plt.xticks(rotation=45)
plt.show()
# Identify categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

# Label Encoding for ordinal categories or when you want to convert to numerical
label_encoder = LabelEncoder()
for col in cat_cols:
    if df[col].nunique() <= 10:  # Threshold for label encoding
        df[col] = label_encoder.fit_transform(df[col])

# One-Hot Encoding for nominal categories
df = pd.get_dummies(df, columns=[col for col in cat_cols if df[col].nunique() > 10], drop_first=True)

# Display encoded dataframe
print("\nData after encoding:")
print(df.head())
print("\nFinal dataset shape:", df.shape)
print("\nMissing values after processing:")
print(df.isnull().sum())
print("\nData types after processing:")
print(df.dtypes)
# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = df
X = data.drop('job_type', axis=1)
y = data['job_type']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.joblib')


# Load model
model = joblib.load('model.joblib')

st.title('ML Prediction App')

# Input fields
st.header('Feature Input')
feature1 = st.number_input('Feature 1')
feature2 = st.number_input('Feature 2')
# Add all your features...

# Prediction button
if st.button('Predict'):
    input_data = pd.DataFrame([[feature1, feature2]], columns=['Feature1', 'Feature2'])
    prediction = model.predict(input_data)[0]
    st.success(f'Prediction: {prediction}')
    import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
   