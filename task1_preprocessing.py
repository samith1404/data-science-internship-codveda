import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("1) iris.csv")

# Check the data
print("First 5 rows:")
print(df.head())

print("\nShape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode species column
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

print("\nAfter encoding:")
print(df.head())

# Scale the features
scaler = StandardScaler()
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df[features] = scaler.fit_transform(df[features])

# Split into train and test
X = df[features]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)
print("\nTask 1 Complete!")