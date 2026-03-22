import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load dataset with correct column names
col_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
    'LSTAT', 'PRICE'
]

df = pd.read_csv("4) house Prediction Data Set.csv",
                 header=0,
                 names=col_names,
                 sep=r'\s+',
                 engine='python')

print("First 5 rows:")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nShape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features and target
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n--- Model Results ---")
print(f"R-squared : {r2:.4f}")
print(f"MSE       : {mse:.2f}")
print(f"RMSE      : {np.sqrt(mse):.2f}")

print("\nModel Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"  {col}: {coef:.4f}")

print("\nTask 2 Complete!")