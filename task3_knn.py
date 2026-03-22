import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("1) iris.csv")

# Encode target
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Try different values of K and compare
print("--- Trying different K values ---")
for k in [1, 3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  K={k} -> Accuracy: {acc*100:.2f}%")

# Use best K=5 for final evaluation
print("\n--- Final Model with K=5 ---")
best_model = KNeighborsClassifier(n_neighbors=5)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=le.classes_))

print("\nTask 3 Complete!")