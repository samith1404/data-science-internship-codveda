import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load and combine churn data
df1 = pd.read_csv("churn-bigml-20.csv")
df2 = pd.read_csv("churn-bigml-80.csv")
df = pd.concat([df1, df2], ignore_index=True)

# Encode text columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("--- Model Results ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision : {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall    : {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1 Score  : {f1_score(y_test, y_pred)*100:.2f}%")

# Cross validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\nCross Validation Scores: {cv_scores.round(2)}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Not Churn', 'Churn']))

# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

print("\nTop 10 Most Important Features:")
for i in range(10):
    print(f"  {i+1}. {feature_names[indices[i]]}: "
          f"{importances[indices[i]]:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(10),
        importances[indices[:10]],
        color='steelblue')
plt.xticks(range(10),
           [feature_names[indices[i]] for i in range(10)],
           rotation=45, ha='right')
plt.title('Top 10 Feature Importances - Random Forest')
plt.tight_layout()
plt.savefig('task7_feature_importance.png')
plt.close()
print("\nFeature importance plot saved!")
print("\nTask 7 Complete!")