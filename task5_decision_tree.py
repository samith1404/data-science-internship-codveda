import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("1) iris.csv")

# Encode target
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train WITHOUT pruning first ---
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("--- Without Pruning ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# --- Train WITH pruning (max_depth limits tree size) ---
model_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
model_pruned.fit(X_train, y_train)
y_pred_pruned = model_pruned.predict(X_test)

print("\n--- With Pruning (max_depth=3) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_pruned)*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_pruned,
      target_names=le.classes_))

# Print tree structure in text
print("\nTree Structure:")
print(export_text(model_pruned,
      feature_names=list(X.columns)))

# Visualize the tree
plt.figure(figsize=(12, 6))
plot_tree(model_pruned,
          feature_names=list(X.columns),
          class_names=le.classes_,
          filled=True,
          rounded=True)
plt.title("Decision Tree - Iris Dataset")
plt.tight_layout()
plt.savefig("task5_decision_tree.png")
plt.show()
print("\nTree saved as task5_decision_tree.png")
print("\nTask 5 Complete!")