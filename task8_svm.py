import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, roc_curve, auc,
                             classification_report)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load Iris dataset
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

# Try Linear and RBF kernels
print("--- Comparing Kernels ---")
for kernel in ['linear', 'rbf']:
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Kernel={kernel} -> Accuracy: {acc*100:.2f}%")

# Final model with best kernel (rbf)
print("\n--- Final Model with RBF Kernel ---")
best_model = SVC(kernel='rbf', probability=True, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f"Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision : {precision_score(y_test, y_pred, average='macro')*100:.2f}%")
print(f"Recall    : {recall_score(y_test, y_pred, average='macro')*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=le.classes_))

# Decision boundary using only 2 features for visualization
X_2d = X_scaled[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42
)

svm_2d = SVC(kernel='rbf', random_state=42)
svm_2d.fit(X_train_2d, y_train_2d)

# Plot decision boundary
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
colors = ['red', 'blue', 'green']
for i, cls in enumerate(le.classes_):
    mask = y == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=colors[i], label=cls, s=50)
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title('SVM Decision Boundary - Iris Dataset')
plt.legend()
plt.tight_layout()
plt.savefig('task8_svm_boundary.png')
plt.close()
print("\nDecision boundary saved as task8_svm_boundary.png")
print("\nTask 8 Complete!")