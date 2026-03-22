import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load Iris dataset
df = pd.read_csv("1) iris.csv")

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(df['species'])
y_cat = to_categorical(y_encoded, num_classes=3)

# Features
X = df.drop('species', axis=1).values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# Build Neural Network
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),  # Input layer
    Dense(8,  activation='relu'),                    # Hidden layer
    Dense(3,  activation='softmax')                  # Output layer
])

model.summary()

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.1,
    verbose=0
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n--- Model Results ---")
print(f"Test Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('task9_neural_network.png')
plt.close()
print("\nTraining plot saved as task9_neural_network.png")
print("\nTask 9 Complete!")
print("\n*** ALL 9 TASKS DONE! INTERNSHIP TASKS COMPLETE! ***")