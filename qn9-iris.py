# iris_mlp.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('lib/reference_dataset/iris.csv')

# Check missing values
print("\nChecking for missing values...")
print(df.isnull().sum())

# Fill missing values if any
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)
# Display input and output features
print("\nInput features:")
print(df.columns[:-1].tolist())

print("\nOutput feature:")
print(df.columns[-1])

# Display counts for each class
print("Class distribution:\n", df['Species '].value_counts())

# Encode output feature using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(df[['Species ']])

# Separate input and output
X = df.drop('Species ', axis=1).values

# Normalize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Shuffle and Split data (70:15:15)
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42, stratify=y_temp)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Build MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=1)

# Evaluate and predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, digits=4))
