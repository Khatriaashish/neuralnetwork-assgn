import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load dataset
df = pd.read_csv('lib/reference_dataset/heart.csv')  

# 2. Check missing values
print("\nChecking for missing values...")
print(df.isnull().sum())

# Fill missing values if any
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)


# 3. Display input and output features
print("\nInput features:")
print(df.columns[:-1].tolist())

print("\nOutput feature:")
print(df.columns[-1])

# 4. Encode non-numeric input attributes
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# 5. Split dataset into X and y
X = df.drop(columns=[df.columns[-1]])  # all except last column
y = df[df.columns[-1]]  # target column

# Normalize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Build MLP Model (11x128x64x32x1)
model = Sequential([
    Dense(128, input_dim=11, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8. Train the model
print("\nTraining the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# 9. Predict on test data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# 10. Evaluate performance
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
