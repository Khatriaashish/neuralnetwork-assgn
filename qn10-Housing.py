import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

# Load your dataset
df = pd.read_csv('lib/reference_dataset/Housing.csv')

# 1. Check for missing values and handle them
print("Missing Values:\n", df.isnull().sum())
df = df.dropna()  # or you can use df.fillna() for imputation

# 2. Display input and output features
print("\nInput Features:\n", df.columns[:-1])
print("\nOutput Feature:\n", df.columns[-1])

# 3. Encode non-numeric input attributes
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 4. Normalize input and output attributes
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = df.iloc[:, :-1].values  # All columns except last
y = df.iloc[:, -1].values.reshape(-1, 1)  # Last column (target)

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 5. Split dataset into 70:15:15
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\nTrain set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

# 6. Construct the MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # input layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # output layer (no activation)
])

model.compile(optimizer='adam', loss='mse')

# 7. Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1
)

# 8. Predict house price for test data
y_pred_scaled = model.predict(X_test)

# 9. Perform inverse transformation
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# 10. Compute and display RMSE, MAE and MAPE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
epsilon = 1e-8  # Small value to prevent division by zero
mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

print(f"\nRMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
