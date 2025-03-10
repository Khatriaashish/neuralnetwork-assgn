import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(2)  
        self.bias = np.random.rand(1)  
    
    def activation(self, x):
        return 1 if x >= 0 else -1
    
    def train(self, X, y):
        for _ in range(self.epochs):
            for inputs, expected in zip(X, y):
                weighted_sum = np.dot(inputs, self.weights) + self.bias
                output = self.activation(weighted_sum)
                error = expected - output
                
                # Update weights and bias
                self.weights += self.learning_rate * error * np.array(inputs)
                self.bias += self.learning_rate * error
    
    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation(weighted_sum)
        
def min_max_scaler(data, feature_range=(0, 1)):

    min_val, max_val = feature_range
    min_data = np.min(data, axis=0)  # Find min of each column
    max_data = np.max(data, axis=0)  # Find max of each column
    
    # Avoid division by zero if min == max for a feature
    if np.any(max_data == min_data):
        return np.zeros_like(data) if min_val == 0 else np.full_like(data, min_val)
    
    scaled_data = (data - min_data) / (max_data - min_data) * (max_val - min_val) + min_val
    return scaled_data

# Training data
data = np.array([
    [5.9, 75],
    [5.8, 86],
    [5.2, 50],
    [5.4, 55],
    [6.1, 85],
    [5.5, 62]
])

# Labels: 1 for Male, -1 for Female
labels = np.array([1, 1, -1, -1, 1, -1])

# Normalize the data using Min-Max scaling (for all columns/features)
normalized_data = min_max_scaler(data)


# Initialize and train the Perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=10)
perceptron.train(normalized_data, labels)

# Test inputs
test_inputs = np.array([
    [6, 82],
    [5.3, 52]
])

# Normalize the test inputs using the same scaling
normalized_test_inputs = min_max_scaler(test_inputs)

# Make predictions for test inputs
predictions = [perceptron.predict(test_input) for test_input in normalized_test_inputs]

# Output predictions
for test_input, prediction in zip(test_inputs, predictions):
    result = "Male" if prediction == 1 else "Female"
    print(f"Input: {test_input} -> Predicted Class: {result}")
