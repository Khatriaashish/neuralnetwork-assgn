import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(2)  # Initialize weights randomly
        self.bias = np.random.rand(1)  # Initialize bias randomly
    
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

# Training data for AND gate
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y = np.array([-1, -1, -1, 1])

# Train perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=10)
perceptron.train(X, y)

# Test perceptron
print("Trained Perceptron Outputs:")
for inputs in X:
    print(f"Input: {inputs} -> Output: {perceptron.predict(inputs)}")
