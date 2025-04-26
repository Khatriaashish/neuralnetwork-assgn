import numpy as np

#XOR DATASET
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

#ACTIVATION FUNCTION
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

#ALGORITHM
def train_xor(X, Y, activation="sigmoid", epochs=10000, learning_rate=0.1, batch_mode=False):
    #Initialize weights and biases
    np.random.seed(42)
    w1 = np.random.rand(2, 2) #INPUT TO HIDDEN
    b1 = np.zeros((1, 2))
    w2 = np.random.rand(2, 1) #HIDDEN TO OUTPUT
    b2 = np.zeros((1, 1))

    #Choose activation
    if activation == "sigmoid":
        activation_function = sigmoid
        activation_derivative = sigmoid_derivative
    elif activation == "tanh":
        activation_function = tanh
        activation_derivative = tanh_derivative
    else:
        raise ValueError("Invalid activation function")
    
    #Train
    for epoch in range(epochs):
        if batch_mode:
            #forward pass
            z1 = np.dot(X, w1) + b1
            a1 = activation_function(z1)
            z2 = np.dot(a1, w2) + b2
            a2 = activation_function(z2)

            #backpropagation
            error = Y - a2
            dz2 = error * activation_derivative(a2)
            dw2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dz1 = np.dot(dz2, w2.T) * activation_derivative(a1)
            dw1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0)

            #updates
            w1 += learning_rate * dw1
            b1 += learning_rate * db1
            w2 += learning_rate * dw2
            b2 += learning_rate * db2

        else:
            for i in range(len(X)):
                x = X[i:i+1]
                y = Y[i:i+1]

                #forward pass
                z1 = np.dot(x, w1) + b1
                a1 = activation_function(z1)
                z2 = np.dot(a1, w2) + b2
                a2 = activation_function(z2)

                #backpropagation
                error = y - a2
                dz2 = error * activation_derivative(a2)
                dw2 = np.dot(a1.T, dz2)
                db2 = dz2

                dz1 = np.dot(dz2, w2.T) * activation_derivative(a1)
                dw1 = np.dot(x.T, dz1)
                db1 = dz1

                #updates
                w1 += learning_rate * dw1
                b1 += learning_rate * db1
                w2 += learning_rate * dw2
                b2 += learning_rate * db2

            if epoch % 1000 == 0:
                loss = np.mean((Y - a2) ** 2)
                print(f"Epoch {epoch}, Loss: {loss: .4f}")

    return w1, b1, w2, b2

#EVALUATION
def evaluate_xor(X, w1, b1, w2, b2, activation="sigmoid"):
    if activation == "sigmoid":
        activation_function = sigmoid
    elif activation == "tanh":
        activation_function = tanh

    a1 = activation_function(np.dot(X, w1) + b1)
    a2 = activation_function(np.dot(a1, w2) + b2)
    return a2

# MAIN
def main():
    print("Train XOR ANN (2x2x1)")
    activation = input("Choose activation function (sigmoid/tanh): ").strip().lower()
    mode = input("Choose training mode (batch/online): ").strip().lower()
    epochs = int(input("Number of training epochs [default: 10000]: ") or 10000)
    lr = float(input("Learning rate [default: 0.1]: ") or 0.1)

    batch_mode = True if mode == "batch" else False

    print("\nTraining started...\n")
    w1, b1, w2, b2 = train_xor(X, Y, activation=activation, epochs=epochs, learning_rate=lr, batch_mode=batch_mode)
    
    preds = evaluate_xor(X, w1, b1, w2, b2, activation=activation)
    print("Input\tExpected\tPredicted")
    for i in range(len(X)):
        x1, x2 = X[i]
        expected = Y[i][0]
        predicted = np.round(preds[i][0])
        print(f"{x1} {x2}\t   {expected}\t\t   {predicted}")

if __name__ == "__main__":
    main()