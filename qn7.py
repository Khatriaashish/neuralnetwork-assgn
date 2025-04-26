import numpy as np

# Majority Function Dataset
X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

Y = np.array([
    [0],
    [0],
    [0],
    [1],
    [0],
    [1],
    [1],
    [1]
])

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Training Function
def train_majority(X, Y, activation="sigmoid", epochs=10000, learning_rate=0.1, batch_mode=False):
    np.random.seed(42)

    w1 = np.random.randn(3, 2)
    b1 = np.zeros((1, 2))

    w2 = np.random.randn(2, 2)
    b2 = np.zeros((1, 2))

    w3 = np.random.randn(2, 1)
    b3 = np.zeros((1, 1))

    if activation == "sigmoid":
        act = sigmoid
        act_derivative = sigmoid_derivative
    elif activation == "tanh":
        act = tanh
        act_derivative = tanh_derivative
    else:
        raise ValueError("Invalid activation function")

    for epoch in range(epochs):
        if batch_mode:
            z1 = np.dot(X, w1) + b1
            a1 = act(z1)

            z2 = np.dot(a1, w2) + b2
            a2 = act(z2)

            z3 = np.dot(a2, w3) + b3
            a3 = act(z3)

            error = Y - a3
            dz3 = error * act_derivative(a3)
            dw3 = np.dot(a2.T, dz3)
            db3 = np.sum(dz3, axis=0, keepdims=True)

            dz2 = np.dot(dz3, w3.T) * act_derivative(a2)
            dw2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dz1 = np.dot(dz2, w2.T) * act_derivative(a1)
            dw1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            w1 += learning_rate * dw1
            b1 += learning_rate * db1
            w2 += learning_rate * dw2
            b2 += learning_rate * db2
            w3 += learning_rate * dw3
            b3 += learning_rate * db3

        else:
            for i in range(len(X)):
                x = X[i:i+1]
                y = Y[i:i+1]

                z1 = np.dot(x, w1) + b1
                a1 = act(z1)

                z2 = np.dot(a1, w2) + b2
                a2 = act(z2)

                z3 = np.dot(a2, w3) + b3
                a3 = act(z3)

                error = y - a3
                dz3 = error * act_derivative(a3)
                dw3 = np.dot(a2.T, dz3)
                db3 = dz3

                dz2 = np.dot(dz3, w3.T) * act_derivative(a2)
                dw2 = np.dot(a1.T, dz2)
                db2 = dz2

                dz1 = np.dot(dz2, w2.T) * act_derivative(a1)
                dw1 = np.dot(x.T, dz1)
                db1 = dz1

                w1 += learning_rate * dw1
                b1 += learning_rate * db1
                w2 += learning_rate * dw2
                b2 += learning_rate * db2
                w3 += learning_rate * dw3
                b3 += learning_rate * db3

        if epoch % 1000 == 0:
            loss = np.mean((Y - a3) ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return w1, b1, w2, b2, w3, b3

# Evaluation Function
def evaluate_majority(X, w1, b1, w2, b2, w3, b3, activation="sigmoid"):
    if activation == "sigmoid":
        act = sigmoid
    elif activation == "tanh":
        act = tanh

    a1 = act(np.dot(X, w1) + b1)
    a2 = act(np.dot(a1, w2) + b2)
    a3 = act(np.dot(a2, w3) + b3)
    return a3

# ==== START ====

# User chooses Activation
print("Choose Activation Function:")
print("1. Sigmoid")
print("2. Tanh")
activation_choice = input("Enter 1 or 2: ").strip()
if activation_choice == "1":
    activation_choice = "sigmoid"
elif activation_choice == "2":
    activation_choice = "tanh"
else:
    print("Invalid choice. Defaulting to Sigmoid.")
    activation_choice = "sigmoid"

# User chooses Training Mode
print("\nChoose Training Mode:")
print("1. Batch Gradient Descent")
print("2. Online Gradient Descent")
mode_choice = input("Enter 1 or 2: ").strip()
if mode_choice == "1":
    batch_mode = True
elif mode_choice == "2":
    batch_mode = False
else:
    print("Invalid choice. Defaulting to Batch mode.")
    batch_mode = True

# Training
print(f"\nTraining using {activation_choice.upper()} activation and {'BATCH' if batch_mode else 'ONLINE'} mode...\n")
w1, b1, w2, b2, w3, b3 = train_majority(X, Y, activation=activation_choice, epochs=10000, learning_rate=0.1, batch_mode=batch_mode)

# Evaluate
preds = evaluate_majority(X, w1, b1, w2, b2, w3, b3, activation=activation_choice)

# Show final predictions
print("\nFinal Predictions:")
print("Input\t\tExpected\tPredicted")
for i in range(len(X)):
    x1, x2, x3 = X[i]
    expected = Y[i][0]
    predicted = np.round(preds[i][0])
    print(f"{x1} {x2} {x3}\t\t  {expected}\t\t  {predicted}")

# === User input for Testing ===
while True:
    user_input = input("\nEnter 3 bits separated by spaces (e.g., '1 0 1') or type 'exit' to quit: ").strip()
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    try:
        bits = list(map(int, user_input.split()))
        if len(bits) != 3 or any(b not in (0, 1) for b in bits):
            print("Invalid input! Please enter exactly three 0 or 1 values.")
            continue
        bits_array = np.array(bits).reshape(1, -1)
        user_pred = evaluate_majority(bits_array, w1, b1, w2, b2, w3, b3, activation=activation_choice)
        user_pred_binary = np.round(user_pred[0][0])
        print(f"Predicted Output: {int(user_pred_binary)}")
    except Exception as e:
        print("Error:", e)
        continue
