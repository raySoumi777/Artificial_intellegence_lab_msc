import numpy as np

# Define activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Mean Squared Error (MSE) Loss function and its derivative
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return -(y_true - y_pred)

# Define Neural Network
class NeuralNetwork:
    def __init__(self):
        np.random.seed(42)  # For reproducibility
        
        # Initialize weights and biases for a 3-2-2-2 network
        self.weights = {
            "w1": np.random.randn(3, 2),  # Input (3) → Hidden1 (2)
            "w2": np.random.randn(2, 2),  # Hidden1 (2) → Hidden2 (2)
            "w3": np.random.randn(2, 2)   # Hidden2 (2) → Output (2)
        }
        self.biases = {
            "b1": np.zeros((1, 2)),  # Bias for first hidden layer
            "b2": np.zeros((1, 2)),  # Bias for second hidden layer
            "b3": np.zeros((1, 2))   # Bias for output layer
        }

    # Forward pass
    def forward(self, x):
        self.z1 = np.dot(x, self.weights["w1"]) + self.biases["b1"]
        self.a1 = relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.weights["w2"]) + self.biases["b2"]
        self.a2 = relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.weights["w3"]) + self.biases["b3"]
        self.a3 = self.z3  # Linear activation in the output layer
        return self.a3

    # Backward pass
    def backward(self, x, y_true, y_pred, learning_rate):
        d_loss = mse_loss_derivative(y_true, y_pred)

        # Backpropagation through output layer
        d_z3 = d_loss
        d_w3 = np.dot(self.a2.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        # Backpropagation through second hidden layer
        d_a2 = np.dot(d_z3, self.weights["w3"].T)
        d_z2 = d_a2 * relu_derivative(self.z2)
        d_w2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        # Backpropagation through first hidden layer
        d_a1 = np.dot(d_z2, self.weights["w2"].T)
        d_z1 = d_a1 * relu_derivative(self.z1)
        d_w1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        # Update weights and biases
        self.weights["w3"] -= learning_rate * d_w3
        self.biases["b3"] -= learning_rate * d_b3
        self.weights["w2"] -= learning_rate * d_w2
        self.biases["b2"] -= learning_rate * d_b2
        self.weights["w1"] -= learning_rate * d_w1
        self.biases["b1"] -= learning_rate * d_b1

# Initialize the network
net = NeuralNetwork()

# Training parameters
sample_input = np.array([[1.0, 0.0, 1.0]])  # Input sample
target_label = np.array([[1.0, 0.0]])       # Target class label
learning_rate = 0.01
num_epochs = 100

# Train the network
for epoch in range(num_epochs):
    # Forward pass
    output = net.forward(sample_input)
    loss = mse_loss(target_label, output)

    # Print the predicted output for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Predicted Output: {output}")

    # Backward pass
    net.backward(sample_input, target_label, output, learning_rate)

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

# Test the network
output = net.forward(sample_input)
print("Final Predicted Output:", output)
