import numpy as np

class MADALINE:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases
        self.hidden_weights = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.hidden_biases = np.random.uniform(-1, 1, hidden_size)
        self.output_weights = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.output_biases = np.random.uniform(-1, 1, output_size)
        self.learning_rate = learning_rate

    def activation(self, x):
        # Step function
        return np.where(x >= 0, 1, -1)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for i in range(len(X)):
                # Forward pass
                hidden_input = np.dot(self.hidden_weights, X[i]) + self.hidden_biases
                hidden_output = self.activation(hidden_input)
                final_input = np.dot(self.output_weights, hidden_output) + self.output_biases
                final_output = self.activation(final_input)

                # Error computation
                output_error = y[i] - final_output

                # Weight and bias updates (Output layer)
                self.output_weights += self.learning_rate * output_error[:, None] * hidden_output
                self.output_biases += self.learning_rate * output_error

                # Weight and bias updates (Hidden layer)
                hidden_error = output_error @ self.output_weights
                self.hidden_weights += self.learning_rate * hidden_error[:, None] * X[i]
                self.hidden_biases += self.learning_rate * hidden_error

    def predict(self, X):
        hidden_input = np.dot(self.hidden_weights, X) + self.hidden_biases
        hidden_output = self.activation(hidden_input)
        final_input = np.dot(self.output_weights, hidden_output) + self.output_biases
        final_output = self.activation(final_input)
        return final_output

# XOR Data with 3 inputs (simulating XOR for 3 inputs)
# We can extend XOR with three inputs as follows:
X = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
y = np.array([[-1], [1], [1], [-1], [1], [-1], [-1], [1]])  # Expected XOR output for three inputs

# Initialize MADALINE
madaline = MADALINE(input_size=3, hidden_size=4, output_size=1, learning_rate=0.1)

# Train the MADALINE on XOR data
madaline.train(X, y, epochs=100)

# Test the trained network
print("Testing on known XOR data with 3 inputs:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {madaline.predict(X[i])}, Expected Output: {y[i]}")

# Test with an unknown input (e.g., new combination)
unknown_input = np.array([1, -1, 1])
predicted_output = madaline.predict(unknown_input)
print(f"Unknown Input: {unknown_input}, Predicted Output: {predicted_output}")
