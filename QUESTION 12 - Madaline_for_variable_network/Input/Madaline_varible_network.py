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
        """ Step function activation """
        return np.where(x >= 0, 1, -1)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            total_errors = 0
            for i in range(len(X)):
                # Forward pass
                hidden_input = np.dot(self.hidden_weights, X[i]) + self.hidden_biases
                hidden_output = self.activation(hidden_input)
                final_input = np.dot(self.output_weights, hidden_output) + self.output_biases
                final_output = self.activation(final_input)

                # Error computation
                output_error = y[i] - final_output
                total_errors += np.sum(np.abs(output_error))

                # Update Output Layer weights and biases
                self.output_weights += self.learning_rate * output_error[:, None] * hidden_output
                self.output_biases += self.learning_rate * output_error

                # Update Hidden Layer weights and biases
                hidden_error = output_error @ self.output_weights
                self.hidden_weights += self.learning_rate * hidden_error[:, None] * X[i]
                self.hidden_biases += self.learning_rate * hidden_error

            # Print training progress
            print(f"Epoch {epoch + 1}/{epochs}, Total Errors: {total_errors}")

            # Stop training if no errors
            if total_errors == 0:
                print("Training Converged.")
                break

    def predict(self, X):
        """ Perform forward pass to get predictions """
        hidden_input = np.dot(self.hidden_weights, X) + self.hidden_biases
        hidden_output = self.activation(hidden_input)
        final_input = np.dot(self.output_weights, hidden_output) + self.output_biases
        final_output = self.activation(final_input)
        return final_output

def run_madaline_experiment(input_size, hidden_size, output_size, learning_rate, epochs, X_train, y_train, X_test, y_test):
    """ Train and test the MADALINE model with the given parameters """
    print(f"\nTraining MADALINE with {input_size} Inputs, {hidden_size} Hidden Neurons, {output_size} Outputs")
    madaline = MADALINE(input_size, hidden_size, output_size, learning_rate)
    madaline.train(X_train, y_train, epochs)

    print("\nTesting MADALINE:")
    for i in range(len(X_test)):
        pred = madaline.predict(X_test[i])
        print(f"Input: {X_test[i]}, Predicted Output: {pred}, Expected Output: {y_test[i]}")

    return madaline  # Return trained model for further testing

# 🛠 **Define XOR Data for 3 Inputs**
X_data = np.array([
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], 
    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
])
y_data = np.array([
    [-1], [1], [1], [-1], [1], [-1], [-1], [-1]
])  # Expected XOR output for three inputs

# 🔧 **Set Network Parameters (Change as needed)**
input_size = 3  # Number of input neurons (features)
hidden_size = 4  # Number of hidden layer neurons (can be changed)
output_size = 1  # Number of output neurons
learning_rate = 0.1
epochs = 100

# Train the MADALINE network
madaline_model = run_madaline_experiment(input_size, hidden_size, output_size, learning_rate, epochs, X_data, y_data, X_data, y_data)

# 🧪 **Test with an Unknown Input**
unknown_input = np.array([1, -1, 1])
predicted_output = madaline_model.predict(unknown_input)
print(f"\nUnknown Input: {unknown_input}, Predicted Output: {predicted_output}")
