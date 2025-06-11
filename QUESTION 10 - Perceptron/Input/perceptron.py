import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def perceptron_learning(X, y, eta=1, epochs=10, theta=1):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    for epoch in range(epochs):
        errors = 0
        for i in range(n_samples):
            net_input = np.dot(X[i], weights)
            if net_input > theta:
                output = 1
            elif net_input <= theta and net_input >= -theta:
                output = 0
            else:
                output = -1
            if output != y[i]:
                weights += eta * y[i] * X[i]
                errors += 1
        print(f"Epoch {epoch + 1}/{epochs}: Weights = {weights}, Errors = {errors}")
        if errors == 0:
            print("No errors, training converged.")
            break
    return weights


def test_perceptron_with_auc(X_test, y_test, weights, theta=1):
    net_inputs = []
    predictions = []
    for i in range(len(X_test)):
        net_input = np.dot(X_test[i], weights)
        net_inputs.append(net_input)
        if net_input > theta:
            output = 1
        elif net_input <= theta and net_input >= -theta:
            output = 0
        else:
            output = -1
        predictions.append(output)
    y_test_binary = np.where(y_test == -1, 0, 1)
    net_inputs_binary = np.array(net_inputs)
    auc_score = roc_auc_score(y_test_binary, net_inputs_binary)
    fpr, tpr, _ = roc_curve(y_test_binary, net_inputs_binary)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve ")
    plt.legend(loc="lower right")
    plt.show()
    return predictions, auc_score

def run_logic_gate_experiment(X_train, y_train, X_test, y_test, gate_name):
    print(f"Logic Gate: {gate_name}")
    weights = perceptron_learning(X_train, y_train, eta=1, epochs=10)
    print(f"Weights: {weights}")
    predictions, auc_score = test_perceptron_with_auc(X_test, y_test, weights)
    print(f"Predictions: {predictions}")
    print(f"AUC-ROC Score: {auc_score:.2f}\n")

# Define inputs and targets for OR, XOR, XNOR, NOR, NAND, AND gates
X_train = np.array([
    [1, 1, 1],   # Bias term x0 = 1, followed by two inputs
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1]
])  # 4 combinations of two binary inputs plus bias

logic_gates = {
    "OR": {
        "y_train": np.array([1, 1, 1, -1]),  # OR outputs
        "X_test": X_train,
        "y_test": np.array([1, 1, 1, -1])
    },
    "XOR": {
        "y_train": np.array([-1, 1, 1, -1]),  # XOR outputs
        "X_test": X_train,
        "y_test": np.array([-1, 1, 1, -1])
    },
    "XNOR": {
        "y_train": np.array([1, -1, -1, 1]),  # XNOR outputs
        "X_test": X_train,
        "y_test": np.array([1, -1, -1, 1])
    },
    "NOR": {
        "y_train": np.array([-1, -1, -1, 1]),  # NOR outputs
        "X_test": X_train,
        "y_test": np.array([-1, -1, -1, 1])
    },
    "NAND": {
        "y_train": np.array([1, 1, 1, -1]),  # NAND outputs
        "X_test": X_train,
        "y_test": np.array([1, 1, 1, -1])
    },
    "AND": {
        "y_train": np.array([1, -1, -1, -1]),  # AND outputs
        "X_test": X_train,
        "y_test": np.array([1, -1, -1, -1])
    }
}

# Run experiments for each logic gate
for gate, data in logic_gates.items():
    run_logic_gate_experiment(data["X_test"], data["y_train"], data["X_test"], data["y_test"], gate)
