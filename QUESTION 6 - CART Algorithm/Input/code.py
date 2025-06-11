# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset (replace with your actual path if needed)
df = pd.read_csv("Buy_Computer.csv")

# Remove leading/trailing spaces from column names, if any
df.columns = df.columns.str.strip()

# Display the column names to ensure they match
print(df.columns)

# Convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Display the first few rows of the dataset
print(df_encoded.head())  # Fixed the missing parenthesis here

# Separate features (X) and target variable (y)
X = df_encoded.drop('Buy_Computer_yes', axis=1).values  # Corrected target column name
y = df_encoded['Buy_Computer_yes'].values  # Corrected target column name

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the Decision Tree Classifier model
model = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)

# Visualize the decision tree
plt.figure(figsize=(12, 20))
plot_tree(model, feature_names=df_encoded.columns[:-1], class_names=['Not Buy', 'Buy'], filled=True)
plt.title("Decision Tree for Buys Computer Dataset")
plt.show()
