import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Load dataset
dataset_path = "Buy_Computer.csv"
data = pd.read_csv(dataset_path)

# Encode categorical features and target label
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':  # Encode only categorical columns
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le  # Store encoder for inverse transformation

# Separate features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column is the target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict on test data
y_pred = nb_model.predict(X_test)

# Model accuracy
accuracy = nb_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Fixing the warning: Convert `new_sample` to DataFrame with column names
new_sample = pd.DataFrame([[1, 0, 0, 0, 1]], columns=X.columns)
print("Predicted Class Labels for Unknown Samples:")
print(new_sample)
# Predict class for new sample
predicted_class = nb_model.predict(new_sample)

# Reverse encoding for the predicted class
predicted_class_label = label_encoders[list(data.columns)[-1]].inverse_transform(predicted_class)
print(f"Predicted Class for New Sample: {predicted_class_label[0]}")
