import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load the dataset
df = pd.read_csv(r'golf-dataset.csv')

# Check the column names to confirm their exact names
print(df.columns)

# Encoding the categorical columns
enc = LabelEncoder()
df_num_cat = pd.DataFrame()

# Correcting the column name to match the dataset
df_num_cat['Outlook'] = enc.fit_transform(df['Outlook'])
df_num_cat['Temperature'] = enc.fit_transform(df['Temperature'])  # Corrected column name
df_num_cat['Humidity'] = enc.fit_transform(df['Humidity'])
df_num_cat['Windy'] = enc.fit_transform(df['Windy'])
df_num_cat['Play Golf'] = enc.fit_transform(df['Play'])  # Assuming 'Play' column is 'Play Golf'

# Setting X as input (features) and Y as output (target)
X = df_num_cat.drop(['Play Golf'], axis=1)
y = df_num_cat['Play Golf']

# Initialize and train the decision tree classifier
dt_clf = DecisionTreeClassifier(criterion="entropy")
dt_clf.fit(X, y)

# Evaluate the model
y_pred = dt_clf.predict(X)

# Print the true and predicted labels
print(y)
print(y_pred)

# Plot the decision tree
plt.figure(figsize=(10, 10))
tree.plot_tree(dt_clf, filled=True)
plt.savefig('tree_id3.png', format='png', bbox_inches="tight")

# Compute and print confusion matrix and other metrics
lbs = [0, 1]
CF = sm.confusion_matrix(y, y_pred, labels=lbs)
acc = sm.accuracy_score(y, y_pred)
p = sm.precision_score(y, y_pred, labels=lbs, pos_label=0)
r = sm.recall_score(y, y_pred, labels=lbs, pos_label=0)
f1 = sm.f1_score(y, y_pred, labels=lbs, pos_label=0)

print("Confusion matrix: \n", CF)
print("Accuracy: ", acc)
print("Precision score: ", p)
print("Recall score: ", r)
print("F1 Score: ", f1)
