import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
# Replace 'boston.csv' with the actual path to your CSV file
#dataset_path = "HousingData.csv"
data = pd.read_csv("Iris.csv")

# Ensure the dataset contains only numeric features (remove target if included)
if 'Species' in data.columns:  # Assuming 'MEDV' is the target column
    data = data.drop('Species', axis=1)

# Normalize the dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply Fuzzy C-Means clustering
n_clusters = 3  # Define the number of clusters
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    scaled_data.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
)

# Assign cluster membership
cluster_membership = np.argmax(u, axis=0)

# Add the cluster membership to the original data
data['Cluster'] = cluster_membership

# Visualize the results (using the first two features for simplicity)
plt.figure(figsize=(8, 6))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['Cluster'], cmap='viridis', s=50)
plt.title("Fuzzy C-Means Clustering on IRIS Dataset")
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.colorbar(label="Cluster")
plt.show()