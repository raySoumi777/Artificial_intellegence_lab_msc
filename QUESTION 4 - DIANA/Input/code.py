import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load the Iris dataset
df = pd.read_csv('iris.csv')

# Drop the 'Id' and 'Species' columns as they are not part of the feature set
X = df.drop(columns=['Id', 'Species']).values

# Linkage methods to evaluate
linkage_methods = ['single', 'complete', 'average']

# Plot dendrograms for each linkage method
plt.figure(figsize=(15, 5))
for i, method in enumerate(linkage_methods, 1):
    # Perform hierarchical clustering
    Z = linkage(X, method)

    # Plot dendrogram
    plt.subplot(1, 3, i)
    dendrogram(Z)
    plt.title(f'Hierarchical Clustering Dendrogram ({method.capitalize()} Linkage)')
    plt.xlabel('Data Point Index')
    plt.ylabel('Distance')

plt.tight_layout()
plt.show()
