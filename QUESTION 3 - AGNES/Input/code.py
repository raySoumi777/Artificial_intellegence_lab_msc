import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
# Sample data: 10 points in 2D space
data = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 6],
    [8, 8],
    [8, 9],
    [25, 25],
    [24, 24],
    [27, 27],
    [28, 28]
])
# Function to perform agglomerative clustering and plot dendrograms
def agglomerative_clustering(data, linkage_method):
    # Compute the linkage matrix
    Z = linkage(data, method=linkage_method)
    # Plot the dendrogram
    plt.figure(figsize=(8, 4))
    plt.title(f"Agglomerative Clustering with {linkage_method.capitalize()} Linkage")
    dendrogram(Z, labels=np.arange(1, len(data) + 1))
    plt.xlabel("Data Point Index")
    plt.ylabel("Distance")
    plt.show()
# Compute and display dendrograms for each linkage method
for method in ["single", "complete", "average"]:
    agglomerative_clustering(data, method)
