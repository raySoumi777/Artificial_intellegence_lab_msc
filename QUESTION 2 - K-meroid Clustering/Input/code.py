import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

# Read the Iris dataset
df = pd.read_csv("Iris.csv")

# Initialize lists and variables
cluster1 = []
cluster2 = []
cluster3 = []
prev1, prev2, prev3 = [], [], []
rows, _ = df.shape

# Initialize centroids (you can also randomly select or set them based on your choice)
petal_length1 = df.iloc[0]['PetalLengthCm']
petal_width1 = df.iloc[0]['PetalWidthCm']
petal_length2 = df.iloc[1]['PetalLengthCm']
petal_width2 = df.iloc[1]['PetalWidthCm']
petal_length3 = df.iloc[2]['PetalLengthCm']
petal_width3 = df.iloc[2]['PetalWidthCm']

# Function to compute distance between points
def compute_distance(petal_length1, petal_width1, petal_length2, petal_width2):
    return math.sqrt((petal_length1 - petal_length2)**2 + (petal_width1 - petal_width2)**2)

# K-Medoids Clustering
while True:
    # Clear previous clusters
    cluster1.clear()
    cluster2.clear()
    cluster3.clear()

    # Assign points to nearest cluster
    for i in range(rows):
        petal_length = df.iloc[i]['PetalLengthCm']
        petal_width = df.iloc[i]['PetalWidthCm']
        
        # Calculate distance from each centroid
        dist1 = compute_distance(petal_length1, petal_width1, petal_length, petal_width)
        dist2 = compute_distance(petal_length2, petal_width2, petal_length, petal_width)
        dist3 = compute_distance(petal_length3, petal_width3, petal_length, petal_width)
        
        # Assign the point to the nearest centroid
        if dist1 < dist2 and dist1 < dist3:
            cluster1.append(i)
        elif dist2 < dist1 and dist2 < dist3:
            cluster2.append(i)
        else:
            cluster3.append(i)

    # Print species in each cluster
    print("Cluster 1 Species:", [df.iloc[i]['Species'] for i in cluster1])
    print("Cluster 2 Species:", [df.iloc[i]['Species'] for i in cluster2])
    print("Cluster 3 Species:", [df.iloc[i]['Species'] for i in cluster3])
    print("-------------------------------------")
    print("Previous Clusters:")
    print(prev1)
    print(prev2)
    print(prev3)
    print("-------------------------------------")

    # Check for convergence (clusters are stable)
    if prev1 == cluster1 and prev2 == cluster2 and prev3 == cluster3:
        plt.figure(figsize=(10, 6))

        # Plot points for each cluster
        plt.scatter([df.iloc[i]['PetalLengthCm'] for i in cluster1],
                   [df.iloc[i]['PetalWidthCm'] for i in cluster1],
                   c='red', label='Cluster 1')
        plt.scatter([df.iloc[i]['PetalLengthCm'] for i in cluster2],
                   [df.iloc[i]['PetalWidthCm'] for i in cluster2],
                   c='blue', label='Cluster 2')
        plt.scatter([df.iloc[i]['PetalLengthCm'] for i in cluster3],
                   [df.iloc[i]['PetalWidthCm'] for i in cluster3],
                   c='green', label='Cluster 3')

        # Plot centroids
        plt.scatter(petal_length1, petal_width1, c='red', marker='*', s=200, label='Centroid 1')
        plt.scatter(petal_length2, petal_width2, c='blue', marker='*', s=200, label='Centroid 2')
        plt.scatter(petal_length3, petal_width3, c='green', marker='*', s=200, label='Centroid 3')

        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Petal Width (cm)')
        plt.title('K-Medoids Clustering of Iris Dataset')
        plt.legend()
        plt.grid(True)
        plt.show()
        break
    else:
        prev1, prev2, prev3 = cluster1.copy(), cluster2.copy(), cluster3.copy()

    # Recalculate centroids (mean of points in each cluster)
    if len(cluster1) > 0:
        petal_length1 = sum(df.iloc[i]['PetalLengthCm'] for i in cluster1) / len(cluster1)
        petal_width1 = sum(df.iloc[i]['PetalWidthCm'] for i in cluster1) / len(cluster1)
    if len(cluster2) > 0:
        petal_length2 = sum(df.iloc[i]['PetalLengthCm'] for i in cluster2) / len(cluster2)
        petal_width2 = sum(df.iloc[i]['PetalWidthCm'] for i in cluster2) / len(cluster2)
    if len(cluster3) > 0:
        petal_length3 = sum(df.iloc[i]['PetalLengthCm'] for i in cluster3) / len(cluster3)
        petal_width3 = sum(df.iloc[i]['PetalWidthCm'] for i in cluster3) / len(cluster3)

    # Print new centroids
    print("New Centroids:")
    print("-------------------------------------")
    print(f"Centroid 1: Length={petal_length1:.2f}, Width={petal_width1:.2f}")
    print(f"Centroid 2: Length={petal_length2:.2f}, Width={petal_width2:.2f}")
    print(f"Centroid 3: Length={petal_length3:.2f}, Width={petal_width3:.2f}")
    print("-------------------------------------")
