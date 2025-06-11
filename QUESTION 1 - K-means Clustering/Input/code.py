from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
import math

file_path = "Iris.csv"
df = pd.read_csv(file_path)

cluster1 = []
cluster2 = []
cluster3 = []
cl = 0
sepal_length = 0
sepal_width = 0
rows, _ = df.shape
prev1 = []
prev2 = []
prev3 = []

# Initialize centroids (you could use the first 3 data points or any random initialization)
sepal_length1, sepal_width1 = df.iloc[0]['SepalLengthCm'], df.iloc[0]['SepalWidthCm']
sepal_length2, sepal_width2 = df.iloc[1]['SepalLengthCm'], df.iloc[1]['SepalWidthCm']
sepal_length3, sepal_width3 = df.iloc[2]['SepalLengthCm'], df.iloc[2]['SepalWidthCm']

while(1):
    cluster1.clear()
    cluster2.clear()
    cluster3.clear()

    for i in range(rows):
        sepal_length = df.iloc[i]['SepalLengthCm']
        sepal_width = df.iloc[i]['SepalWidthCm']
        
        dist1 = math.sqrt(pow((sepal_length1 - sepal_length), 2) + pow((sepal_width1 - sepal_width), 2))
        dist2 = math.sqrt(pow((sepal_length2 - sepal_length), 2) + pow((sepal_width2 - sepal_width), 2))
        dist3 = math.sqrt(pow((sepal_length3 - sepal_length), 2) + pow((sepal_width3 - sepal_width), 2))

        if dist1 < dist2:
            if dist1 < dist3:
                cluster1.append(i)
                cl = 1
            else:
                cluster3.append(i)
                cl = 3
        else:
            if dist2 < dist3:
                cluster2.append(i)
                cl = 2
            else:
                cluster3.append(i)
                cl = 3
    
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
    
    if prev1 == cluster1 and prev2 == cluster2 and prev3 == cluster3:
        plt.figure(figsize=(10, 6))
        # Plot points for each cluster
        plt.scatter([df.iloc[i]['SepalLengthCm'] for i in cluster1],
                    [df.iloc[i]['SepalWidthCm'] for i in cluster1],
                    c='red', label='Cluster 1')
        plt.scatter([df.iloc[i]['SepalLengthCm'] for i in cluster2],
                    [df.iloc[i]['SepalWidthCm'] for i in cluster2],
                    c='blue', label='Cluster 2')
        plt.scatter([df.iloc[i]['SepalLengthCm'] for i in cluster3],
                    [df.iloc[i]['SepalWidthCm'] for i in cluster3],
                    c='green', label='Cluster 3')
        # Plot centroids
        plt.scatter(sepal_length1, sepal_width1, c='red', marker='*', s=200, label='Centroid 1')
        plt.scatter(sepal_length2, sepal_width2, c='blue', marker='*', s=200, label='Centroid 2')
        plt.scatter(sepal_length3, sepal_width3, c='green', marker='*', s=200, label='Centroid 3')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Sepal Width (cm)')
        plt.title('K-Means Clustering of Iris Dataset')
        plt.legend()
        plt.grid(True)
        plt.show()
        break
    else:
        prev1 = cluster1.copy()
        prev2 = cluster2.copy()
        prev3 = cluster3.copy()
    
    # Recalculate centroids
    sepal_length1 = sum(df.iloc[i]['SepalLengthCm'] for i in cluster1) / len(cluster1) if len(cluster1) > 0 else sepal_length1
    sepal_width1 = sum(df.iloc[i]['SepalWidthCm'] for i in cluster1) / len(cluster1) if len(cluster1) > 0 else sepal_width1
    
    sepal_length2 = sum(df.iloc[i]['SepalLengthCm'] for i in cluster2) / len(cluster2) if len(cluster2) > 0 else sepal_length2
    sepal_width2 = sum(df.iloc[i]['SepalWidthCm'] for i in cluster2) / len(cluster2) if len(cluster2) > 0 else sepal_width2
    
    sepal_length3 = sum(df.iloc[i]['SepalLengthCm'] for i in cluster3) / len(cluster3) if len(cluster3) > 0 else sepal_length3
    sepal_width3 = sum(df.iloc[i]['SepalWidthCm'] for i in cluster3) / len(cluster3) if len(cluster3) > 0 else sepal_width3
    
    print("New Centroids:")
    print("-------------------------------------")
    print(f"Centroid 1: Length={sepal_length1:.2f}, Width={sepal_width1:.2f}")
    print(f"Centroid 2: Length={sepal_length2:.2f}, Width={sepal_width2:.2f}")
    print(f"Centroid 3: Length={sepal_length3:.2f}, Width={sepal_width3:.2f}")
    print("-------------------------------------")
