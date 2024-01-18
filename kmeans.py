# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
np.random.seed(0)

# Read the input file name from CLI
file_name = sys.argv[1]

# Read the data from the file, considering whitespace as delimiter and excluding the last column
# Assumes last column contains class of instance
df = pd.read_csv(file_name, delim_whitespace = True, header = None).iloc[:,:-1]

# Use dicts to hold intermediate and final results
clusters = {}
clusters_final = {}

# Iterate over different numbers of clusters (k) from 2 to 10
for n_clusters in range(2, 11):
    best_error = float('inf')
    best_clusters = None

    # Attempt initialization 20 times
    for initialization_attempt in range(20):
        # Choose 2 random points from the cluster in order to bisect
        target_coordinate1 = df.iloc[np.random.randint(len(df))].values
        target_coordinate2 = df.iloc[np.random.randint(len(df))].values

        # Readjust clusters by finding new centroids
        for i in range(20):
            # Find the euclidean distances between each data point and target_coordinate1
            distance1 = np.linalg.norm(df.values - target_coordinate1, axis=1)
            # Find the euclidean distances between all the data points and target_coordinate2
            distance2 = np.linalg.norm(df.values - target_coordinate2, axis=1)

            # If the point is closer to target_coordinate1, assign to cluster 1
            cluster1 = df[distance1 < distance2].reset_index(drop=True)
            # If the point is closer to target_coordinate2, assign to cluster 2
            cluster2 = df[distance2 < distance1].reset_index(drop=True)

            # Find the new centroids of each cluster
            target_coordinate1 = np.mean(cluster1, axis=0).values
            target_coordinate2 = np.mean(cluster2, axis=0).values

        # For both clusters, sum the errors between each point in the cluster and its centroid
        sum1 = np.sum(np.linalg.norm(cluster1.values - target_coordinate1, axis=1))
        sum2 = np.sum(np.linalg.norm(cluster2.values - target_coordinate2, axis=1))
        total_error = sum1 + sum2

        # Determine best initializatino attempt by finding lowest total error
        if total_error < best_error:
            best_error = total_error
            best_clusters = [sum1, sum2, target_coordinate1, target_coordinate2, cluster1, cluster2]
    
    # Store best clustering results
    clusters[best_clusters[0]] = [best_clusters[2], best_clusters[4]]
    clusters[best_clusters[1]] = [best_clusters[3], best_clusters[5]]
    
    #  Calculate the final cumulative error amongst the clusters and store final results
    final_error = sum(clusters.keys())
    clusters_final[n_clusters] = final_error
    
    # Find the worst performing cluster by most error
    # Worst cluster will be bisected in next iteration
    worst_cluster_key = max(clusters.keys())
    df = clusters[worst_cluster_key][1]
    del clusters[worst_cluster_key]

for k, error in clusters_final.items():
    print(f'For k = {k} After 20 iterations: Error = {error:.4f}')

# Plot the error values against the number of clusters
plt.plot(clusters_final.keys(), clusters_final.values(), marker='o')
plt.title('Error Values by K')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Error')
plt.show()
