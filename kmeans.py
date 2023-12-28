import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
np.random.seed(0)

file_name = sys.argv[1]

df = pd.read_csv(file_name, delim_whitespace = True, header = None).iloc[:,:-1]
clusters = {}
clusters_final = {}

for n_clusters in range(2, 11):
    best_error = float('inf')
    best_clusters = None

    for initialization_attempt in range(20):
        target_coordinate1 = df.iloc[np.random.randint(len(df))].values
        target_coordinate2 = df.iloc[np.random.randint(len(df))].values

        for i in range(20):
            distance1 = np.linalg.norm(df.values - target_coordinate1, axis=1)
            distance2 = np.linalg.norm(df.values - target_coordinate2, axis=1)
            cluster1 = df[distance1 < distance2].reset_index(drop=True)
            cluster2 = df[distance2 < distance1].reset_index(drop=True)
            target_coordinate1 = np.mean(cluster1, axis=0).values
            target_coordinate2 = np.mean(cluster2, axis=0).values

        sum1 = np.sum(np.linalg.norm(cluster1.values - target_coordinate1, axis=1))
        sum2 = np.sum(np.linalg.norm(cluster2.values - target_coordinate2, axis=1))
        total_error = sum1 + sum2
        if total_error < best_error:
            best_error = total_error
            best_clusters = [sum1, sum2, target_coordinate1, target_coordinate2, cluster1, cluster2]
    
    clusters[best_clusters[0]] = [best_clusters[2], best_clusters[4]]
    clusters[best_clusters[1]] = [best_clusters[3], best_clusters[5]]
    
    final_error = sum(clusters.keys())
    clusters_final[n_clusters] = final_error
    
    worst_cluster_key = max(clusters.keys())
    df = clusters[worst_cluster_key][1]
    del clusters[worst_cluster_key]

for k, error in clusters_final.items():
    print(f'For k = {k} After 20 iterations: Error = {error:.4f}')
    
plt.plot(clusters_final.keys(), clusters_final.values(), marker='o')
plt.title('Error Values by K')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Error')
plt.show()
