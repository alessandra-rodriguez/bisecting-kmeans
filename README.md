# Bisecting K Means

## Description:
The script reads a dataset from a text file and performs bisecting K-Means clustering. The output displays the error for the number of clusters (K) ranging from 2 to 10. The error is defined as the sum of the Euclidean distances between data points and their respective cluster centroids.

Two random data points are chosen as initial cluster centroids. Cluster centroids are iteratively updated and data points are assigned to clusters based the nearest centroid. This process is repeated 20 times, and the clustering with the minimum error is selected for each K number of clusters.

# Features:
* Dynamic K Selection: The script explores a range of cluster numbers (k) from 2 to 10, calculating the error for each k
* Random Initialization: 20 sets of inital centroids are chosen to enhance the chances of finding a global minimum for the error
* Iterative Refinement: Clusters are refined by itteratively updating centroids after data points are assigned to clusters based on the nearest centroids
* Error Visualization: Errors for each k are plotted to visualize the trade-off between the number of clusters and the resulting error

## Usage:
1. Clone this repo locally
2. Install and update relevant libraries
3. Save your dataset in a text file with space-separated numeric values, with the last column as the class label. Example data is provided.
4. Run the script from the command line, providing the dataset file as an argument: 
python3 kmeans.py dataset.txt