import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt

# memazsukan data/memuat data
data = pd.read_csv('C:/Users/salma/Documents/coding/datamining(teori&praktikum)/teori/tugas6/price.csv')

# Pengcodean data kategorikal
label_encoder = LabelEncoder()
data['Parking'] = label_encoder.fit_transform(data['Parking'])
data['City_Category'] = label_encoder.fit_transform(data['City_Category'])
# Impute NaN values with column means
data.fillna(data.mean(), inplace=True)

# Standarisasi data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop('Observation', axis=1))

# Hierarchical Clustering
def hierarchical_clustering(data, n_clusters):
    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = hierarchical_cluster.fit_predict(data)
    return cluster_labels

# DBSCAN
def dbscan_clustering(data, eps, min_samples):
    dbscan_cluster = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan_cluster.fit_predict(data)
    return cluster_labels

# menentukan optimal number dari  clusters untuk Hierarchical Clustering
def optimal_hierarchical_clusters(data):
    silhouette_scores = []
    for n_clusters in range(2, 11):
        cluster_labels = hierarchical_clustering(data, n_clusters)
        silhouette_scores.append(silhouette_score(data, cluster_labels))
    return silhouette_scores.index(max(silhouette_scores)) + 2



# menentukan parameter optimal untuk DBSCAN
def optimal_dbscan_params(data):
    eps_values = [0.1, 0.5, 1, 2, 3]
    min_samples_values = [2, 5, 10, 20]
    max_silhouette_score = -1
    optimal_eps = None
    optimal_min_samples = None
    for eps in eps_values:
        for min_samples in min_samples_values:
            cluster_labels = dbscan_clustering(data, eps, min_samples)
            unique_labels = len(set(cluster_labels))
            if unique_labels > 1:
                silhouette_avg = silhouette_score(data, cluster_labels)
                if silhouette_avg > max_silhouette_score:
                    max_silhouette_score = silhouette_avg
                    optimal_eps = eps
                    optimal_min_samples = min_samples
    return optimal_eps, optimal_min_samples

# melakukan Hierarchical Clustering
optimal_clusters_hierarchical = optimal_hierarchical_clusters(scaled_data)
hierarchical_cluster_labels = hierarchical_clustering(scaled_data, optimal_clusters_hierarchical)

# melakukan DBSCAN
optimal_eps_dbscan, optimal_min_samples_dbscan = optimal_dbscan_params(scaled_data)
dbscan_cluster_labels = dbscan_clustering(scaled_data, optimal_eps_dbscan, optimal_min_samples_dbscan)

# Evaluasi silhouette scores
silhouette_score_hierarchical = silhouette_score(scaled_data, hierarchical_cluster_labels)
silhouette_score_dbscan = silhouette_score(scaled_data, dbscan_cluster_labels)

print(f'Silhouette Score for Hierarchical Clustering: {silhouette_score_hierarchical}')
print(f'Silhouette Score for DBSCAN: {silhouette_score_dbscan}')

# Plot hasil nya
plt.figure(figsize=(12, 6))

# Plot hasil untuk Pengelompokan Hierarchical Clustering
plt.subplot(1, 2, 1)
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=hierarchical_cluster_labels, cmap='viridis')
plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()

# Plot the results for DBSCAN
plt.subplot(1, 2, 2)
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=dbscan_cluster_labels, cmap='viridis')
plt.title('DBSCAN')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()

plt.show()

# menentukan algotima mana yang silhouette score mana yang optimal
if silhouette_score_hierarchical > silhouette_score_dbscan:
    print('Hierarchical Clustering has the optimal silhouette score.')
    print('Visualisasi hasil Hierarchical Clustering:')
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=hierarchical_cluster_labels, cmap='viridis')
    plt.title('Hierarchical Clustering (Optimal)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()
else:
    print('DBSCAN has the optimal silhouette score.')
    print('Visualisasi hasil DBSCAN:')
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=dbscan_cluster_labels, cmap='viridis')
    plt.title('DBSCAN (Optimal)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()
