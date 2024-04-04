from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
# Membaca data dari file CSV
data = pd.read_csv('C:/Users/salma/Documents/coding/datamining(teori&praktikum)/teori/tugas3/price.csv')

# Menghapus kolom non-numerik
data_numeric = data.select_dtypes(include=['number'])

# Menghapus baris dengan nilai NaN
data_numeric = data_numeric.dropna()

# Menentukan nilai K menggunakan Elbow Method atau Silhouette Method
# Elbow Method
inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_numeric)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 10), inertia, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Berdasarkan grafik Elbow Method, kita dapat memilih nilai K yang optimal
# Misalnya, kita pilih nilai K = 3

# Melakukan pengelompokan menggunakan K-Means dengan nilai K yang sudah ditentukan
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_numeric)

# Melakukan pengelompokan menggunakan K-Medoids dengan nilai K yang sudah ditentukan
kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids_labels = kmedoids.fit_predict(data_numeric)

# Evaluasi pengelompokan menggunakan silhouette score
silhouette_kmeans = silhouette_score(data_numeric, kmeans_labels)
silhouette_kmedoids = silhouette_score(data_numeric, kmedoids_labels)

print("Silhouette Score (K-Means):", silhouette_kmeans)
print("Silhouette Score (K-Medoids):", silhouette_kmedoids)

# Ilustrasi hasil pengelompokan
plt.scatter(data_numeric.iloc[:, 0], data_numeric.iloc[:, 1], c=kmeans_labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='red', s=200)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

plt.scatter(data_numeric.iloc[:, 0], data_numeric.iloc[:, 1], c=kmedoids_labels, cmap='viridis')
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], marker='x', c='red', s=200)
plt.title('K-Medoids Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()