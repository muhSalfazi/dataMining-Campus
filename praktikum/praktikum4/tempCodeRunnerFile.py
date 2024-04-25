from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Memuat data Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Menggunakan Agglomerative Hierarchical Clustering
agg_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
df['cluster'] = agg_cluster.fit_predict(df)

# Visualisasi dendogram
plt.figure(figsize=(12,8))
linked = linkage(df.iloc[:, :-1], 'ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

from scipy.cluster.hierarchy import cophenet
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist

# Menghitung Cophenetic Correlation Coeffiecient
coph_corr, _ = cophenet(linked, pdist(df.iloc[:, :-1]))

# Menghitung Silhouette Score 
Silhouette_avg = silhouette_score(df.iloc[:, :-1], df['cluster'])
print(f'cophenetic Correlation Coefficient: {coph_corr}')
print(f'Silhouette Score for Hierarchical Clustering: {Silhouette_avg}')

# visualisassi pengelompokan menggunakan scatter plot
import seaborn as sns

# visualiasasi pengelompokan menggunakan scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x ='sepal length(cm)',y='sepal width(cm)',hue='cluster',data=df,palette='viridis',s=100)
plt.title('hierarchical clustering of iris dataset')
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.legend()
plt.show()