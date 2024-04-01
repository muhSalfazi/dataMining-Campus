# 1)import library yang akan digunakan
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

# 2)load datasets
#  memuat data iris
iris = load_iris()
df= pd.DataFrame(iris.data, columns=iris.feature_names)

#3)menentukan nilai K terbaik dengan metode elbow
distortions =[]
k_range = range(1,10)
for k in k_range:  # Menggunakan k sebagai iterasi bukan i
    Kmeans = KMeans(n_clusters=k)  # Menggunakan k sebagai jumlah cluster
    Kmeans.fit(df)
    distortions.append(Kmeans.inertia_)
    
# ploting elbow method
plt.plot(k_range,distortions,marker='o' )
plt.title("Elbow Method For Optimal K(K-Means)")
plt.xlabel("Number of Cluster(K)")
plt.ylabel("Distortion")
plt.show()  # Memanggil plt.show() untuk menampilkan plot

print("=======================================")
# 5)implementasi K-Means dan Evaluasi dengan silhouette
# implementasi kmeans dan evaluasi silhouette
from sklearn.metrics import silhouette_score
# menenttukan nilai K yang optimal
optimal_K = 3
# melakukan K-Means dengan K optimal
kmeans = KMeans(n_clusters=k)
df['cluster']=kmeans.fit_predict(df)
# menghitung silhouette dengan score 
silhouette_svg = silhouette_score(df,df['cluster'])
print(f'silhouette score for K-Means :{silhouette_svg}')

# 6)visualisasi hasil pengelompokan dengan scatterplot
import seaborn as sns 

# visualisasi pengelopokan menggunakan scatterplot
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='cluster', data=df, palette='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='red', marker='x', label='Centroids')
plt.title('K-Means Clustering of iris dataset')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend()
plt.show()  # Memanggil plt.show() untuk menampilkan plot
