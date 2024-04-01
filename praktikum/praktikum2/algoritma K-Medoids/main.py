from sklearn_extra.cluster import KMedoids
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

#2) load dataset
# membua data iris
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
# 3) menentukan nilai K terbaik dengan metode elbow
distortions = []
k_range = range(1,10)
for k in k_range:
    Kmedoid =KMedoids(n_clusters=k)
    Kmedoid.fit(df)
    distortions.append(Kmedoid.inertia_)
    
# ploting elbow method
plt.plot(k_range,distortions,marker= 'o')
plt.title('elbow method for optimal k(k-means)')
plt.xlabel('number of clusters(k)')
plt.ylabel('distortions')
plt.show()

# implementasi k-medoid dan evaluasi silhouette

from sklearn.metrics import silhouette_score
# menentukan nilai k yang optimal
Kmedoid =KMedoids(n_clusters=k)
df['cluster'] =Kmedoid.fit_predict(df)
# menghitung silhouette score 
silhouette_svg= silhouette_score(df,df['cluster'])
print(f'silhouette score for k-means: { silhouette_svg}')

# visualisasi hasil pengelompokan dengan scatter plot

import seaborn as sns 
# visualisasi pengelompokan dengan scatter plot
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='cluster', data=df, palette='viridis', s=100)
plt.scatter(Kmedoid.cluster_centers_[:,0], Kmedoid.cluster_centers_[:,1], s=300, c='red', marker='x', label='Centroids')
plt.title('K-Means Clustering of iris dataset')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend()
plt.show() 