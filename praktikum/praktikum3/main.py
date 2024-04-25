from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#3)load dataset
# membuat data iris
iris = load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)

# 4) implementasi DBSCAN
# menggunakan dbscan 
dbscan = DBSCAN(eps=0.5,min_samples=5)
df['cluster']=dbscan.fit_predict(df)

# 5)menampilkan jumlah kluster
# menampilkan jumkah kluster yang dihasilkan(klaster -1 menanandakan noise)
print("jumlah kluster",df['cluster'].nunique())
# visualisasi pengelompokan mengggunakan scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x='sepal length (cm)',y='sepal width (cm)',hue='cluster',data=df,palette='viridis',s=100)
plt.title("DBSCAN Clustering iris dataset")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.legend()
plt.show()

#6)Evaluasi dengan silhouette score,davies bouldin score
from sklearn.metrics import silhouette_score,davies_bouldin_score
# menghitung silhouette score
silhoutte_avg = silhouette_score(df.iloc[:,:-1],df['cluster'])
print(f'silhoutte score for dbscan: {silhoutte_avg}')
# menghitung devies-bouldin index(semakin rendah,semakin baik)
db_index = davies_bouldin_score(df.iloc[:,:-1],df['cluster'])
print(f'davies-bouldin index for dbscan: {db_index}')

#7)visualisasi pengelompokan menggunakan scatter plot
# visualisasi pengelompokan menggunakan scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x='sepal length (cm)',y='sepal width (cm)',hue='cluster',data=df,palette='viridis',s=100)
plt.title('DBScan Clustering of iris dataset')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend()
plt.show()