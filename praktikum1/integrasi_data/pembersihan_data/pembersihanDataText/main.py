import pandas as pd
import re
import seaborn as sns  
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #pip install scikit-learn



df = pd.read_excel("c:/Users/salma/Documents/coding/datamining-praktikum/praktikum1/integrasi_data/pembersihan_data/pembersihanDataText/data_text.xlsx")
# untuk membaca file excel :pip install openpyxl
print("menampilkan data set")
print(df.tail())
print("---------------")
# mengindentifikasi text yang kosong atau missing value
print(df.isnull().sum())

print("----------------------------------------------------------------")
# menangani missing value
df.dropna(how='any', inplace=True)
print("menampilkan hasil penanganan missing value")
print(df.isnull().sum())

print("----------------------------------------------------")
print("pembersihan text (penghapusan karakter khusus)")
# Melakukan pembersihan teks (penghapusan karakter khusus)
df['Text'] = df['Text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))
# Menampilkan dataframe setelah pembersihan teks
print(df.tail())

print("----------------------------------------------------------------")
print("4.Transformasi dan Reduksi Data ")
print("1)Transpormasi data dan dengan standard scaler")
# import data
data = sns.load_dataset('iris')
# memilih fitur yang akan diubah
features = ['sepal_length', 'sepal_width']
# menggunakan StandardScaler
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])
print(data.head())

print("--------------------------------")
print("2)Transformasi logaritma ")
import numpy as np
# membaca data 
data = sns.load_dataset('iris')
# memilih fitur yang akan di ubah dengan transformasi logaritma
features = ['petal_length']
# menggunakan transformasi logaritma
data[features] = np.log1p(data[features])
print(data.head())

print("--------------------------------")
print("3)mengemlompokan nilai numerik menjadi interval atau kategori diskrit menggunakan teknik binning dan diskritisasi")
data = sns.load_dataset('iris')  #membaca data

# memilih fitur yang akan di bining
features = 'sepal_length'
# menggunakan binning pada fitur tertentu
data[features] = pd.cut(data[features], bins=3,labels=['Low','Medium','High',])
print(data.head())

print("--------------------------------")
print("4)Transformasi data dengan PCA")

data = sns.load_dataset('iris')
# Standarisasi data
features1 = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
x = data[features1]  # Perbaiki penulisan nama kolom
x_standardized = StandardScaler().fit_transform(x)

# Mengaplikasikan PCA
pca = PCA(n_components=2)
principal_component = pca.fit_transform(x_standardized)
principal_df = pd.DataFrame(principal_component, columns=['PC1', 'PC2'])
print(principal_df.head())

print(" Transformasi data dengan t-distributed Stochastic Neighbor Embedding(t-SNE)")
# 5)Transformasi data dengan t-distributed Stochastic Neighbor Embedding(t-SNE)
# membaca data
data = sns.load_dataset('iris')
# standarisasi data
features = ['sepal_length','sepal_width','petal_length','petal_width']
x= data[features]
x_standardized = StandardScaler().fit_transform(x)
# mengaplikasikan t-SNE
tsne = TSNE (n_components=2,perplexity=30)
tsne_results = tsne.fit_transform(x_standardized)
tsne_df = pd.DataFrame(tsne_results,columns=['t-SNE1', 't-SNE2'])
print(tsne_df.head())

print("6)Transformasi data dengan LDA")
# 6)Transformasi data dengan LDA
# membaca data
data =sns.load_dataset('iris')
# standarisasi data
features = ['sepal_length','sepal_width','petal_length','petal_width']
x = data[features]
x_standardized = StandardScaler().fit_transform(x)
# implementasi LDA
lda = LinearDiscriminantAnalysis(n_components=2)
lda_results = lda.fit_transform(x_standardized,data['species'])
lda_df=pd.DataFrame(lda_results,columns=['LDA1','LDA2'])
print(lda_df)