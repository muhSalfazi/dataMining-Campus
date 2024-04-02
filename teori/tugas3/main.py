import pandas as pd

# Membaca file CSV
df = pd.read_csv("C:/Users/salma/Documents/coding/datamining(teori&praktikum)/teori/tugas3/price.csv")

# Menampilkan informasi dasar tentang data
print("Informasi dasar tentang data:")
print(df.info())

# Menampilkan 5 baris pertama data
print("\n5 baris pertama data:")
print(df.head())

# Menampilkan statistik deskriptif untuk variabel numerik
print("\nStatistik deskriptif:")
print(df.describe())

# Menampilkan jumlah missing values untuk setiap kolom
print("\nJumlah missing values untuk setiap kolom:")
print(df.isnull().sum())

# Melakukan visualisasi data, misalnya dengan menggunakan matplotlib atau seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Misalnya, visualisasi histogram untuk variabel House_Price
plt.figure(figsize=(10, 6))
sns.histplot(df['House_Price'], bins=30, kde=True)
plt.title('Distribusi Harga Rumah')
plt.xlabel('Harga Rumah')
plt.ylabel('Frekuensi')
plt.show()

# Misalnya, visualisasi hubungan antara House_Price dan Dist_Taxi
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Dist_Taxi', y='House_Price')
plt.title('Hubungan Antara Harga Rumah dan Jarak ke Taxi')
plt.xlabel('Jarak ke Taxi')
plt.ylabel('Harga Rumah')
plt.show()

