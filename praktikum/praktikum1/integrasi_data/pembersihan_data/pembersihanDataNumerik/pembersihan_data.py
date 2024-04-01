import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 1)load dataset
data = sns.load_dataset('tips')
#2) menampilkan data set
print(data.head())
# 3)Mengidentifikasi nilai hilang
print("=> Mengidentifikasi nilai hilang <==")
missing_values = data.isnull().sum()
print("Jumlah data hilang:")
print(missing_values)

# Mengambil hanya kolom numerik
numeric_data = data.select_dtypes(include=['number'])

# Menangani data yang hilang dengan nilai mean untuk kolom-kolom numerik
numeric_data.fillna(numeric_data.mean(), inplace=True)

print("\nilai data yang hilang :",numeric_data)
# Menggabungkan kembali kolom numerik dengan kolom kategorikal
data_cleaned = pd.concat([numeric_data, data.select_dtypes(exclude=['number'])], axis=1)

print("==> Menangani duplikasi data <==")
data_cleaned.drop_duplicates(inplace=True)
print(data_cleaned)

# Mengidentifikasi outlier dengan boxplot
sns.boxplot(data=data_cleaned, y='total_bill')
plt.title("Mengidentifikasi outlier dengan boxplot")
plt.show()

print("==> Menangani outlier dengan menghapusnya menggunakan z-score <==")

# Menghitung z-score
data_cleaned['total_bill_z'] = stats.zscore(data_cleaned['total_bill'])

print(data_cleaned.head())

# Hapus outlier dengan z-score kurang dari 2
print("==============================================")
data_no = data_cleaned[(data_cleaned.total_bill_z < 2)]
print(data_no.head())

print("==> Menampilkan hasil data setelah menghapus outlier <==")
sns.boxplot(data=data_no, y="total_bill")
plt.title("Menampilkan hasil data setelah menghapus outlier)")
plt.show()

# mengidentifikasi type data 
print("==> Mengidentifikasi type data <==")
print(data.dtypes)

# menangani anomali
print("==>penangananan anomali(misal,dengan menghapus nilai yang tidak sesuai <==")
df = data [data['total_bill']>10.00]
df['total_bill'].min()
print(df.head())

# 14)normalisasi dan standarisi

# install  pip install scikit-learn
print("==> normalisasi dan standarisi <==")
from sklearn.preprocessing import MinMaxScaler
data = sns.load_dataset('iris')
# normalisasi(misal dengan MinMaxScaler)
scaler = MinMaxScaler()
print(data.head())

# 15)menampilkan hasil normalisasi
data.drop('species',axis=1,inplace=True)
data_sc = scaler.fit_transform(data)
data = pd.DataFrame(data_sc,columns=data.columns)
print("==> menampilkan hasil normalisasi <==")
print(data.head())

#16)validasi konsistensi
assert data['sepal_length'].between(0,100).all(),"kesalahan:nilai di luar rentang yang di inginkan"