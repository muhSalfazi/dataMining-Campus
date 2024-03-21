import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


data = sns.load_dataset('tips')

print(data.head())

print("=> Mengidentifikasi nilai hilang <==")
missing_values = data.isnull().sum()
print("Jumlah data hilang:")
print(missing_values)

# Mengambil hanya kolom numerik
numeric_data = data.select_dtypes(include=['number'])

# Menangani data yang hilang dengan nilai mean untuk kolom-kolom numerik
numeric_data.fillna(numeric_data.mean(), inplace=True)

# Menggabungkan kembali kolom numerik dengan kolom kategorikal
data_cleaned = pd.concat([numeric_data, data.select_dtypes(exclude=['number'])], axis=1)

print("==> Menangani duplikasi data <==")
data_cleaned.drop_duplicates(inplace=True)

# Mengidentifikasi outlier dengan boxplot
sns.boxplot(data=data_cleaned, y='total_bill')
plt.title("Box Plot Example")
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
plt.title("Box Plot Example (Outliers Removed)")
plt.show()

# mengidentifikasi type data 
print("==> Mengidentifikasi type data <==")
print(data.dtypes)

# menangani anomali
print("==>penangananan anomali(misal,dengan menghapus nilai yang tidak sesuai <==")
df = data [data['total_bill']>10.00]
df['total_bill'].min()
print(df.head())

# normalisasi dan standarisi

# install  pip install scikit-learn
print("==> normalisasi dan standarisi <==")
from sklearn.preprocessing import MinMaxScaler
data = sns.load_dataset('iris')
# normalisasi(misal dengan MinMaxScaler)
scaler = MinMaxScaler()
print(data.head())


# validasi konsistensi
