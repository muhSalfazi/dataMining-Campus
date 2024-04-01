import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Membaca dataset properti
df = pd.read_csv('price.csv')

# Menampilkan informasi data
print(df.head())
print(df.info())
print(df.describe())

# Menghapus missing values dan outliers
df_clean = df.dropna()
df_clean = df_clean[(np.abs(df_clean['House_Price'] - df_clean['House_Price'].mean()) <= (3 * df_clean['House_Price'].std()))]

# Memilih variabel yang relevan
df_selected = df_clean[['Dist_Taxi', 'Dist_Market', 'Dist_Hospital', 'Carpet', 'Builtup', 'Rainfall', 'House_Price']]

# Memisahkan variabel independen dan dependen
X = df_selected.drop('House_Price', axis=1)
y = df_selected['House_Price']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Memprediksi harga rumah menggunakan data uji
y_pred = model.predict(X_test)

# Evaluasi kinerja model
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R-squared:", r_squared)
print("Mean squared error:", mse)

# Output
print("Model regresi linear:", model)

# Menampilkan boxplot untuk setiap variabel
plt.figure(figsize=(12, 8))
for i, col in enumerate(df_selected.columns[:-1]):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=df_selected[col])
    plt.title(f'Boxplot {col}')
plt.tight_layout()
plt.show()

