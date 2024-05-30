import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Memuat data
data = pd.read_csv('/mnt/data/bank.csv')

# Menampilkan beberapa baris data untuk pemeriksaan
print(data.head())

# Preprocessing
# Mengonversi variabel kategorikal menjadi variabel dummy/one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Memisahkan fitur dan target
X = data.drop('y_yes', axis=1)
y = data['y_yes']

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membuat dan melatih model decision tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Memprediksi pada set pengujian
y_pred = clf.predict(X_test)

# Evaluasi model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
