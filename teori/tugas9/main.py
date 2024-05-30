import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Pengumpulan Data
def load_data(file_path):
    data = pd.read_excel(file_path)  # Use read_excel for Excel files
    return data

# 2. Pra-pemrosesan Data
def preprocess_data(data):
    # Menghapus kolom yang memiliki terlalu banyak nilai yang hilang
    data = data.dropna(thresh=len(data)*0.6, axis=1)
    # Mengisi nilai yang hilang dengan median
    data = data.fillna(data.median())
    return data

# 3. Analisis Statistik Deskriptif
def descriptive_statistics(data):
    stats = data.describe()
    print("Statistik Deskriptif:\n", stats)
    return stats

# 4. Visualisasi Data
def visualize_data(data):
    # Histogram untuk distribusi
    data.hist(bins=30, figsize=(20, 15))
    plt.show()
    
    # Heatmap untuk korelasi
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

# 5. Analisis Keakuratan
def accuracy_analysis(data, criteria):
    # Contoh kriteria: Memastikan kolom 'age' tidak memiliki nilai negatif
    if 'age' in data.columns:
        if (data['age'] < 0).any():
            print("Terdapat nilai negatif dalam kolom 'age'.")
        else:
            print("Kolom 'age' tidak memiliki nilai negatif.")
    # Tambahkan kriteria lain sesuai kebutuhan penelitian

# 6. Analisis Relevansi
def relevance_analysis(data, target_column):
    # Contoh analisis korelasi dengan target_column
    if target_column in data.columns:
        correlation = data.corr()[target_column]
        print(f"Korelasi dengan {target_column}:\n", correlation)
    else:
        print(f"Kolom {target_column} tidak ditemukan dalam data.")

# Main Function
def main():
    file_path = 'C:\\Users\\salma\\Documents\\coding\\datamining(teori&praktikum)\\teori\\tugas9\\Online_Retail.xlsx'  # Ganti dengan path dataset Anda
    data = load_data(file_path)
    data = preprocess_data(data)
    
    print("Analisis Statistik Deskriptif:")
    descriptive_statistics(data)
    
    print("Visualisasi Data:")
    visualize_data(data)
    
    print("Analisis Keakuratan Data:")
    accuracy_analysis(data, criteria={})
    
    print("Analisis Relevansi Data:")
    relevance_analysis(data, target_column='target')  # Ganti 'target' dengan kolom target penelitian Anda

if __name__ == "__main__":
    main()
