from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Contoh data transaksional
data = {'Transaction': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'Items': [['A', 'B', 'D'], ['B', 'C', 'E'], ['A', 'B', 'D', 'E'], ['B', 'E'], ['A', 'C', 'D']]}
df = pd.DataFrame(data)

# Konversi data transaksional menjadi format yang sesuai
transactions = df['Items'].values.tolist()

transaction_encoder = TransactionEncoder()
transaction_array = transaction_encoder.fit(transactions).transform(transactions)
transaction_dataframe = pd.DataFrame(transaction_array, columns=transaction_encoder.columns_)

# Aplikasi Algoritma Apriori
frequent_itemsets_apriori = apriori(transaction_dataframe, min_support=0.2, use_colnames=True)

# Penerapan aturan asosiasi untuk Apriori
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.7)

# Aplikasi Algoritma FP-Growth
frequent_itemsets_fp_growth = fpgrowth(transaction_dataframe, min_support=0.2, use_colnames=True)

# Penerapan aturan asosiasi untuk FP-Growth
rules_fp_growth = association_rules(frequent_itemsets_fp_growth, metric="confidence", min_threshold=0.7)

# Tampilkan hasil
print("Frequent Itemsets Apriori:")
print(frequent_itemsets_apriori)
print("\nAssociation Rules Apriori:") 
print(rules_apriori)

print("\nFrequent Itemsets FP-Growth:")
print(frequent_itemsets_fp_growth)
print("\nAssociation Rules FP-Growth:")
print(rules_fp_growth)