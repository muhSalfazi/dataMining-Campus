import dask.dataframe as dd
import pandas as pd

# inisialisasi data
df1 = dd.from_pandas(pd.DataFrame({'ID': [1, 2, 3], 'Name': ['alice', 'bob', 'charlie']}), npartitions=2)
df2 = dd.from_pandas(pd.DataFrame({'ID': [1, 2, 3], 'Age': [25, 50, 22]}), npartitions=2)

merged_df = dd.merge(df1, df2, on='ID', how='inner')

print(merged_df)
