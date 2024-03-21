from sqlalchemy import create_engine
import pandas as pd

# Membuat koneksi database
engine = create_engine('sqlite:///:memory:')

# Membuat DataFrames
df1 = pd.DataFrame({'ID':[1,2,3],'Name':['salman','udin','sarmin']})
df2 = pd.DataFrame({'ID':[1,2,4],'Age':[25,30,22]})

# Menyimpan DataFrames ke dalam database
df1.to_sql('table1', engine) 
df2.to_sql('table2', engine) 

# Melakukan JOIN antara kedua tabel
query = "SELECT * FROM table1 INNER JOIN table2 ON table1.ID = table2.ID"
result_df = pd.read_sql(query, engine)

print(result_df)
