import pandas as pd

df1 = pd.DataFrame({'ID':[1,2,3],'Name':['alice','Bob','charlies']})
df2 = pd.DataFrame({'ID':[1,2,4],'age':[25,20,22]})

merge_df = pd.merge(df1,df2, how='inner')
print(merge_df)
