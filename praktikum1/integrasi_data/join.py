import pandas as pd
df1 = pd.DataFrame({'ID':[1,2,3],'Name':['alice','Bob','charlies']})
df2 = pd.DataFrame({'Age':[25,30,22]},index=[1,2,4])

joined_df =df1.join(df2,on='ID',how='left')
print(joined_df)