import pandas as pd

df1 = pd .DataFrame({'ID':[1,2,3],'name':['alice','bob','charlie']})

df2 = pd .DataFrame({'ID':[4,5,6],'name':['David','Evan','Frank']})

concatenated_df = pd.concat([df1,df2],axis=0)
print(concatenated_df)