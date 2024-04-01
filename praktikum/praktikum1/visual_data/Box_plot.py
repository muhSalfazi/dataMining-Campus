import matplotlib.pyplot as plt
import seaborn as sns

data = sns.load_dataset('tips')
sns.boxplot(data=data,x='days',y='total_bill')
plt.title("Box Plot Example")
plt.show()