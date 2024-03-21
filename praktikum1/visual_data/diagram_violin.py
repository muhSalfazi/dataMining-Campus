import matplotlib.pyplot as plt
import seaborn as sns

data = sns.load_dataset('iris')
sns.violinplot(data,x='species',y='sepal_length')
plt.title('violin plot example')
plt.show()