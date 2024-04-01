import matplotlib.pyplot as plt
import seaborn as sns

data = sns.load_dataset('flights').pivot_table(index='month', columns='year', values='passengers')

plt.figure(figsize=(12, 5))

sns.heatmap(data, annot=True, cmap='coolwarm')
# plt.title('Heatmap Example')

plt.show()
