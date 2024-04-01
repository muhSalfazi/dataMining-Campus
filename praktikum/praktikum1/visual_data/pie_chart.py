import matplotlib.pyplot as plt

labels = ['Categories A', 'Categories B', 'Categories C']
sizes = [25,40,35]
plt.pie(sizes,labels=labels,autopct ='%1.1f%%')
plt.title('pie chart examples')

plt.show()          