import  matplotlib.pyplot as plt
categories = ['A', 'B', 'C', 'D']
values = [5,12,8,15]
plt.bar(categories, values)
plt.xlabel('categories')
plt.ylabel('values')
plt.title('Bar chart examples')

plt.show()