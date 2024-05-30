from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Load the Iris dataset
iris = datasets.load_iris()
x = iris.data[:, :2]  # Only use the first two features for visualization
y = iris.target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Build the SVM model
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(x_train, y_train)

# Make predictions
y_pred_svm = svm_classifier.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_svm)
print(f'Accuracy: {accuracy:.2f}')

# Visualize the classification results
plt.figure(figsize=(10, 6))
plot_decision_regions(x_test, y_test, clf=svm_classifier, legend=2)
plt.title('SVM Classification on Iris Data')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

# Print the classification report
classification_rep = classification_report(y_test, y_pred_svm)
print("Classification Report:\n", classification_rep)
