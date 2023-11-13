#..................
import cv2
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Define the training dataset with numerical labels (0 for 'cat' and 1 for 'dog')
X_train = np.array([[2, 3], [3, 5], [4, 2], [6, 8], [7, 5], [8, 7], [9, 6], [10, 8]])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Create the k-NN classifier with k=3
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Fit the classifier to the training data
knn_classifier.fit(X_train, y_train)

# Define a new data point to classify
new_data_point = np.array([[5, 4]])

# Predict the class of the new data point
predicted_class = knn_classifier.predict(new_data_point)

# Plot the training data points
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Class cat', color='blue')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Class dog', color='red')

# Plot the new data point
plt.scatter(new_data_point[:, 0], new_data_point[:, 1], label=f'New Data (Class {predicted_class[0]})', color='green', marker='x', s=100)

# Create a mesh grid to plot the decision boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.4, cmap='cool')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('k-NN Classification')
plt.legend()
plt.show()
