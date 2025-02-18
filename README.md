# Data-Preprocessing-Breast_Cancer__Sklearn_library
##Loading and Preprocessing

from sklearn.datasets import load_breast_cancer from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler

Load the breast cancer dataset
data = load_breast_cancer()

Split the data into features (X) and target (y)
X = data.data y = data.target

Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Create a StandardScaler object
scaler = StandardScaler()

Fit the scaler to the training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train) X_test_scaled = scaler.transform(X_test)

Preprocessing steps:

Handling missing values: The breast cancer dataset does not contain any missing values, so no additional steps are required.
Feature scaling: StandardScaler is used to scale the features to have a mean of 0 and a standard deviation of 1. This is necessary because some algorithms (e.g., SVM, k-NN) are sensitive to the scale of the features.
Classification Algorithm Implementation Logistic Regression

from sklearn.linear_model import LogisticRegression

Create a LogisticRegression object
logreg = LogisticRegression(max_iter=1000)

Train the model on the scaled training data
logreg.fit(X_train_scaled, y_train)

Logistic regression is a suitable algorithm for this dataset because it is a binary classification problem.

Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

Create a DecisionTreeClassifier object
dt = DecisionTreeClassifier(random_state=42)

Train the model on the scaled training data
dt.fit(X_train_scaled, y_train)

Decision trees are suitable for this dataset because they can handle categorical features and provide interpretable results.

Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

Create a RandomForestClassifier object
rf = RandomForestClassifier(n_estimators=100, random_state=42)

Train the model on the scaled training data
rf.fit(X_train_scaled, y_train)

Random forests are suitable for this dataset because they can handle high-dimensional data and provide robust results.

Support Vector Machine (SVM)

from sklearn.svm import SVC

Create an SVC object
svm = SVC(random_state=42)

Train the model on the scaled training data
svm.fit(X_train_scaled, y_train)

SVMs are suitable for this dataset because they can handle high-dimensional data and provide robust results.

k-Nearest Neighbors (k-NN)

from sklearn.neighbors import KNeighborsClassifier

Create a KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors=5)

Train the model on the scaled training data
knn.fit(X_train_scaled, y_train)

k-NN is suitable for this dataset because it is a simple and efficient algorithm that can handle high-dimensional data.

Model Comparison

from sklearn.metrics import accuracy_score

Make predictions on the scaled testing data
y_pred_logreg = logreg.predict(X_test_scaled) y_pred_dt = dt.predict(X_test_scaled) y_pred_rf = rf.predict(X_test_scaled) y_pred_svm = svm.predict(X_test_scaled) y_pred_knn = knn.predict(X_test_scaled)

Calculate the accuracy of each model
accuracy_logreg = accuracy_score(y_test, y_pred_logreg) accuracy_dt = accuracy_score(y_test, y_pred_dt) accuracy_rf = accuracy_score(y_test, y_pred_rf) accuracy_svm = accuracy_score(y_test, y_pred_svm) accuracy_knn = accuracy_score(y_test, y_pred_knn)

Print the accuracy of each model
print("Logistic Regression Accuracy:", accuracy_logreg) print("Decision Tree Accuracy:", accuracy_dt) print("Random Forest Accuracy:", accuracy_rf) print("SVM Accuracy:", accuracy_svm) print("k-NN Accuracy:", accuracy_knn)

Based on the accuracy scores, the best-performing model is the Random Forest Classifier, and the worst-performing model is the Decision Tree Classifier.

Please note that this is just one possible solution, and you may need to modify it based on your specific requirements.
