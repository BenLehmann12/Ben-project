from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

flowers = load_iris()
X = flowers.data   
Y = flowers.target

#Training and Testing accuracy with K-neighbors using Model
X_train,X_test,y_train,y_test = train_test_split(X,Y)
k = range(1,28)  #1 to 28 k-neighbors  
test_accuracy = []
train_accuracy = []
for neighbors in k:
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(X_train,y_train)
    train_accuracy.append(knn.score(X_train,y_train))
    test_accuracy.append(knn.score(X_test,y_test))
plt.plot(k, train_accuracy, label="train accuracy")
plt.plot(k, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.legend()
plt.show()

#Find the accuracy of the prediction model
model = LogisticRegression()
model.fit(X_train,y_train)
predict = model.predict(X_test)
print(predict)
print(accuracy_score(y_test, predict))

#Create a confusion matrix
cm = confusion_matrix(y_test, predict)
print(cm)

#The Accuracy of Both Training and Testing Sets in KNN
standard = StandardScaler()  #Scaling the data set for more accuracy
standard.fit(X_train)
X_train_std = standard.transform(X_train)
X_test_std = standard.transform(X_test)

new_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')  #Using the minkowski distance
new_knn.fit(X_train_std,y_train)
print("The Accuracy of the KNN on training set is {:.2f}".format(new_knn.score(X_train_std,y_train)))
print("The Accuracy of the KNN on testing set is {:.2f}".format(new_knn.score(X_test_std,y_test)))
