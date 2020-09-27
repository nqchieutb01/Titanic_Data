import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# machine learning
from sklearn import ensemble, linear_model, naive_bayes, neighbors, svm, tree, model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix

train_df = pd.read_csv('C:/Users/Admin/Desktop/Titanic Task/train.csv')
test_df = pd.read_csv('C:/Users/Admin/Desktop/Titanic Task/test.csv')
combine = [train_df,test_df]

# Điền hết dữ liệu trống
for dataset in combine:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

# Chia giá vé thành 4 phần
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
# chia độ tuổi thành 5 phần
train_df['AgeBand'] = pd.cut(train_df['Age'].astype(int), 5)
# map đến tên
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:

    # Familysize là số lượng người thân đi cùng
    dataset['FamilySize'] = dataset['SibSp'] + train_df['Parch'] + 1

    # xác định xem người đó có đi 1 mình hay ko ?
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    # tên gọi, chức danh
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # có những chức danh xuất hiện với tần số rất ít nên ta gán là Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    # format lại 1 số tên
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # Chuyển các tên thành dạng số 1->5
    dataset['Title'] = dataset['Title'].map(title_mapping)

    # format giới tính về dạng số
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # Chia 4 khoảng tuổi vào các đoạn 0->3
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

    # Chia các loại giá vé từ 0->3
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # format lại Embarked S : 0 , C:1 , Q:2
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['SibSp', 'Parch','Ticket','Cabin','PassengerId','Age','FareBand','AgeBand','Name'], axis=1)
test_df = test_df.drop(['SibSp', 'Parch','Ticket','Cabin','PassengerId','Age','Name'], axis=1)

X_train = train_df.drop(['Survived'],axis=1)
Y_train = train_df.Survived
X_test = test_df

random_forest = RandomForestClassifier(n_estimators=100,criterion = 'entropy',max_depth=10)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_train)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

#Đánh giá
print(acc_random_forest)
target_names = ['0','1']
print(classification_report(Y_train, Y_pred, target_names=target_names))

# confusion maxtrix
disp = plot_confusion_matrix(random_forest, X_train, Y_train,display_labels=target_names, cmap=plt.cm.Blues)
disp.ax_.set_title("confusion matrix")
plt.show()

#--------------------------------------------------------------------------------------------------------------------
"""
Accuracy = []
#logistics regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
Accuracy.append(acc_log)

#SVM
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
Accuracy.append(acc_svc)

#decision tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
Accuracy.append(acc_decision_tree)

# random_forest
random_forest = RandomForestClassifier(n_estimators=1000)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
Accuracy.append(acc_random_forest)

#k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
Accuracy.append(acc_knn)

print(Accuracy)
"""