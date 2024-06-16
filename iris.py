import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


iris = pd.read_csv('Iris.csv')
print(iris.head())
iris.info()

iris.drop('Id', axis = 1, inplace=True)

#A Classification problem

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#I don't know what this does.
from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

print(iris.shape)
#Checking corelation is important as it affects the model accuracy

iris2 = iris.drop(['Species'], axis=1)
#I dropped the species column because it was a string but it's a categorical variable so maybe could have applied label encoding.
sns.heatmap(iris2.corr(), annot=True, cmap='cubehelix_r')
plt.show()

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
y = iris['Species']
#x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=34)
#Something wrong with this, don't know why it's not working maybe I am making an error with some parameters.

train, test = train_test_split(iris, test_size=0.3)

x_train = train[features]
y_train = train.Species     #I believe I can write train.y

x_test = test[features]
y_test = test.Species 

model = LogisticRegression()
model.fit(x_train, y_train)
prediction=model.predict(x_test)
print('The accuracy of the Logistic Regression is', accuracy_score(prediction,y_test))

model=DecisionTreeClassifier()
model.fit(x_train, y_train)
prediction=model.predict(x_test)
print('The accuracy of the Decision Tree is', accuracy_score(prediction,y_test))








