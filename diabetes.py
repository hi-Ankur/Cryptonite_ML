import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


#I don't know what this does???

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict

#from imblearn.over_sampling import RandomOverSampler
#I don't know what this does either so not going to use it but it's supposed to increase accuracy.

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC            #Support Vector Machines : supervised learning method  with advantages

from sklearn.ensemble import RandomForestClassifier
#We used this in Titanic. High accuracy because of multiple decision trees!!

#The notebook also shows how to use GradientBoostingClassifier and XGBClassifier but we wont use that!!
#Compute confusion matrix to evaluate the accuracy of a classification. User Guide

from sklearn.metrics import accuracy_score, recall_score, f1_score

data = pd.read_csv('diabetes.csv')

print(data.duplicated().sum()) #That's a new one 

#Ofcourse the correlation graph
print(data.corr())
sns.heatmap(data.corr(), annot=True, fmt='0.1f', linewidths=5)
plt.show()

print(data.isnull().sum())

x = data.drop('Outcome', axis=1)
y = data['Outcome']

#rm = RandomOverSampler(random_state=41)
#For this to work we need to import imblearn but that module isn't available so we wont use it now.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

model_LR = LogisticRegression()
model_RFC = RandomForestClassifier(n_estimators=100, class_weight='balanced')

#Never made a function in python but this seems pretty simple

def cal(model):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    accuracy = accuracy_score(pred, y_test)
    recall = recall_score(pred, y_test)
    f1 = f1_score(pred, y_test)

    print(model)
    print('Accuracy: ', accuracy)
    print('Recall: ', recall)
    print('F1 Score: ', f1)
    print()

cal(model_LR)
cal(model_RFC)









