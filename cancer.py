import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split


#imblearn is imbalance learn. Like scikit learn it's also something. 
#Unfamiliar with the concept of oversampling

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('Cancer_Data.csv')

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
#inplace modifies the values in the dataframe itself and cant be accessed once removed. 

print(df.isnull().sum())


#Outliers are data point that differ significantly from other observations in dataset. 
#Removing them increasing accuracy of model!

threshhold = 5
for column in df.loc[:, ~df.columns.isin(['diagnosis'])]:
    mean = df[column].mean()
    std = df[column].std()

    lower_limit = mean - threshhold * std;
    upper_limit = mean + threshhold * std;

df = df[(df[column] >= lower_limit)  & (df[column] <= upper_limit)]

print(df.shape)

#Diagnosis is the outcome we want to predict. To train we replace O and M with 0 and 1

df['diagnosis'] = df['diagnosis'].map({'B' :0, 'M' :1})

X = df.drop('diagnosis', axis =1)
y = df['diagnosis']

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size= 0.25, random_state=42)


rf = RandomForestClassifier(max_depth = 5, random_state= 0 )
rf.fit(xtrain, ytrain)

pred = rf.predict(xtest)

accuracy = accuracy_score(ytest, pred)
precision = precision_score(ytest, pred)
recall = recall_score(ytest, pred)
f1 = f1_score(ytest, pred)

print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)

#Data balancing that's new. 
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state = 0)
xtrain_res, ytrain_res = smt.fit_resample(xtrain, ytrain)

#I will explore into the significance of this later. 
#Don't know why but  VS Code says module not found but otherwise I think this part should work! 
#Get better results with this. 



