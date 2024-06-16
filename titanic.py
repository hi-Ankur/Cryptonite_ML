import numpy as np
import pandas as pd 

from sklearn.metrics import accuracy_score

train_data =pd.read_csv('train.csv')
print(train_data.head())

test_data = pd.read_csv('test.csv')
print(test_data.head())

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)*100
print("% of women who survivied:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)*100
print("% of men whwo survived:", rate_men)


#Random Forest Model???
#sklearn.sensemble???? You can do classification regression ( I think it's also called Logistic Regression)

from sklearn.ensemble import RandomForestClassifier
y = train_data["Survived"]


features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

#y_test = pd.read_csv('gender_submission.csv')
#y_test = y_test.drop('PassangerID', axis=1, inplace=True)
# I am having trouble creating a y_test to check the accuracy because the true result are given in a different csv file. 




#pd.get_dummies???? Convert categorical variable into dummy/indicator variables. Is this the same as label encoding??? 
#Don't know for sure but get_dummies seems pretty important for LR

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

#100 trees and random state 1. I guess random_state LR can only be 0 or 1??????

model.fit(X,y)
predictions = model.predict(X_test)
#accuracy = accuracy_score(predictions, y_test)


output = pd.DataFrame({'PassangerID:': test_data.PassengerId, 'Survived': predictions})
#print('\nAccuracy = ', accuracy)
output.to_csv('submission.csv', index=False)

#There's some error here. 

#I guess I am just saving the prediciton in a csv file!! Maybe I can compare the two through pandas but can't really maneuver that right now??





