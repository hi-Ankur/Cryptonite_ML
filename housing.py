import statsmodels.api as sm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
print('Hellow World')
df = pd.read_csv('housing.csv')
print(df.head())
print('\n ')

print(df.info())
print(df.isnull().sum())
print(df.isnull().sum()/len(df)*100)
df = df.dropna()
print(df.isnull().sum())
print(df.describe())

df2 = pd.get_dummies(df, columns = ['ocean_proximity'])
print( )
df3 = df2.drop('ocean_proximity_ISLAND', axis = 1)
print(df3.head())

#Model Training & Testing

features = df3.drop(columns= ['median_house_value'])
target = df3['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state = 69)

print("Length of Train", len(x_train))
print("Length of Test", len(x_test))

x_train_sm = sm.add_constant(x_train)
y_train = y_train.astype(float)
x_train_sm = x_train_sm.astype(float)

model = sm.OLS(y_train, x_train_sm).fit()
print(model.summary())

x_test_sm = sm.add_constant(x_test)
y_pred = model.predict(x_test_sm)
print(y_pred)

plt.scatter(y_test, y_pred, color = 'grey')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual Values vs Predicted Values')
plt.plot(y_test, y_test, color = 'black')
plt.show()


#You can't do both!


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
pd.DataFrame(x_train_scaled)

lr = LinearRegression()
lr.fit(x_train_scaled, y_train)
y_pred = lr.predict(x_test_scaled)
#finding mse, rmse and r square as well as adjusted r squared values
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-x_test_scaled.shape[1]-1)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")
print(f"Adjusted R-squared: {adj_r2}")
