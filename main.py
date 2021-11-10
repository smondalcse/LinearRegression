import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#   Read CSV File
df = pd.read_csv('dhaka homeprices.csv')
#print(df)
#print(df.head(6))
#print(df.shape)
# print(df.isnull().any())
# print(df.isnull().sum())


# Plot data to graph
plt.xlabel('Area in square ft')
plt.ylabel('Price in taka')
plt.scatter(df['area'], df['price'], color='red', marker='+')
plt.title('Homeprices in Dhaka city')
#plt.show()


#   Separate dependent and independent variable
x = df[['area']]
y = df['price']
#print(x)
#print(y)


#   Split Data set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state =1)
#print(xtest)
#print(xtrain)


#   Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
result_prediction = regressor.predict(xtest)
print(result_prediction)
# print(regressor.predict([[3500]]))
c = regressor.intercept_
print(f"intercept:  {c}")
m = regressor.coef_
print(f"Coefficient: {m}")
# y = mx + c    ==> Simple Linear regression equation


#   Best fit line
plt.xlabel('Area in square ft')
plt.ylabel('Price in taka')
plt.scatter(df['area'], df['price'], color='red', marker='+')
plt.title('Homeprices in Dhaka city')
plt.plot(df.area, regressor.predict(df[['area']]))
# plt.show()

# Predicted score will be (0 to 1).
# 1 means 100% correct
print(f"Prediction Score:  {regressor.score(xtest,ytest)}")



