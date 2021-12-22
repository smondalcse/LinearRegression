import pandas as pd
from sklearn import linear_model

df = pd.read_csv("salesforcusting1.csv")
print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['Year']], df.Sale)
predict = reg.predict([[2021]])
print(predict)
