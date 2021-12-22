import pandas as pd
from sklearn import linear_model

df = pd.read_csv("salesforcusting.csv")
#print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['YearID', 'MonthID', 'MNo']], df.Total)
predict = reg.predict([[2021, 12, 293]])
print(predict)
