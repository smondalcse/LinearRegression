import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("nmlsales.csv")
mean_totsales = df.totsales.mean()
df.totsales = df.totsales.fillna(mean_totsales)
print(df)

dummy = pd.get_dummies(df['Month'])
dummy1 = pd.get_dummies(df['Model'])
#   print(d)
marged = pd.concat([df, dummy], axis='columns')
marged1 = pd.concat([marged, dummy1], axis='columns')
#   print(marged)

finaldataset = marged1.drop(['Month'], axis='columns')
finaldataset1 = finaldataset.drop(['Model'], axis='columns')
#print(finaldataset1)


#   using linear regression algo we have to train the model
reg = linear_model.LinearRegression()
reg.fit(finaldataset1[['Year', 'lpt1', 'lpt2', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']], df.totsales)
predict = reg.predict([[2020, 0, 1, 0,0,0,0,0,0,1,0,0,0,0,0]])
print(f"predicted sales: {predict}")
score = reg.score(finaldataset1[['Year', 'lpt1', 'lpt2', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']], df.totsales)
print(f"score: {score}")

