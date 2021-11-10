import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("car data.csv")
#   print(df)
#   print(df.experience)


#   find the mean value of experience column
exp_mean = df.experience.mean()

#   find the median value of experience column
exp_median = df.experience.median()

#   fill all the num value by median
df.experience = df.experience.fillna(exp_median)
print(df)

#   using linear regression algo we have to train the model
reg = linear_model.LinearRegression()
reg.fit(df[['speed', 'car_age', 'experience']], df.risk)
predict = reg.predict([[160, 10, 5]])
print(f"predict result: {predict}")

