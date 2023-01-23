# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:20:53 2022

@author: RAFA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sn = "Sheet1"
dataset = pd.read_excel('D:/ABC Company.xlsx',sheet_name = sn)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train) 

#Visualizing the Training set Results
viz_train = plt
viz_train.scatter(x_train,y_train,color='red')
viz_train.plot(x_train, regressor.predict(x_train),color='blue')
viz_train.title('Salary VS Experience(Training set)')
viz_train.xlabel('Year of Experience')
viz_train.ylabel('Salary')
viz_train.show()

#Visualizing the Training set Results
viz_test = plt
viz_test.scatter(x_test,y_test,color='red')
viz_test.plot(x_train, regressor.predict(x_train),color='blue')
viz_test.title('Salary VS Experience(Test set)')
viz_test.xlabel('Year of Experience')
viz_test.ylabel('Salary')
viz_test.show()

y_e = float(input('Years Experience: '))
y_pred = regressor.predict([[y_e]])
print("With your",y_e,"Years Experience of work, your salary is",y_pred)
