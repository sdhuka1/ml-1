''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

#how many samples and How many features?
print(diabetes.data.shape)

# What does feature s6 represent?
print(diabetes.DESCR)
print(f"Feature s6 represents glu, blood sugar level")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=11)

mymodel = LinearRegression()
mymodel.fit(x_train, y_train)

#print out the coefficient
print(mymodel.coef_)

#print out the intercept
print(mymodel.intercept_)


# create a scatterplot with regression line
predicted = mymodel.predict(x_test)

expected = y_test

plt.plot(expected, predicted, ".")

x = np.linspace(0,330,100)
y = x
plt.plot(x,y)
plt.xlabel("Expected")
plt.ylabel("Predicted")
plt.show()