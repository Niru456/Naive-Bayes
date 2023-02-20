# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 18:30:50 2022

@author: 20050
"""


# Import the data
import pandas as pd
df_test = pd.read_csv("SalaryData_Test (2).csv")
df_test

df_train = pd.read_csv("SalaryData_Train (1).csv")
df_train

	
#=============================================================================
#                      EDA (Exploratory Data Analysis)
#=============================================================================
# Pie Chart
import matplotlib.pyplot as plt
df_test['Salary'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()

# Histogram
plt.hist(df_test['Salary'], bins=5)
plt.show()

# let's make scatter plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data = df_test, hue = 'Salary')

# Data Transformation
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df_test['Salary'] = LE.fit_transform(df_test['Salary'])
df_train['Salary'] = LE.fit_transform(df_train['Salary'])

df_new_test = pd.get_dummies(df_test)
df_new_train = pd.get_dummies(df_train)

# Drop the variable and split the variable from the data
x_train = df_new_train.drop('Salary',axis=1)
y_train = df_new_train['Salary']
x_test = df_new_test.drop('Salary',axis=1)
y_test = df_new_test['Salary']

y_train  

# Model development using naive bayes
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(x_train, y_train)

y_pred_train = MNB.predict(x_train)
y_pred_test = MNB.predict(x_test)


# confusion matrix and accuracy
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_train, y_pred_train).round(2)
print("naive bayes model training score", acc1)
acc2 = accuracy_score(y_test, y_pred_test).round(2)
print("naive bayes model test score", acc2)

# prediction of salary = 77%





















































 