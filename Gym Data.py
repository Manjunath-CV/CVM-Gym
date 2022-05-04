# -*- coding: utf-8 -*-
"""
Created on Mon May  2 21:42:40 2022

@author: manju
"""
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
gym=pd.read_excel(r'D:\ManjuBepec\Bepec november 2021\Phase II\dataGYM.xlsx')
gym['Class'] = LabelEncoder().fit_transform(gym['Class'])
x=gym.iloc[:,:3]
y=gym.iloc[:,5:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)
model_gym=RandomForestClassifier(n_estimators=20)
model_gym.fit(x_train,y_train)
expected=y_test
predicted=model_gym.predict(x_test)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
import pickle
pickle.dump(model_gym,open("model_gym.pkl","wb"))
model=pickle.load(open("model_gym.pkl","rb"))
print(model.predict([[40,5.6,70]]))
