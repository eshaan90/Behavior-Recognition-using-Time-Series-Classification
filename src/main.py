# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:24:34 2019

@author: rajar
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

file=open('detection.txt','r')
det = file.readlines()
det = [float(x[3:-1]) for x in det]
#print(det[1375:1380])
i=0
y=[]
while i < len(det)-75:
    y.append(det[i+75])
    i+=75
y.append(det(np.ceil((len(det)+i)/2)))
X=pd.read_csv('DimRedData.csv',header=None)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.20)


model=SVC(C=10,kernel='linear',gamma='scale')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred))

      
