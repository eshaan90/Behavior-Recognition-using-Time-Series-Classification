#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:09:10 2019

@author: MyReservoir
"""

import pandas as pd
import numpy as np
import math
import keras 
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, TimeDistributed, Dropout,MaxPooling1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

class Sessions:
    def __init__(self, subject_name, armIMU, wristIMU, time, detection):
        self.subject_name=subject_name
        self._armIMU=armIMU
        self._wristIMU=wristIMU
        self._time=time
        self._detection=detection

    def get_arm_accelerometer_data(self):
        return self._armIMU.iloc[:,0:3]
    
    def get_arm_gyro_data(self):
        return self._armIMU.iloc[:,3:]    
   
    def get_wrist_accelerometer_data(self):
        return self._wristIMU.iloc[:,0:3]
    
    def get_wrist_gyro_data(self):
        return self._wristIMU.iloc[:,3:]
    
    def join_features(self):
        X=pd.concat([self._time, self._armIMU, self._wristIMU, self._detection], axis=1)
        return X
    
    def get_y(self):
        return self._detection     


def get_featuresIMU(x,fs):
    '''
    This function computes some temporal and frequency based features from
    IMU signals. It is assumed that x is of size N x 3, where N is the number of 
    samples in the window, and 3 is because we are considering accelleration or 
    gyroscope alone. The 'fs' is the nominal frequency of sampling used for this
    device.
    '''
    f1=x.mean().values
    C=x.cov()
    f2=np.array([C.iloc[0,0:].values,C.iloc[1,1:].values,C.iloc[2,2]])
    f2=np.hstack((f2[0],f2[1],f2[2]))
    f3=x.skew().values
    f4=x.kurtosis().values
    
    f5=[0]*3
    f6=[0]*3
    for i in range(3):
        g=abs(np.fft.fft(x.iloc[:,i]))
        l=len(g)
        g=g[0:round(l/2)]
        g[0]=0
        w=fs*(np.array([*range(l)])/(2*l))
        v,idx=max(g),np.argmax(g)
        f5[i]=v
        f6[i]=w[idx]
    
    F=np.hstack((f1, f2, f3, f4, f5, f6))
    
    return F.T
    


path=os.getcwd()+'/Training Data/'
sessions=[]
session_name=[]
for file in os.listdir(path):
     if 'Session' in file: 
         session_name.append(file)
         subject_path=os.path.join(path, file)
         arm=pd.read_fwf(subject_path+'/armIMU.txt',  header= None)
         wrist=pd.read_fwf(subject_path+'/wristIMU.txt', header=None)
         time=pd.read_csv(subject_path+'/time.txt', header=None)
         detection=pd.read_csv(subject_path+'/detection.txt', header=None, dtype='int64')
         sessions.append(Sessions(file, arm, wrist, time, detection))
     else:
         continue
     
X=sessions[0].join_features()
color=['r','g','b','y']
num_classes=2

cmap=matplotlib.colors.ListedColormap(color)
plt.figure(figsize=(20,20))
plt.plot(X.iloc[:,0],X.iloc[:,3])
plt.xlabel('Time')
plt.ylabel('Signal')
plt.show()


plt.figure(figsize=(20,20))
plt.scatter(X.iloc[:30000,0],X.iloc[:30000,3],c=X.iloc[:30000,13])
plt.legend(['Normal','Body Rocking'])
plt.xlabel('Time')
plt.ylabel('Signal')
plt.savefig('arm1')


def folding_data(X):
    fs=50
    tau=150
    R=.83
    T=len(X)
    S=math.floor(tau*(1-R))
    frames=(T-tau)/S+1
    new_dim=math.floor(frames)
    
    data=[]
    labels=[]
    #d2=np.zeros((150,12,new_dim))
    y=X.iloc[:,13].values
    for i in range(0,T,S):
        data.append(X.iloc[i:i+tau,1:13].values)
        frame_y=y[i:i+tau]
        labels.append(stats.mode(frame_y)[0][0])
    
    data=data[:new_dim]
    labels=labels[:new_dim]
    
    data_stacked=np.stack(data, axis=0)
    return (data_stacked,labels)

data={}

for i in range(len(sessions)):
    data[session_name[i]]=folding_data(sessions[i].join_features())

for key,val in data.items():
    print(key,val.shape)

pickle_out=open('dict.pickle','wb')
pickle.dump(data,pickle_out)
pickle_out.close()


pickle_in = open("dict.pickle","rb")
data = pickle.load(pickle_in)


'''
Statistical tests make strong assumptions about your data. They can only be used to inform the degree to which a null hypothesis can be rejected or fail to be reject.

The Augmented Dickey-Fuller test is a type of statistical test called a unit root test.

The intuition behind a unit root test is that it determines how strongly a time series is defined by a trend.

There are a number of unit root tests and the Augmented Dickey-Fuller may be one of the more widely used. It uses an autoregressive model and optimizes an information criterion across multiple different lag values.

The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.
'''


from statsmodels.tsa.stattools import adfuller
adf=adfuller(X.iloc[:,1].values)
print('ADF Statistic: %f' % adf[0])
print('p-value: %f' % adf[1])
print('Critical Values:')
for key, value in adf[4].items():
	print('\t%s: %.3f' % (key, value))



def evaluate_model(trainX, trainy,testX,testy):
    model=Sequential()
    model.add(LSTM(100,input_shape=(150,12)))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    #fit network
    history=model.fit(trainX,trainy, epochs=epochs, batch_size=batch_size, verbose=1)
    _,accuracy=model.evaluate(testX,testy,batch_size=batch_size,verbose=1)
    return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
    
# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# run the experiment
run_experiment()


def evaluate_model2(trainX, trainy,testX,testy):
    # define model
	verbose, epochs, batch_size = 0, 25, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = 4, 32
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	# define model
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
	model.add(TimeDistributed(Dropout(0.5)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(100, return_sequences=True))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy



readings=X.iloc[:,1:4]
g=get_featuresIMU(readings,fs)


#unfolding
T=math.floor((frames-1)*S+tau)