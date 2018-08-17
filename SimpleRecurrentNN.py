#Project Alpha. Simple Recurrent Neural Network
import numpy as np
import pandas as pd
import time
loc = "Data\COE.csv"
temp=pd.read_csv(loc)
#Till here we load data from outside to loc.
 
data = temp.drop(temp.columns[[0,1]],axis=1)   #Here we are removing columns 0 and 1
y=data['COE$']  #This is the target data. The COE$ which is stored in y

x=data.drop(data.columns[[0,4]],axis=1) #All the other variables are stored in x
x=x.apply(np.log)
x=pd.concat([x,data['Open?']],axis=1) #Since the Open? data is binary it added later as concat

from sklearn import preprocessing
scaler_x=preprocessing.MinMaxScaler(feature_range=(0,1))    #Attributes are scaled
x=np.array(x).reshape((len(x),4))
x=scaler_x.fit_transform(x)

scaler_y=preprocessing.MinMaxScaler(feature_range=(0,1))    #The main target is scaled
y=np.array(y).reshape((len(y),1))
y=np.log(y)
y=scaler_y.fit_transform(y)

#import data_processing  This imports the data_processing.py from directory
#If required it would be nice to copy the code of data_processing.py
#Training starts. 95% of data is used for training and rest for testing
end=len(x)
learn_end=int(end*0.954)
x_train=x[0:learn_end - 1,]
x_test=x[learn_end:end,]
y_train=y[0:learn_end - 1,]
y_test=y[learn_end:end,]
x_train=x_train.reshape(x_train.shape + (1,))
x_test=x_test.reshape(x_test.shape + (1,))

#The test and train variables are set in the above set.

from keras.models import Sequential  #Linear stcking of layers
from keras.optimizers import SGD #
from keras.layers.core import Dense #Activation function and dense layer for output layer.
from keras.layers.recurrent import SimpleRNN #Imports a RNN were the outputs are fed into inputs.

#Determining model structure
seed=2016
np.random.seed(seed)
model=Sequential() #Model is stored in fit1
model.add(SimpleRNN(16,activation="tanh",input_shape=(4,1))) #Input shape is number of arguments. In this case 4
#The above line is for the first layer and must be done in a new console. If a layer is changed it wont work
model.add(Dense(1,activation='linear')) #Output layer connected to forecast via activation function

#Choosing momentum value
sgd=SGD(lr=0.0001, momentum=0.95,nesterov=True)
model.compile(loss='mean_squared_error',optimizer=sgd)
#Training is done here. With 700 iterations.
model.fit(x_train, y_train, batch_size=10, epochs=1000)
print("\n")
score_train=model.evaluate(x_train,y_train, batch_size=10)
print("\n")
score_test=model.evaluate(x_test, y_test, batch_size=10)
#Printing Values
time.sleep(1)
print("\n")
print("In train MSE =" , round(score_train,6))
time.sleep(1)
print("\n")
print("In test MSE =", round(score_test,6))
pred1=model.predict(x_test)
pred1=scaler_y.inverse_transform(np.array(pred1).reshape((len(pred1),1)))
pred1=np.exp(pred1)
time.sleep(3)
print("\n")
pred=np.rint(pred1)
print(pred)

y_test1=scaler_y.inverse_transform(np.array(y_test).reshape((len(y_test),1)))
y_test1=np.exp(y_test1)
y_test1=np.rint(y_test1)

import matplotlib.pyplot as plt
plt.figure(figsize=(5.5,5.5))
plt.plot(range(12),y_test1[:12],linestyle='-',marker='*',color='r')
plt.plot(range(12),pred[:12],linestyle='-',marker='.',color='b')
plt.legend(['Actual','Predicted'],loc=2)
plt.title('Actual vs Predicted')
plt.ylabel ('Value')
plt.xlabel('Index')
plt.show()