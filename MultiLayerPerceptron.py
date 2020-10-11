import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense ##fully connected neural network
import keras
from OptimizationTools import PSO
from keras.callbacks import  CSVLogger




white_wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red_wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
red_wine["type"]=1
white_wine["type"]=0
wines = [red_wine,white_wine]
wines = pd.concat(wines)
corr=wines.corr()

y = np.ravel(wines.type)

x=wines.ix[:,:12]
y=wines.ix[:,12]
x_train,x_test,y_train,y_test_ =train_test_split(x,y,test_size=0.33)

model= Sequential() #initialize network
model.add(Dense(13,activation='relu',input_shape=(12,))) #first layer, inputs
model.add(Dense(8,activation='relu')) #hidden layer
model.add(Dense(1,activation='sigmoid')) #output layer

opt = keras.optimizers.Adam(learning_rate=.5)
model.compile(loss='binary_crossentropy',optimizer=opt)

csv_logger = CSVLogger('log.csv',append=True,separator=';')#archivo vacio para guardar resultados de f costo
model.fit(x_train,y_train,epochs=10,batch_size=1,verbose=1,callbacks=[csv_logger])
loss_history = pd.read_csv("C:/Users/anuno/OneDrive/Documents/ITESO/PAP 2/log.csv",sep=';')



