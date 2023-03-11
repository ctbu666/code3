import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import scipy.io as scio

from keras import backend as K
from keras import regularizers
from keras.layers import Input, Dense, Dropout,Permute,multiply,Lambda
from keras.models import Model,Sequential
from keras.utils.np_utils import to_categorical

#from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.layers import concatenate
from keras.utils import plot_model
from keras.models import load_model

K.clear_session()

# 读取
Data = scio.loadmat('Dataforpython.mat') # 读取mat文件
Trajectory=Data['SOC']
Trajectory=Trajectory.tolist()
Trajectory=Trajectory[0]


window=20
def  attentiont(inputs):

    #input_dim =  int(inputs.shape[2]) # shape = (batch_size, time_steps, input_dim)
    
    a = Permute((2, 1))(inputs) # shape = (batch_size, input_dim, time_steps)
    #a = Reshape((input_dim, TimeStep))(a) # this line is not useful. It's just to know which dimension is what.
    Fa = Dense(10, activation='softmax', name='fa')(a)# 为了让输出的维数和时间序列数相同（这样才能生成各个时间点的注意力值）    
    
    
    Fa = multiply([a, Fa]) #把注意力值和输入按位相乘，权重乘以输入
    Fa = Permute((2, 1), name='at')(Fa) # shape = (batch_size, time_steps, input_dim)
    
    return Fa
Trajectory_reorg=list()
RUL_reorg=list()
for tra in Trajectory:
    RUL=np.linspace(start = tra.size-1, stop = 0, num = tra.size)
    RUL=RUL[window-1:]
    tra_reorganize=np.zeros((tra.size-window+1,window))
    for i in range(tra.size-window+1):
        tra_reorganize[i,:]=tra[i:i+window].T
    Trajectory_reorg.append(tra_reorganize)
    RUL_reorg.append(RUL)
    del tra_reorganize, RUL


RUL_reorg=Trajectory_reorg
random.seed(58) #初始seed是1
Index_train=random.sample(range(0, len(Trajectory_reorg)),100)
Index_train.sort()

Trajectory_train=[Trajectory_reorg[i] for i in Index_train]
x_train=np.vstack(Trajectory_train)
RUL_train=[RUL_reorg[i] for i in Index_train]
y_train=np.concatenate(RUL_train)

Index_test = list(set(range(0,len(Trajectory_reorg))) ^ set(Index_train))
Trajectory_test=[Trajectory_reorg[i] for i in Index_test]
x_test=np.vstack(Trajectory_test)
RUL_test=[RUL_reorg[i] for i in Index_test]
y_test=np.concatenate(RUL_test)

y_train=y_train[:,1]
y_test=y_test[:,1]

from data_provider import *


path_in = './dataset/encoded/data_encoded.pkl'
with open(path_in, 'rb') as f:
    images_train,images_test = pickle.load(f)

x_train=images_train
x_test=images_test

x_train=x_train.reshape(len(x_train),10,1)
x_test=x_test.reshape(len(x_test),10,1)
y_train=y_train.reshape(len(y_train),1)
y_test=y_test.reshape(len(y_test),1)

# Train GRU and save results
from keras.layers import GRU, Masking
model_input = Input(shape=(10, 1))
a = Masking(mask_value=0.)(model_input)

a=attentiont(a)
#a = (GRU(50,return_sequences=True))(a)
#a = Lambda(lambda x: x, output_shape=lambda s:s)(a)
a = (GRU(100,return_sequences=False))(a)
a = (Dense(1))(a)
model = Model(model_input, a)
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
cc=model.predict(x_test)


mse=((y_test- cc) ** 2).mean()
rmse=np.sqrt(((y_test- cc) ** 2).mean())
mae=(np.abs(y_test- cc)).mean()

print('sae+gru')
print(mse)
print(rmse)
print(mae)

y_test=y_test*100
cc=cc*100
plt.figure()
plt.plot(y_test,color='black',label='real')
plt.plot(cc,color='red',label='pred')
plt.title('gru test')

print('GRU')
plt.show()