# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:24:04 2018

@author: Qi
"""
from DataFactory import loadData
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D,Dense,Lambda,Flatten,MaxPooling2D
from keras.models import Sequential    

#get train/val data as generator 
batch_size = 128   
train_generator,validation_generator,train_epoch_num,valid_epoch_num = loadData(batch_size)   
ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(row, col,ch),output_shape=(row, col,ch)))
model.add(Cropping2D(((65,25),(0,0))))
model.add(Convolution2D(32,(5,5),activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(64,(5,5),activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(256,(3,3),activation='relu'))
model.add(Convolution2D(32,(1,1),activation='relu'))
model.add(Convolution2D(16,(1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
            train_epoch_num/batch_size, validation_data=validation_generator, \
            nb_val_samples=valid_epoch_num/batch_size, nb_epoch=6)
model.save('model_small.h5')

