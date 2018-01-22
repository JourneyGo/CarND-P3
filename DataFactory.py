# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:18:39 2018

Doing load data things

@author: Qi
"""
import csv
import cv2
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

#Need to CHANGE this var in different environment
current_data_path = os.getcwd()+'/data/'

def readLog(data_path = './data/'):
    data_path = data_path
    lines = []
    with open(data_path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    print("batch data:",num_samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            steerings = []
            direction_bias = [0,0.2,-0.2]
            for batch_sampel in batch_samples:
                cam_pos = np.random.randint(3)
                imgname = current_data_path +'IMG/'+ batch_sampel[cam_pos].split('\\')[-1]
                img = cv2.imread(imgname)
                angle = float(batch_sampel[3])+direction_bias[cam_pos]
                if np.random.randint(10)%2 == 0:
                    img = np.fliplr(img)
                    angle = -angle
                images.append(img)
                steerings.append(angle)
                
            X_train = np.array(images)
            y_train = np.array(steerings)

            yield sklearn.utils.shuffle(X_train,y_train)
            
            

def loadData(batch_size = 32):
    '''
    return TRAIN,VALIDATION generator that provide data in batch_size
    '''
    samples = readLog()
    print('All',len(samples),'data loaded')
    train_samples, validation_samples = train_test_split(samples, test_size = 0.1)
    train_epoch_num = len(train_samples)
    valid_epoch_num = len(validation_samples)
    train = generator(train_samples,batch_size)
    val = generator(validation_samples,batch_size)
    return train,val,train_epoch_num,valid_epoch_num

def testDataFactory():
    import matplotlib.pyplot as plt
    train,val,train_epoch_num,valid_epoch_num = loadData(2)
    n = np.random.randint(valid_epoch_num/2)
    i=0
    while i < n-1:
        next(train)
        next(val)
        i+=1
    t_img,t_v = next(train)
    v_img,v_v = next(val)
    plt.subplot(2,2,1)
    plt.imshow(t_img[0])
    plt.title(t_v[0])
    plt.subplot(2,2,2)
    plt.imshow(t_img[1])
    plt.title(t_v[1])
    plt.subplot(2,2,3)
    plt.imshow(v_img[0])
    plt.title(v_v[0])
    plt.subplot(2,2,4)
    plt.imshow(v_img[1])
    plt.title(v_v[1])
    