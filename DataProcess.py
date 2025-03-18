# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
import random
from pylab import *
from numpy import *
from scipy import interpolate
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#from sklearn.metrics import cohen_kappa_scor


class DataProcess():

    def load_npy_for_raw(self,data_path, data_files):
        print('load Raw data')
        # Design by yourself                                 
        return data1,label1,data2,label2
    

#    Sliding window data augmentation
#    '''
#    该函数实现窗口宽度为3s、滑动步长为1s的滑动窗口截取序列数据
#    '''        
    def sliding_window(self,data,lable, fs, windows_long = 3):
        
        X_data,y_data = [],[]
        for i in range(data.shape[0]):
            data_t = data[i].T
            lable_temp = lable[i]
            X, y = [], []
            for start in range(8):
                in_ = start*fs
                end = in_ + windows_long*fs
                train_seq = data_t[:,in_:end]
                #print(data_t.shape)
                train_seq = train_seq.T
                X.append(train_seq)
                y.append(lable_temp)
            X_data.append(X)
            y_data.append(y)
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        X_data = X_data.reshape(data.shape[0]*8,windows_long*fs,data.shape[-1])
        y_data = y_data.reshape(data.shape[0]*8,lable.shape[-1])
        
        return X_data, y_data

    

