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

#load ea dataset
    def load_npy_for_ea(self,data_path,data_files):
        print('load EA data')
        
        data1 = []
        label1 = []
        data2 = []
        label2 = []
        
        for i in range(1,data_files):
#             print(str(i))
             data_path1 = data_path + 'EEG/EA_eeg'+str(i)+'.npy'
             label_path1 = data_path + 'EEG/y'+str(i)+'.npy'
             if i == 1:        
                data1 = np.load(data_path1)
                label1 = np.load(label_path1)
             else:
                data_t = np.load(data_path1)
#                data_t = data_t.swapaxes(0,1)
#                data_t = data_t.swapaxes(1,2)
                label_t = np.load(label_path1)
                data1 = np.concatenate((data1,data_t),axis = 0)
                label1 = np.concatenate((label1,label_t),axis = 0)
                
        for j in range(1,data_files):
#            print(str(j))
            data_path2 = data_path + 'HbR/EA_hbr'+str(j)+'.npy'
            label_path2 = data_path + 'HbR/y'+str(j)+'.npy'
            if j == 1:
                data2 = np.load(data_path2)               
                label2 = np.load(label_path2)                                
            else:
                data_t = np.load(data_path2)
#                data_t = data_t.swapaxes(0,1)
#                data_t = data_t.swapaxes(1,2)
                label_t = np.load(label_path2)
                data2 = np.concatenate((data2,data_t),axis = 0)
                label2 = np.concatenate((label2,label_t),axis = 0)
                   
        return data1,label1,data2,label2
    
  
#    load raw dataset

    def load_npy_for_raw(self,data_path, data_files):
        print('load Raw data')
        data1 = []
        label1 = [] 
        data2 = []
        label2 = []
#        subjects = 3
        for i in range(1,data_files):
#            print(i) 
            data_path1 = data_path + 'EEG/X_eeg'+str(i)+'.npy'
            label_path1 = data_path + 'EEG/Y_eeg'+str(i)+'.npy'
            if i == 1:
                data1 = np.load(data_path1)         
                label1 = np.load(label_path1)
            else:
                 data_t = np.load(data_path1)
                 label_t = np.load(label_path1)
                 data1 = np.concatenate((data1,data_t),axis = 0)
                 label1 = np.concatenate((label1,label_t),axis = 0)
                  

        for j in range(1,data_files):
             data_path2 = data_path + 'HbR/X_hbr'+str(j)+'.npy'            
             label_path2 = data_path + 'HbR/Y_hbr'+str(j)+'.npy'
             if j == 1:
                 data2 = np.load(data_path2)
                 data2 = data2[:,50:150,:]
                 label2 = np.load(label_path2)
             elif j < 11:
                data_t = np.load(data_path2)
                data_t = data_t[:,50:150,:]
                label_t = np.load(label_path2)
                data2 = np.concatenate((data2,data_t),axis = 0)
                label2 = np.concatenate((label2,label_t),axis = 0)                
#                print(str(j))
             else:
                data_t = np.load(data_path2)
                data_t = data_t[:,:,12:36]
#                data_t = data_t.swapaxes(0,1)
#                data_t = data_t.swapaxes(1,2)
                label_t = np.load(label_path2)
                data2 = np.concatenate((data2,data_t),axis = 0)
                label2 = np.concatenate((label2,label_t),axis = 0)
                                    
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

    

