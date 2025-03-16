# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten,Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input,Dense,Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from tensorflow.keras import backend as K


class M2NN_model:

#    def __init__(self,
#                channels_eeg,
#                samples_eeg,
#                channels_hbr,
#                samples_hbr):
#        self.channels_eeg = 8
#        self.samples_eeg = 600
#        self.channels_hbr = 24
#        self.samples_hbr = 30               
#        self.learning_rate = 0.0001
#        self.feature_size = 2
#        self.dropout = 0.8

        
#        self.model = self.build_model(self.channels_eeg,self.samples_eeg,
#                                      self.channels_hbr,self.samples_hbr)
#        adam = optimizers.Adam(lr = self.learning_rate) 
##        self.model.compile(optimizer = adam,
##                           loss=losses.binary_crossentropy,                           
##                           metrics=['mae',metrics.binary_accuracy])
#        self.model.compile(optimizer = adam,
#                           loss=['binary_crossentropy',lambda y_true,y_pred:y_pred],
#                           loss_weights=[2.5,5.0],
#                           metrics={'softmax':'accuracy'})
                                        
    def build_model(self,channels_eeg,samples_eeg,channels_hbr,samples_hbr):    
    
        input1 = Input(shape=(samples_eeg,channels_eeg,1))
        input2 = Input(shape=(samples_hbr,channels_hbr,1))        
#        input1 = Input(shape=(600,8,1))
#        input2 = Input(shape=(30,24,1))
        
       
        x1 = self.Block1(input1, filters=16, kernel_size=(1, 2), pool_size=(1,2))
    #    x1 = channel_attention(x1,4)     
        x1 = self.Block1(x1, filters=32, kernel_size=(1, 2), pool_size=(1,2))
        x1 = self.Block1(x1, filters=32, kernel_size=(4, 1), pool_size=(4,1))    
        x1 = self.Block1(x1, filters=64, kernel_size=(4, 1), pool_size=(4,1))
    #    x1 = self_attention(x1,16)
        x1 = Flatten()(x1)
    
       
        x2 = self.Block2(input2, filters=16, kernel_size=(1, 4), pool_size=(1,4)) 
    #    x2 = channel_attention(x2,4)     
        x2 = self.Block2(x2, filters=32, kernel_size=(1, 4), pool_size=(1,2)) 
        x2 = self.Block2(x2, filters=32, kernel_size=(2, 1), pool_size=(2,1)) 
        x2 = self.Block2(x2, filters=64, kernel_size=(2, 1), pool_size=(2,1))
    #    x2 = self_attention(x2,16)
        x2 = Flatten()(x2)
        x3 = Concatenate()([x1, x2])
    #    x = channel_attention(x,2) 
        out = Dense(256, activation='sigmoid')(x3)
        out = Dense(128, activation='sigmoid')(out)  
        out = Dense(64, activation='sigmoid',name ='hidden_features')(out)
        
        outputs = Dense(2, activation='softmax',name ='softmax')(out)
        
        out_feature = Dense(2, activation='sigmoid')(out)
    
        
        input3 = Input(shape=(1,))
        
        centers =  Embedding(2,2)(input3)
        
        intra_loss =  keras.layers.Lambda(lambda x:K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True))([out_feature,centers])
        
        model = Model([input1,input2,input3],[outputs,intra_loss])
        #    model_with_center1 = keras.Model([input1,input2],outputs)
        #    model_with_center2 = keras.Model([input1,input2,input3],intra_loss)
        return model 
            
    
        
    '''
    eeg_block
    '''
    def Block1(self,inputs, filters, kernel_size, pool_size):
        x = Conv2D(filters, kernel_size, strides=1, padding='same', activation='elu')(inputs)
        x = Conv2D(filters, kernel_size, strides=1, padding='same', activation='elu')(x)
    #    x = channel_attention(x,4)  
        x = BatchNormalization(axis=-1)(x)
        s = pool_size
        x = MaxPooling2D(pool_size=pool_size, strides=s)(x)
        return x
        
    '''
    hbr_block
    '''
    def Block2(self,inputs, filters, kernel_size, pool_size):
        x = Conv2D(filters, kernel_size, strides=1, padding='same', activation='elu')(inputs)
    #    x = keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same', activation='elu')(x)    
    #    x = keras.layers.Dropout(0.1)(x)
        x = BatchNormalization(axis=-1)(x)    
        s = pool_size
        x = MaxPooling2D(pool_size=pool_size, strides=s)(x)
    #    x = keras.layers.Dropout(0.1)(x)
    
        return x
     
    def create_callbacks(self,opt,cv):
        '''
        回调函数
        :return:callbacks，类型list
        '''
     
        # 一般是在model.fit函数中调用callbacks，fit函数中有一个参数为callbacks。
        # 注意这里需要输入的是list类型的数据，所以通常情况只用EarlyStopping的话也要是[EarlyStopping()]
        filepath ='model_M2NN_EA/best_model_' + str(cv+1)+'_'+str(opt) + '.h5'        
        callbacks = [
            EarlyStopping(monitor='val_softmax_accuracy',patience=50,verbose=0,mode='max'),
            ModelCheckpoint(filepath, monitor='val_softmax_accuracy', save_best_only=True, verbose=0)
        ]
        return callbacks
 

