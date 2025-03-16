# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix,cohen_kappa_score,f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc 
class Evalution:
    
#    def __init__ (self,
#                   history,
#                   y_pred,
#                   y_true,
#                   loss_score,
#                   error,
#                   validation_score):
#        self.history = history
#        self.y_pred = y_pred
#        self.y_true = y_true
#        self.loss_score = loss_score
#        self.error = error
#        self.validation_score = validation_score
#        

        
    # Matrix confusion and the kappa value,f1
    def matrix_and_kappa(self,y_pre,y_true):
#        for i in y_pre:
#            if i > 0.5:
#                i = 1
#            else:
#                i = 0
        y_pred = np.array(y_pre)        
        y_pred_1 = []
        y_true_1 = []
        for i in y_pred:
            if i > 0.5:
                y_pred_1.append(1)
            else:
                y_pred_1.append(0)
        for i in y_true:
            if i == 1:
                y_true_1.append(1)
            else:
                y_true_1.append(0)
        C2 = confusion_matrix(y_true_1,y_pred_1)
        kappa_value = cohen_kappa_score(y_true_1, y_pred_1)
        f1 = f1_score(y_true_1,y_pred_1, average = None)
        fpr,tpr,threshold1 = roc_curve(y_true_1,y_pred_1) ###计算真正率和假正率
        #fpr,tpr = roc_curve(original_test_label[:,[1]],y_pre[:,[1]]) ###计算
        roc_auc = auc(fpr,tpr) ###计
        return C2,kappa_value,f1,roc_auc
    

    
    def draw_pict(self,history,types = 1):
        if types == 1:
            plt.plot(history.history['softmax_accuracy'])
            plt.plot(history.history['val_softmax_accuracy'])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='lower right')  
            plt.title('accuracy')
            plt.show()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.ylabel('loss') 
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='upper right')  
            plt.title('loss')
            plt.show()
        else:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='lower right')  
            plt.title('accuracy')
            plt.show()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.ylabel('loss') 
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='upper right')  
            plt.title('loss')
            plt.show()
        return
    