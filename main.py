
# -*- coding: utf-8 -*-
# ---------------------MEEG_被试内------------
import argparse
from DataProcess import * 
from MEEG_model import *
from Evalution import * 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description = 'input the dataset dir path.')
    parser.add_argument('--getrawpath',
					   type = str,
					   default = 'Raw_data/',
					   help = 'File path to the dataset raw')
    parser.add_argument('--geteapath',
					   type = str,
					   default = 'EA_data/',
					   help = 'input the ea dataset dir')
    parser.add_argument('--choosedata',
						type = str,
						default = 'raw',
						help = 'choose raw:input raw,choose ea:input ea')
    args = parser.parse_args()
    
    data_raw_path = args.getrawpath
    data_raw_files =30
    data_ea_path = args.geteapath
    data_ea_files = 30
        
    if args.choosedata == 'raw':
        GetData = DataProcess()
        data_eeg,label_eeg,data_hbr,label_hbr = GetData.load_npy_for_raw(data_raw_path,data_raw_files)
    else: 
        GetData = DataProcess()
        data_eeg,label_eeg,data_hbr,label_hbr = GetData.load_npy_for_ea(data_ea_path,data_ea_files)
                
    validation_acc = []# validation scores
    test_acc = []
    matrixes = []         # Confusion matrix
    kappas = []           # kappa 
    f1_scores = []        # f1 scores
    loss_scores = []           # loss
    errors = []           # errors  
    AUC = []         
    k = 10                
    for fold in range(k): 
        cv=27 
        print(cv)
        X = data_eeg[60 * cv:60 * (cv+1)]
        Y = label_eeg[60 * cv:60 * (cv+1)]
        eeg, test_eeg, label, test_label = train_test_split(X, Y, test_size=0.1,stratify = Y)          
        train_eeg, validation_eeg,train_label, validation_label = train_test_split(eeg,label, test_size=0.1,stratify = label)
        GetData = DataProcess()
        train_eeg,train_label = GetData.sliding_window(train_eeg,train_label,200)
        validation_eeg,validation_label = GetData.sliding_window(validation_eeg,validation_label,200)
        test_eeg,test_label = GetData.sliding_window(test_eeg,test_label,200)
		# shuffle
        index_validation = [i for i in range(len(validation_eeg))] 
        np.random.shuffle(index_validation)
        validation_eeg = validation_eeg[index_validation]
        validation_label = validation_label[index_validation]
        
        index_train = [i for i in range(len(train_eeg))] 
        np.random.shuffle(index_train)
        train_eeg = train_eeg[index_train]
        train_label = train_label[index_train]
       
        
        train_eeg = train_eeg.reshape(train_eeg.shape[0],train_eeg.shape[1], 8, 1)
        validation_eeg = validation_eeg.reshape(validation_eeg.shape[0], 
                                                validation_eeg.shape[1],8, 1)
        test_eeg = test_eeg.reshape(test_eeg.shape[0],test_eeg.shape[1], 8, 1)
                    
        print(validation_eeg.shape,"\n",validation_label.shape)
        print(train_eeg.shape,"\n",train_label.shape) 

        
        dummy_matrix1=np.zeros((train_eeg.shape[0],1))             
        dummy_matrix2=np.zeros((validation_eeg.shape[0],1))
        dummy_matrix3=np.zeros((test_eeg.shape[0],1))
        y_train = train_label[:,0] 
        y_vali = validation_label[:,0]
        y_test = test_label[:,0] 
        
        MEEG = MEEG_model()            
        model = MEEG.build_model(channels_eeg = 8,samples_eeg = 600)
        model.summary()
        opt = fold+1
        
        filepath = 'EEG_cl/best_model' + str(cv+1)+'_'+str(opt) + '.h5' 
        
        callbacks = MEEG.create_callbacks(opt,cv,filepath)  
        adam = optimizers.Adam()    
        model.compile(optimizer = adam,
                       loss=['binary_crossentropy',lambda y_true,y_pred:y_pred],
                       loss_weights=[2.5,5.0],
                       metrics={'softmax':'accuracy'})
                                                            
        hist = model.fit([train_eeg,y_train],[train_label,dummy_matrix1],
                         epochs=200,		 
                         batch_size=16,
                         callbacks=callbacks,
                         validation_data = ([validation_eeg,y_vali],
                         [validation_label,dummy_matrix2]))                   
                                                                                     
  					    							
        best_epoch = np.argmax(hist.history['val_softmax_accuracy'])#返回最大下标
        best_acc = hist.history['val_softmax_accuracy'][best_epoch]
        
        #加载具有最高验证精度的模型  							                  
        model.load_weights(filepath)   
        scores =  model.evaluate([test_eeg,y_test],
                                 [test_label,dummy_matrix3])
 #    #
 #            print("%s: %.2f%%" % (model.metrics_names[-1], scores[-1]*100))
    
        y_pred =model.predict([test_eeg,y_test]) 
        y_pre = y_pred[0] 
        y_pre = y_pre[:,0] 
        Evalutions = Evalution()                                           
        Result = Evalutions.matrix_and_kappa(y_pre,y_test)
         
#            histories.append(hist.history)    
        test_acc.append(scores[-1])
        save('results_EEG_cl/test_acc'+str(cv+1)+'_'+str(opt)+'.npy',test_acc) 
        
        matrixes.append(Result[0])
        save('results_EEG_cl/matrixes'+str(cv+1)+'_'+str(opt)+'.npy',matrixes) 
        
        kappas.append(Result[1])
        save('results_EEG_cl/kappas'+str(cv+1)+'_'+str(opt)+'.npy',kappas) 
        
        f1_scores.append(Result[2])
        save('results_EEG_cl/f1_scores'+str(cv+1)+'_'+str(opt)+'.npy',f1_scores)
        
        AUC.append(Result[3])
        save('results_EEG_cl/AUC'+str(cv+1)+'_'+str(opt)+'.npy',AUC) 
        
        validation_acc.append(best_acc)
        save('results_EEG_cl/vali_acc'+str(cv+1)+'_'+str(opt)+'.npy',validation_acc) 

           
        acc = hist.history['softmax_accuracy']
        val_acc = hist.history['val_softmax_accuracy']
            
        softmax_loss = hist.history['softmax_loss']
        val_softmax_loss = hist.history['val_softmax_loss']            
#            center_loss=   hist.history['lambda_57_loss'] 
#            val_center_loss = hist.history['val_lambda_57_loss']   
                                                                                                                                                                                            
        epochs = range(len(acc))
        
        plt.figure()     
        plt.plot(epochs, acc, 'y', label='Training acc') # 'bo'为画蓝色圆点，不连线
        plt.plot(epochs, val_acc, 'g', label='Validation acc') 
        plt.title('Training and validation accuracy')
        plt.legend() # 绘制图例，默认在右上角
        plt.savefig('results_EEG_cl/acc_'+str(cv+1)+'_'+str(opt)+'.png')
        
        plt.figure()                 
        plt.plot(epochs, softmax_loss, 'y', label='Training loss')
        plt.plot(epochs, val_softmax_loss, 'g', label='Validation loss')    
#            plt.plot(epochs, softmax_loss, 'r', label='Training loss1')
#            plt.plot(epochs, val_softmax_loss, 'b', label='Validation loss1')
#            
#            plt.plot(epochs, center_loss, 'o', label='Training loss2')
#            plt.plot(epochs, val_center_loss, 'hotpink', label='Validation loss2')
#            
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig('results_EEG_cl/loss_'+str(cv+1)+'_'+str(opt)+'.png')
        plt.show() 

   
#        T-sne
#        get_feature_model = keras.Model(inputs = model.input,
#                                  outputs = model.get_layer('hidden_features').output)
        
#        hidden_features = get_feature_model.predict([train_eeg,y_train])
#        print(hidden_features.shape)
#        
#        import numpy as np
#        #import matplotlib.pyplot as plt
#        #from keras import backend as K
#        from sklearn import manifold,decomposition
#        pca = decomposition.PCA(n_components=2)# 总的类别
#        pca_result = pca.fit_transform(hidden_features)
#        print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))
#        
#        #Run T-SNE on the PCA features.
#        tsne = manifold.TSNE(n_components=2, verbose = 1)
#        tsne_results = tsne.fit_transform(pca_result[:5000])
#        
#        #-------------------------------可视化--------------------------------
#        from keras.utils import np_utils
#        y_test_cat = np_utils.to_categorical(y_train, num_classes = 2)# 总的类别
#        color_map = np.argmax(y_test_cat, axis=1)
#        plt.figure(figsize=(12,10))
#        #for cl in range(2):# 总的类别
#        #    indices = np.where(color_map==cl)
#        #    indices = indices[0]
#        #    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1])
#        indices = np.where(color_map==0)
#        indices = indices[0]
#        plt.scatter(tsne_results[indices,0], tsne_results[indices, 1],s=15,c='b')
#        indices = np.where(color_map==1)
#        indices = indices[0]
#        plt.scatter(tsne_results[indices,0], tsne_results[indices, 1],s=15,c='r')
#           
#        font_size = 15
#        plt.tick_params(labelsize=font_size)
#        plt.legend(('left', 'right'), loc='upper left')
##        plt.xlim(-50, 50.0)
##        plt.ylim(-50, 50.0)
##        #plt.axis('off')
##        plt.xticks([])
##        plt.yticks([])
#        plt.savefig('results_MEEG/tsne_'+str(cv+1)+'_'+str(opt)+'.png')
#        plt.show()
### 相关库
##
#def plot_matrix(y_true, y_pred, labels_name, title=None, axis_labels=None):
## 利用sklearn中的函数生成混淆矩阵并归一化
#    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵 
#    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
#    plt.figure(figsize=(7, 5))
## 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
#    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
#    pl.colorbar()  # 绘制图例
#
#
## 图像标题
#    if title is not None:
#        pl.title(title,fontsize=15)
## 绘制坐标
#    num_local = np.array(range(len(labels_name)))
#    if axis_labels is None:
#        axis_labels = labels_name
#    pl.xticks(num_local, axis_labels,fontsize=15, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
#    pl.yticks(num_local, axis_labels,fontsize=15)  # 将标签印在y轴坐标上
#    pl.ylabel('True label',fontsize=15)
#    pl.xlabel('Predicted label',fontsize=15)
#    
#    thresh = cm.max() / 2.
# 
## 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
#    for i in range(np.shape(cm)[0]):
#        for j in range(np.shape(cm)[1]):
#            if int(cm[i][j] * 100 + 0.5) > 0:
#                pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
#                        ha='center', va='center',
#                        color="white" if cm[i][j] > thresh else "black",fontsize=15)  # 如果要更改颜色风格，需要同时更改此行
## 显示
#    pl.show()
#    
#
#plot_matrix(original_test_label[:,[1]],predictions1, [0, 1], title='confusion_matrix',
#            axis_labels=['left', 'right'])