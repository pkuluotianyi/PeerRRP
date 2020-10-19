from __future__ import print_function
from os import listdir
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from os import listdir
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

import csv

x_train_list = []
y_train_list = []
x_test_list = []
y_test_list = []


with open("./data/metrics_classifier_feature_temp.csv","r") as metrics_classifier_feature: 
    reader_train = csv.reader(metrics_classifier_feature)
    
    line_num = 0
    for row in reader_train:
        print(str(row))
        line_num += 1
        if line_num <= 36: 
            x_train_list.append([float(row[1]), float(row[2]), float(row[3])])
            y_train_list.append(int(row[5]))
        else:
            x_test_list.append([float(row[1]), float(row[2]), float(row[3])])
            y_test_list.append(int(row[5]))


    #X = preprocessing.normalize(X_original, norm='l2', axis=0)
    #print(X.shape[1])
        
    X_train_original = numpy.array(x_train_list, dtype='float')
    X_train = preprocessing.normalize(X_train_original, norm='l2', axis=0)
    #X_train = X_train_original
    y_train = numpy.array(y_train_list, dtype='int64')
    
        
    #replace noise label ends!
    
    X_test_original = numpy.array(x_test_list, dtype='float')
    X_test = preprocessing.normalize(X_test_original, norm='l2', axis=0)
    #X_test = X_test_original
    y_test = numpy.array(y_test_list, dtype='int64')
    
#     clf_lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
#     clf_lr.fit(X_train, y_train)
#     score_train_lr = clf_lr.score(X_train, y_train) 
#     predict_train_lr = clf_lr.predict(X_train)
#     print("Accuracy on train data using LR:   " + str(score_train_lr))
#     print("Prediction on train data using LR:   " + str(predict_train_lr))
#     score_test_lr = clf_lr.score(X_test, y_test)
#     predict_test_lr = clf_lr.predict(X_test)
#     predict_proba_test_lr = clf_lr.predict_proba(X_test)
#     print("Accuracy on test data using LR:   " + str(score_test_lr))
#     print("Prediction on test data using LR:   " + str(predict_test_lr))
#     print("Prediction probability on test data using LR:   " + str(predict_proba_test_lr))
    
    
    
    clf_svm = svm.SVC(kernel='linear', gamma='scale', decision_function_shape='ovr', probability=True)
    clf_svm.fit(X_train, y_train)
    score_train_svm = clf_svm.score(X_train, y_train) 
    predict_train_svm = clf_svm.predict(X_train)
    print("Accuracy on train data using SVM:   " + str(score_train_svm))
    print("Prediction on train data using SVM:   " + str(predict_train_svm))
    score_test_svm = clf_svm.score(X_test, y_test)
    predict_test_svm = clf_svm.predict(X_test)
    predict_proba_test_svm = clf_svm.predict_proba(X_test)
    print("Accuracy on test data using SVM:   " + str(score_test_svm))
    print("Prediction on test data using SVM:   " + str(predict_test_svm))
    print("Prediction probability on test data using SVM:   " + str(predict_proba_test_svm))
    print("\n")


#     clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4*X_train.shape[1], 2), random_state=1)
#     clf_mlp.fit(X_train, y_train)
#     score_train_mlp = clf_mlp.score(X_train, y_train) 
#     predict_train_mlp = clf_mlp.predict(X_train)
#     predict_proba_test_mlp = clf_mlp.predict_proba(X_test)
#     print("Accuracy on train data using MLP:   " + str(score_train_mlp))
#     print("Prediction on train data using MLP:   " + str(predict_train_mlp))
#     score_test_mlp = clf_mlp.score(X_test, y_test)
#     predict_test_mlp = clf_mlp.predict(X_test)
#     print("Accuracy on test data using MLP:   " + str(score_test_mlp))
#     print("Prediction on test data using MLP:   " + str(predict_test_mlp))

    
    
    
    
    
    
    
    
    
            
         
        
