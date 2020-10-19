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

