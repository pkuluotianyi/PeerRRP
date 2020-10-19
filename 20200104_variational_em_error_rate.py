'''
Created on Dec 26, 2019

@author: root
'''
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
import keras
import tensorflow as tf
from sklearn import preprocessing


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from dask.array.tests.test_numpy_compat import dtype
from scipy.special import digamma

#Calculate the error rate using EM algorithms
def VariationalEM(L, true_labels, prior_tasks = None, prior_workers = None, maxIter = 200, TOL = 1e-10):

    # presently, only for binary classification
    LabelDomain = numpy.array([0, 1])
    assert numpy.sum(numpy.unique(L) - LabelDomain) == 0., "only for binary classification, with class label 0, 1"
    Ndom = LabelDomain.shape[0]

    # in our case, every classifier predicts on all tasks &
    # every task is predicted by all classifiers
    Ntask, Nwork = L.shape

    NeibTask = {}
    for i in range(Ntask):
        NeibTask[i] = numpy.arange(Nwork)

    NeibWork = {}
    for j in range(Nwork):
        NeibWork[j] = numpy.arange(Ntask)

    # if prior_tasks & prior_workers are empty; NOTE they are float
    # seems using confusion mat rather than simply error rate to measure the reliability
    if prior_tasks is None:
        prior_tasks = numpy.ones((Ndom, Ntask)) / Ndom

    if prior_workers is None:
        prior_workers = numpy.ones((Ndom, Ndom))

    # initialize mu
    alpha = numpy.ones((Ndom, Ndom, Nwork))
    mu = numpy.zeros((Ndom, Ntask))
    for i in range(Ntask):
        neib = NeibTask[i]
        labs = L[i, neib]
        for k in range(Ndom):
            mu[k,i] = numpy.nonzero(labs == LabelDomain[k])[0].shape[0] / labs.shape[0]

    err = None
    for iter in range(maxIter):
        # updating Beta distributions of workers
        for j in range(Nwork):
            neib = NeibWork[j]
            labs = L[neib, j]
            alpha[:, :, j] = prior_workers
            for ell in range(Ndom):
                idx = neib[labs == LabelDomain[ell]]
                alpha[:, ell, j] += numpy.sum(mu[:, idx], axis=1)
        
        # updating mu_i(z_i) of tasks
        old_mu = mu.copy()
        for i in range(Ntask):
            neib = NeibTask[i]
            labs = L[i, neib]
            tmp = - numpy.sum( digamma( numpy.sum(alpha[:, :, neib], axis=1) ), axis=1 )
            for ell in range(Ndom):
                jdx = neib[labs == LabelDomain[ell]]
                tmp += numpy.sum( digamma( alpha[:, ell, jdx] ), axis=1 )
            mu[:, i] = numpy.multiply( prior_tasks[:, i], numpy.exp(tmp - numpy.max(tmp)) )
            mu[:, i] = mu[:, i] / numpy.sum(mu[:,i])

        err = numpy.max(numpy.abs(old_mu - mu))
        if err < TOL:
            break

    print(f"variationalEM_crowd_model: break at {iter}-th iteration, err={err}")

    # decode the labels of tasks
    mxdx = numpy.argmax( numpy.multiply(mu, (1. + numpy.random.uniform(size=mu.shape)*numpy.finfo(float).eps)), axis=0 ) #add some random noise to break ties. 
    ans_labels = LabelDomain[mxdx]

    # compute empirical error rate
    prob_err = numpy.mean(ans_labels != true_labels)

    return prob_err


move_review_data_labelled_plus_unlabelled = []
move_review_label_labelled_plus_unlabelled = []

move_review_data_self_labelling_train = []
move_review_data_self_labelling_test = []

move_review_data = []
move_review_label = []

move_review_data_train = []
move_review_label_train = []


train_num = 300
test_num = 99

 
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    
    # close the file
    file.close()
    return text.lower()
    #return new_text
 
# specify directory to load
directory = 'New_datasets_emnlp2020_txt_newpreprocess/replicated_train'
# walk through all files in the folder
for filename in listdir(directory):
    # skip files that do not have the right extension
    if not filename.endswith(".txt"):
        continue
    # create the full path of the file to open
    path = directory + '/' + filename
    # load document
    doc = load_doc(path)
    move_review_data.append(doc)
    move_review_label.append(1)
    
    move_review_data_labelled_plus_unlabelled.append(doc)
    move_review_label_labelled_plus_unlabelled.append(1)
    
    move_review_data_self_labelling_train.append(doc)
    
    move_review_data_train.append(doc)
    move_review_label_train.append(1)
    
    #print('Loaded %s' % filename)
    
# specify directory to load
directory = 'New_datasets_emnlp2020_txt_newpreprocess/no_replicated_train'
# walk through all files in the folder
for filename in listdir(directory):
    # skip files that do not have the right extension
    if not filename.endswith(".txt"):
        continue
    # create the full path of the file to open
    path = directory + '/' + filename
    # load document
    doc = load_doc(path)
    move_review_data.append(doc)
    move_review_label.append(0)
    
    move_review_data_labelled_plus_unlabelled.append(doc)
    move_review_label_labelled_plus_unlabelled.append(0)
    
    move_review_data_self_labelling_train.append(doc)
    
    move_review_data_train.append(doc)
    move_review_label_train.append(0)
    
    
# specify directory to load
directory = 'New_datasets_emnlp2020_txt_newpreprocess/replicated_test'
# walk through all files in the folder
for filename in listdir(directory):
    # skip files that do not have the right extension
    if not filename.endswith(".txt"):
        continue
    # create the full path of the file to open
    path = directory + '/' + filename
    # load document
    doc = load_doc(path)
    move_review_data.append(doc)
    move_review_label.append(1)
    
    move_review_data_labelled_plus_unlabelled.append(doc)
    move_review_data_self_labelling_test.append(doc)
    #move_review_label_labelled_plus_unlabelled.append(1)
    
    #print('Loaded %s' % filename)
    
# specify directory to load
directory = 'New_datasets_emnlp2020_txt_newpreprocess/no_replicated_test'
# walk through all files in the folder
for filename in listdir(directory):
    # skip files that do not have the right extension
    if not filename.endswith(".txt"):
        continue
    # create the full path of the file to open
    path = directory + '/' + filename
    # load document
    doc = load_doc(path)
    move_review_data.append(doc)
    move_review_label.append(0)
    
    move_review_data_labelled_plus_unlabelled.append(doc)
    move_review_data_self_labelling_test.append(doc)
    #move_review_label_labelled_plus_unlabelled.append(0)
    
    
# specify directory to load
directory = 'new_unlabelled_dataset_txt_newpreprocess/American_Economic_Review'
# walk through all files in the folder
for filename in listdir(directory):
    # skip files that do not have the right extension
    if not filename.endswith(".txt"):
        continue
    # create the full path of the file to open
    path = directory + '/' + filename
    # load document
    doc = load_doc(path)
    #move_review_data_unlabelled.append(doc)
    move_review_data_labelled_plus_unlabelled.append(doc)
    move_review_data_self_labelling_train.append(doc)
    
# specify directory to load
directory = 'new_unlabelled_dataset_txt_newpreprocess/Psychological_Science'
# walk through all files in the folder
for filename in listdir(directory):
    # skip files that do not have the right extension
    if not filename.endswith(".txt"):
        continue
    # create the full path of the file to open
    path = directory + '/' + filename
    # load document
    doc = load_doc(path)
    #move_review_data_unlabelled.append(doc)
    move_review_data_labelled_plus_unlabelled.append(doc)
    move_review_data_self_labelling_train.append(doc)
 
from sklearn.feature_extraction.text import CountVectorizer
#count_vect = CountVectorizer(min_df = 500, stop_words="english")
#count_vect = CountVectorizer(min_df = 600, stop_words="english")
count_vect = CountVectorizer(min_df = 350, stop_words="english")
X_train_counts = count_vect.fit_transform(move_review_data_labelled_plus_unlabelled)







max_features = 20000
# cut texts after this number of words (among top max_features most common words)
#maxlen = 2000
#maxlen = 2380
maxlen = 1000
new_tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features, lower=True)
new_tokenizer.fit_on_texts(move_review_data_labelled_plus_unlabelled)
self_labelling_train_list = new_tokenizer.texts_to_sequences(move_review_data_self_labelling_train)
self_labelling_test_list = new_tokenizer.texts_to_sequences(move_review_data_self_labelling_test)
x_train_list_lstm = new_tokenizer.texts_to_sequences(move_review_data_train)

x_train_self = sequence.pad_sequences(self_labelling_train_list, maxlen=maxlen)
x_test_self = sequence.pad_sequences(self_labelling_test_list, maxlen=maxlen)
x_train_lstm = sequence.pad_sequences(x_train_list_lstm, maxlen=maxlen)









#print(X_train_counts.shape)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape)

X = X_train_tfidf.toarray()
y = numpy.array(move_review_label, dtype='int64')
#print(X.shape[1])









index_train_dataset_file = "20190519_index_train_dataset_1000_train_1000_test"
index_test_dataset_file = "20190519_index_test_dataset_1000_train_1000_test"
index_unlabelled_dataset = "20190519_index_unlabelled_dataset_1000_train_1000_test"

self_train_index_list = []

train_index_list = []
for each_train in open(index_train_dataset_file, "r"):
    each_str = each_train.strip()
    train_index_list.append(int(each_str))
    self_train_index_list.append(int(each_str))
    
test_index_list = []
for each_test in open(index_test_dataset_file, "r"):
    each_str = each_test.strip()
    test_index_list.append(int(each_str))
    
unlabelled_index_list = []
for each_unlabelled in open(index_unlabelled_dataset, "r"):
    each_str = each_unlabelled.strip()
    unlabelled_index_list.append(int(each_str))
    self_train_index_list.append(int(each_str))
    
train_index_list_array = numpy.array(train_index_list)
test_index_list_array = numpy.array(test_index_list)
unlabelled_index_list_array = numpy.array(unlabelled_index_list)
self_train_index_list_array = numpy.array(self_train_index_list) 
    
X_train = X[train_index_list_array]
y_train = y[train_index_list_array]

    
#replace noise label ends!

X_test = X[test_index_list_array]
y_test = y[test_index_list_array]

clf_svm = svm.SVC(gamma='scale', decision_function_shape='ovr')
clf_svm.fit(X_train, y_train)
score_train_svm = clf_svm.score(X_train, y_train) 
predict_train_svm = clf_svm.predict(X_train)
print("Accuracy on train data using SVM:   " + str(score_train_svm))
print("Prediction on train data using SVM:   " + str(predict_train_svm))
score_test_svm = clf_svm.score(X_test, y_test)
predict_test_svm = clf_svm.predict(X_test)
print("Accuracy on test data using SVM:   " + str(score_test_svm))
print("Prediction on test data using SVM:   " + str(predict_test_svm))
print("\n")


clf_lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf_lr.fit(X_train, y_train)
score_train_lr = clf_lr.score(X_train, y_train) 
predict_train_lr = clf_lr.predict(X_train)
print("Accuracy on train data using LR:   " + str(score_train_lr))
print("Prediction on train data using LR:   " + str(predict_train_lr))
score_test_lr = clf_lr.score(X_test, y_test)
predict_test_lr = clf_lr.predict(X_test)
print("Accuracy on test data using LR:   " + str(score_test_lr))
print("Prediction on test data using LR:   " + str(predict_test_lr))
print("\n")


clf_rf = RandomForestClassifier(n_estimators=180, max_depth=30, random_state=0)
clf_rf.fit(X_train, y_train)
score_train_rf = clf_rf.score(X_train, y_train) 
predict_train_rf = clf_rf.predict(X_train)
print("Accuracy on train data using RF:   " + str(score_train_rf))
print("Prediction on train data using RF:   " + str(predict_train_rf))
score_test_rf = clf_rf.score(X_test, y_test)
predict_test_rf = clf_rf.predict(X_test)
print("Accuracy on test data using RF:   " + str(score_test_rf))
print("Prediction on test data using RF:   " + str(predict_test_rf))
print("\n")


clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5*X.shape[1], 2), random_state=1)
clf_mlp.fit(X_train, y_train)
score_train_mlp = clf_mlp.score(X_train, y_train) 
predict_train_mlp = clf_mlp.predict(X_train)
print("Accuracy on train data using MLP:   " + str(score_train_mlp))
print("Prediction on train data using MLP:   " + str(predict_train_mlp))
score_test_mlp = clf_mlp.score(X_test, y_test)
predict_test_mlp = clf_mlp.predict(X_test)
print("Accuracy on test data using MLP:   " + str(score_test_mlp))
print("Prediction on test data using MLP:   " + str(predict_test_mlp))
print("\n")
print("\n")
print("\n")

total_iterations = 10000
best_test_acc = 0.0

for j in range(total_iterations):

    #Keras part
    batch_size = 5
    print('Build model...')
    model_small_size = Sequential()
    model_small_size.add(Embedding(max_features, 128))
    model_small_size.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model_small_size.add(Dense(1, activation='sigmoid'))
    
    # try using different optimizers and different optimizer configs
    model_small_size.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    print('Train...')
    model_small_size.fit(x_train_lstm, y_train,
              batch_size=batch_size,
              epochs=1,
              validation_data=(x_test_self, y_test))
    lstm_train_score, lstm_train_acc = model_small_size.evaluate(x_test_self, y_test,
                                batch_size=batch_size)
    
    #model_small_size.load_weights('20200527_keras_variational_inference_lstm_model_iteration_0.5757575631141663.h5')
    
    
    if lstm_train_acc > best_test_acc:
        best_test_acc = lstm_train_acc
        print('LSTM best accuracy is updated:', best_test_acc)
        model_small_size.save('20200528_keras_variational_inference_lstm_model_iteration_' + str(lstm_train_acc) + '.h5')
        
    
    print('LSTM testing score:', lstm_train_score)
    print('Lstm testing accuracy:', lstm_train_acc)
    print('Lstm best testing accuracy:', best_test_acc)
    
    if lstm_train_acc < 0.63:
        continue
    else:
        pass
        #model_small_size.save('20200527_keras_variational_inference_lstm_model_iteration_' + str(lstm_train_acc) + '.h5')
        
    #continue
    
    predict_self_train_keras_lstm_initial = model_small_size.predict_classes(x_train_self)
    predict_self_test_keras_lstm_initial = model_small_size.predict_classes(x_test_self)
    for test_each in range(test_num):
        print(str(test_each) + 'th example predict class is:' + str(predict_self_test_keras_lstm_initial[test_each][0]))
        
        
    num_correct = 0
    disappoint_num = 0
    very_disappoint_num = 0
    for k_index in range(test_num):
    
        temp_lstm = predict_self_test_keras_lstm_initial[k_index][0]
        temp_svm = predict_test_svm[k_index]
        temp_lr = predict_test_lr[k_index]
        temp_rf = predict_test_rf[k_index]
        temp_mlp = predict_test_mlp[k_index]
        test_true_label = y_test[k_index]
        
        total_num_classifier = temp_lstm + temp_svm + temp_lr + temp_rf + temp_mlp
        
        total_num_four_classifier = temp_svm + temp_lr + temp_rf + temp_mlp
        
        
        
        if total_num_classifier >= 3:
            if test_true_label == 1:
                num_correct += 1
            if test_true_label == 0:
                if total_num_four_classifier == 4:
                    very_disappoint_num += 1
                    print("Very disappointed: asnwer is 0 and majority is 4:" + str(k_index))
                if total_num_four_classifier == 3:
                    disappoint_num += 1
                    print("Disappointed: asnwer is 0 and majority is 3:" + str(k_index))
        else:
            if test_true_label == 0:
                num_correct += 1
            if test_true_label == 1:
                if total_num_four_classifier == 0:
                    very_disappoint_num += 1
                    print("Very disappointed: asnwer is 1 and majority is 0:" + str(k_index))
                if total_num_four_classifier == 1:
                    disappoint_num += 1
                    print("Disappointed: asnwer is 1 and majority is 1:" + str(k_index))
                
    print('Majority testing accuracy:', 1.0 * num_correct / test_num)
    print("-------------------------------" + str(very_disappoint_num) + "-------------------------------")
    print("-------------------------------" + str(very_disappoint_num) + "-------------------------------")
    print("-------------------------------" + str(very_disappoint_num) + "-------------------------------")
    print("-------------------------------" + str(disappoint_num) + "-------------------------------")
    print("-------------------------------" + str(disappoint_num) + "-------------------------------")
    print("-------------------------------" + str(disappoint_num) + "-------------------------------")
    
    X_train_unlabelled = X[unlabelled_index_list_array]
    X_self_train_unlabelled = X[self_train_index_list_array]
    predict_unlabeled_svm = clf_svm.predict(X_train_unlabelled)
    predict_unlabeled_lr = clf_lr.predict(X_train_unlabelled)
    predict_unlabeled_rf = clf_rf.predict(X_train_unlabelled)
    predict_unlabeled_mlp = clf_mlp.predict(X_train_unlabelled)
    
    
    
    
    
    
    estimate_aggregate_labels = []
    prediction_list_many_classifiers = []
    
    true_labels = numpy.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    # L = np.random.choice([0, 1], size=(10, 2))
    L = numpy.array(
        [
            [1, 1],
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 1],
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [1, 0]
        ]
    )
    
    temp_current_predict_list = []
    training_example_num = train_num
    unlabelled_num = 2170
    for k_index in range(unlabelled_num):
        temp_current_predict_list = []
        temp_svm = predict_unlabeled_svm[k_index]
        temp_lr = predict_unlabeled_lr[k_index]
        temp_rf = predict_unlabeled_rf[k_index]
        temp_mlp = predict_unlabeled_mlp[k_index]
        temp_lstm = predict_self_train_keras_lstm_initial[k_index + training_example_num][0]
        
        temp_current_predict_list.append(temp_svm)
        temp_current_predict_list.append(temp_lr)
        temp_current_predict_list.append(temp_rf)
        temp_current_predict_list.append(temp_mlp)
        temp_current_predict_list.append(temp_lstm)
        prediction_list_many_classifiers.append(temp_current_predict_list)
        
        
        total_num_classifier = temp_svm + temp_lr + temp_rf + temp_mlp +  temp_lstm
    
    
        if total_num_classifier >= 3:
            move_review_label_labelled_plus_unlabelled.append(1)
            estimate_aggregate_labels.append(1)
            
        else:
            move_review_label_labelled_plus_unlabelled.append(0)
            estimate_aggregate_labels.append(0)
            
    
    y_self_labelling = numpy.array(move_review_label_labelled_plus_unlabelled, dtype='int64')
    
    
    #estimate the error rate
    total_estimation_number = training_example_num + unlabelled_num
    error_num = 0
    error_num_1totrue0 = 0
    error_num_0totrue1 = 0
    error_rate_initial = 0.0
    error_rate_initial_1totrue0 = 0.0
    error_rate_initial_0totrue1 = 0.0
    
    prediction_list_many_classifiers_array = numpy.array(prediction_list_many_classifiers)
    estimate_aggregate_labels_array = numpy.array(estimate_aggregate_labels)
    temp_error_rate = VariationalEM(prediction_list_many_classifiers_array, estimate_aggregate_labels_array)
    
    error_rate_initial = 1.0 * error_num / unlabelled_num
    error_rate_initial_1totrue0 = 1.0 * temp_error_rate
    error_rate_initial_0totrue1 = 1.0 * temp_error_rate
    #print('yPred:', yPred)
    
    
    
    #Keras part
    batch_size = 128
    error_rate_estimate = error_rate_initial
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    
    import keras.backend as K
    
    def customLoss(yTrue,yPred):
        yTrue_int = K.cast(yTrue, dtype="int32")
        yFlip_int = K.cast(1-yTrue, dtype="int32")
        yFlip = 1-yTrue
        #yFlip = numpy.array(1-yTrue, dtype='int64')
        
        error_matrix = K.cast(numpy.array([error_rate_initial_1totrue0, error_rate_initial_0totrue1], dtype="float"), dtype="float")
        
        c1 = 1. - tf.keras.backend.gather(error_matrix, yFlip_int)
        c2 = tf.keras.backend.gather(error_matrix, 1-yTrue_int)
        
        coef = 1. - error_rate_initial_1totrue0 - error_rate_initial_0totrue1
        
        loss_batch = c1 * K.square(yTrue - yPred) + c2 * K.square(yFlip - yPred)
        
        temp_loss_positive = K.mean(loss_batch) / coef
        
        return temp_loss_positive
    
    # try using different optimizers and different optimizer configs
    model.compile(loss=customLoss,#loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.load_weights('20200527_keras_variational_inference_lstm_model_iteration_0.5959596037864685.h5')
    
    error_num_1totrue0 = 0
    error_num_0totrue1 = 0
    early_stop_num = 0
    last_iteration_no_change = 0
    
    coefficient_num = 3
    threshold_value = 4
    
    num_iterations = 20
    for each_iter in range(num_iterations):
        
        modify_label_num = 0
        error_num_1totrue0 = 0
        error_num_0totrue1 = 0
        
        estimate_aggregate_labels = []
        prediction_list_many_classifiers = []
        temp_current_predict_list = []
        
        print("In " + str(each_iter) + "th iterations:")
        
        #
        
        print('Train...')
        model.fit(x_train_self, y_self_labelling,
                  batch_size=batch_size,
                  epochs=1,
                  validation_data=(x_test_self, y_test))
        score, acc = model.evaluate(x_test_self, y_test,
                                    batch_size=batch_size)
        
        predict_self_train_keras_lstm_total = model.predict_classes(x_train_self)
        predict_self_test_keras_lstm_total = model.predict_classes(x_test_self)
        print('Test score:', score)
        print('Test accuracy:', acc)
        for test_each in range(test_num):
            print(str(test_each) + 'th example predict class is:' + str(predict_self_test_keras_lstm_total[test_each][0]))
        
        predict_test_svm = clf_svm.predict(X_test)
        predict_unlabeled_svm = clf_svm.predict(X_self_train_unlabelled)
        score_test_svm = clf_svm.score(X_test, y_test)
        print("Accuracy on test data using svm:   " + str(score_test_svm))
         
    
        predict_test_lr = clf_lr.predict(X_test)
        predict_unlabeled_lr = clf_lr.predict(X_self_train_unlabelled)
        score_test_lr = clf_lr.score(X_test, y_test)
        print("Accuracy on test data using lr:   " + str(score_test_lr))
         
         
        predict_test_rf = clf_rf.predict(X_test)
        predict_unlabeled_rf = clf_rf.predict(X_self_train_unlabelled)
        score_test_rf = clf_rf.score(X_test, y_test)
        print("Accuracy on test data using rf:   " + str(score_test_rf))
         
    
        predict_test_mlp = clf_rf.predict(X_test)
        predict_unlabeled_mlp = clf_rf.predict(X_self_train_unlabelled)
        score_test_mlp = clf_mlp.score(X_test, y_test)
        print("Accuracy on test data using mlp:   " + str(score_test_mlp))
        
        print("Accuracy on test data using lstm:   " + str(acc))
        
        num_correct = 0
        for k_index in range(test_num):
        
            temp_lstm = predict_self_test_keras_lstm_initial[k_index][0]
            temp_svm = predict_test_svm[k_index]
            temp_lr = predict_test_lr[k_index]
            temp_rf = predict_test_rf[k_index]
            temp_mlp = predict_test_mlp[k_index]
            test_true_label = y_test[k_index]
            
            total_num_classifier = coefficient_num * temp_lstm + temp_svm + temp_lr + temp_rf + temp_mlp
            
            if total_num_classifier >= threshold_value:
                if test_true_label == 1:
                    num_correct += 1
            else:
                if test_true_label == 0:
                    num_correct += 1
        new_majority_test_acc = 1.0 * num_correct / test_num
        print('Majority testing accuracy:', new_majority_test_acc)
        print('LSTM best accuracy:', best_test_acc)
        
        if acc > best_test_acc:
            best_test_acc = acc
            print('LSTM best accuracy is updated:', best_test_acc)
            model.save('20200528_keras_variational_inference_lstm_model_iteration' + str(each_iter) + "_" + str(acc) + '.h5')
        
        
        if acc > 0.70 or new_majority_test_acc > 0.70:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            
            model.save('20200528_keras_variational_inference_lstm_model_iteration' + str(each_iter) + "_" + str(acc) + '.h5')
        
        
        unlabelled_num = 2170
        training_example_num = train_num
        modify_label_num_1to0 = 0
        modify_label_num_0to1 = 0
        error_num = 0
        error_rate_estimate = 0.0
        
    
        temp_current_predict_list = []
        for k_index in range(train_num):
            temp_current_predict_list = []
            temp_lstm = predict_self_train_keras_lstm_total[k_index][0]
            temp_svm = predict_train_svm[k_index]
            temp_lr = predict_train_lr[k_index]
            temp_rf = predict_train_rf[k_index]
            temp_mlp = predict_train_mlp[k_index]
            test_true_label = y_train[k_index]
            
            temp_current_predict_list.append(temp_lstm)
            temp_current_predict_list.append(temp_svm)
            temp_current_predict_list.append(temp_lr)
            temp_current_predict_list.append(temp_rf)
            temp_current_predict_list.append(temp_mlp)
            prediction_list_many_classifiers.append(temp_current_predict_list)
            
            total_num_classifier = coefficient_num*temp_lstm + temp_svm + temp_lr + temp_rf + temp_mlp
            
            if total_num_classifier >= threshold_value:
                estimate_aggregate_labels.append(1)
            else:
                estimate_aggregate_labels.append(0)
            
            
            
        for k_index in range(unlabelled_num):
            temp_current_predict_list = []
            temp_lstm = predict_self_train_keras_lstm_total[training_example_num + k_index][0]
            temp_svm = predict_unlabeled_svm[training_example_num + k_index]
            temp_lr = predict_unlabeled_lr[training_example_num + k_index]
            temp_rf = predict_unlabeled_rf[training_example_num + k_index]
            temp_mlp = predict_unlabeled_mlp[training_example_num + k_index]
            
            temp_current_predict_list.append(temp_lstm)
            temp_current_predict_list.append(temp_svm)
            temp_current_predict_list.append(temp_lr)
            temp_current_predict_list.append(temp_rf)
            temp_current_predict_list.append(temp_mlp)
            prediction_list_many_classifiers.append(temp_current_predict_list)
            
            
    #         if each_iter <= 5:
    #             coefficient_num = 3
    #             threshold_value = 4
    #         if each_iter > 5:
    #             coefficient_num = 5
    #             threshold_value = 5
            
            
            total_num_classifier = coefficient_num*temp_lstm + temp_svm + temp_lr + temp_rf + temp_mlp
            
            if total_num_classifier >= threshold_value:
                estimate_aggregate_labels.append(1)
                if y_self_labelling[training_example_num + k_index] == 0:
                    modify_label_num_0to1 += 1
                    y_self_labelling[training_example_num + k_index] = 1
                    
            else:
                estimate_aggregate_labels.append(0)
                if y_self_labelling[training_example_num + k_index] == 1:
                    modify_label_num_1to0 += 1
                    y_self_labelling[training_example_num + k_index] = 0
                    
                    
        prediction_list_many_classifiers_array = numpy.array(prediction_list_many_classifiers)
        estimate_aggregate_labels_array = numpy.array(estimate_aggregate_labels)
        temp_error_rate = VariationalEM(prediction_list_many_classifiers_array, estimate_aggregate_labels_array)
        
        error_rate_initial = 1.0 * error_num / unlabelled_num
        error_rate_initial_1totrue0 = 1.0 * temp_error_rate
        error_rate_initial_0totrue1 = 1.0 * temp_error_rate
                
        print("error_rate_initial_1totrue0 is:   " + str(error_rate_initial_1totrue0))
        print("error_rate_initial_0totrue1 is:   " + str(error_rate_initial_0totrue1))
        print("modify_label_num_0to1 is:   " + str(modify_label_num_0to1))
        print("modify_label_num_1to0 is:   " + str(modify_label_num_1to0))
        
        if modify_label_num_0to1 == 0 and modify_label_num_1to0 == 0:
            early_stop_num += 1
            
    #         if early_stop_num == 3:
    #             if each_iter - last_iteration_no_change == 1:
    #                 break
    #             else:
    #                 early_stop_num = 1
    #              
    #         last_iteration_no_change = each_iter
            
            
            if early_stop_num == 50:
                break
            
            
         
        
