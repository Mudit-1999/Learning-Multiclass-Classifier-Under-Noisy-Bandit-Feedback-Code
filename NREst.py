import tensorflow as tf 
from tensorflow import keras
from keras.layers import Input
from keras import backend as K
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import ResNet50,VGG16

from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.models import Model
from keras.layers import MaxPooling2D,AveragePooling2D,Dropout
from keras.layers.core import Dense, Activation, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adagrad
from keras.optimizers import SGD



import os
import sys
import copy
import random
import getopt
import numpy as np
import numba as nb
import numpy.linalg as LA
from six.moves import urllib
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn import preprocessing
from numba import jit,float32,int32 
from collections import defaultdict
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import train_test_split


# To add a dataclass for noise rate esimation 
# 1. Build a network model for your dataset (size of the last , input  and hidden layer)
# (you can choose any one of the below specified network)
# 2. create the load data function 


def crossentropy(y_true, y_pred):
    # this gives the same result as using keras.objective.crossentropy
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -K.sum(y_true * K.log(y_pred), axis=-1)

class KerasModel():

    def get_data(self):

        (X_train, y_train), (X_test, y_test) = self.load_data()

        idx_perm = np.random.RandomState(101).permutation(X_train.shape[0])
        X_train, y_train = X_train[idx_perm], y_train[idx_perm]

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # print('X_train shape:', X_train.shape)
        # print(X_train.shape[0], 'train samples')
        # print(X_test.shape[0], 'test samples')

        return X_train, X_test, y_train, y_test


    def compile(self, model, loss):

        if self.optimizer is None:
            ValueError()
        metrics = ['accuracy']
        model.compile(loss=crossentropy,optimizer=self.optimizer, metrics=metrics)
        # model.summary()
        self.model = model

    def load_model(self, file):
        # print('Loaded model from %s' % file)
        self.model.load_weights(file)


    def fit_model(self, model_file, X_train, Y_train, validation_split=None,validation_data=None):

        # cannot do both
        if validation_data is not None and validation_split is not None:
            return ValueError()

        callbacks = []
        monitor = 'val_loss'
        # monitor = 'val_acc'

        mc_callback = ModelCheckpoint(model_file, monitor=monitor,verbose=0, save_best_only=True)
        callbacks.append(mc_callback)


        history = self.model.fit(
                    X_train, Y_train, batch_size=self.num_batch,
                    epochs=self.epochs,
                    validation_split=validation_split,
                    validation_data=validation_data,
                    verbose=0, callbacks=callbacks)

        # use the model that reached the lowest loss at training time
        self.load_model(model_file)
        return history.history

    def evaluate_model(self, X, Y):
        score = self.model.evaluate(X, Y, batch_size=self.num_batch, verbose=0)
        # print('Test score:', score[0])
        # print('Test accuracy:', score[1])
        return score[1]

    def predict_proba(self, X):
        pred = self.model.predict(X, batch_size=self.num_batch, verbose=0)
        return pred


# Fashion MNIST
class FashionMNIST(KerasModel,data=None):
    def __init__(self, num_batch=32):
        self.num_batch = num_batch
        self.classes = 2
        self.epochs = 70
        self.optimizer = None
        self.data=data


    def load_data(self):
        X= np.array(self.data)
        X, predicted_label,y=X[:,0:100],X[:,-2],X[:,-1]
        normalized = preprocessing.normalize(X)
        predicted_label=predicted_label.reshape((-1,1))
        X=np.concatenate((normalized,predicted_label),axis=1)
        # print("Shape of Dataset concatenated with labels",X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        return (X_train, y_train), (X_test, y_test)

    def build_model(self, loss):
        input = Input(shape=(101,))
        x = Dense(32, kernel_initializer='he_normal')(input)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(2, kernel_initializer='he_normal')(x)
        output = Activation('softmax')(output)
        model = Model(inputs=input, outputs=output)
        self.compile(model, loss)


# SYN_SEP
class SynSepModel(KerasModel):
    def __init__(self, num_batch=32,data=None):
        self.num_batch = num_batch
        self.classes = 2
        self.epochs = 70
        self.optimizer = None
        self.data=data


    def load_data(self):
        X= np.array(self.data)
        X, predicted_label,y=X[:,0:400],X[:,-2],X[:,-1]
        normalized = preprocessing.normalize(X)
        predicted_label=predicted_label.reshape((-1,1))
        X=np.concatenate((normalized,predicted_label),axis=1)
        # print("Shape of Dataset concatenated with labels",X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        return (X_train, y_train), (X_test, y_test)

    def build_model(self, loss):
        input = Input(shape=(401,))
        x = Dense(48, kernel_initializer='he_normal')(input)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(48, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(2, kernel_initializer='he_normal')(x)
        output = Activation('softmax')(output)
        model = Model(inputs=input, outputs=output)
        self.compile(model, loss)

# MNIST
class MNISTModel(KerasModel):
    def __init__(self, num_batch=32,data=None):
        self.num_batch = num_batch
        self.classes = 2
        self.epochs = 70
        self.optimizer = None
        self.normalize=True
        self.data=data


    def load_data(self):
        X= np.array(self.data)
        X, predicted_label,y=X[:,0:64],X[:,-2],X[:,-1]
        normalized = preprocessing.normalize(X)
        predicted_label=predicted_label.reshape((-1,1))
        X=np.concatenate((normalized,predicted_label),axis=1)
        # print("Shape of Dataset concatenated with labels",X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        return (X_train, y_train), (X_test, y_test)
        
    def build_model(self, loss):
        input = Input(shape=(65,))
        x = Dense(128, kernel_initializer='he_normal')(input)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(2, kernel_initializer='he_normal')(x)
        output = Activation('softmax')(output)
        model = Model(inputs=input, outputs=output)
        self.compile(model, loss)



# USPS
class USPSModel(KerasModel):
    def __init__(self, num_batch=32,data=None):
        self.num_batch = num_batch
        self.classes = 2
        self.epochs = 20
        self.optimizer = None
        self.data=data

    def load_data(self):
        X= np.array(self.data)
        X, predicted_label,y=X[:,0:256],X[:,-2],X[:,-1]
        normalized = preprocessing.normalize(X)
        predicted_label=predicted_label.reshape((-1,1))
        X=np.concatenate((normalized,predicted_label),axis=1)
        # print("Shape of Dataset concatenated with labels",X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        return (X_train, y_train), (X_test, y_test)

    def build_model(self, loss):
        input = Input(shape=(257,))
        x = Dense(32, kernel_initializer='he_normal')(input)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(256, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(2, kernel_initializer='he_normal')(x)
        output = Activation('softmax')(output)
        model = Model(inputs=input, outputs=output)
        self.compile(model, loss)



# IRIS
class IRISModel(KerasModel):
    def __init__(self, num_batch=32,data=None):
        self.num_batch = num_batch
        self.classes = 2
        self.epochs = 70
        self.normalize = True
        self.optimizer = None
        self.data=data


    def load_data(self):
        X= np.array(self.data)
        X, predicted_label,y=X[:,0:4],X[:,-2],X[:,-1]
        normalized = preprocessing.normalize(X)
        predicted_label=predicted_label.reshape((-1,1))
        X=np.concatenate((normalized,predicted_label),axis=1)
        # print("Shape of Dataset concatenated with labels",X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        return (X_train, y_train), (X_test, y_test)

    def build_model(self, loss):
        input = Input(shape=(5,))
        x = Dense(32, kernel_initializer='he_normal')(input)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(2, kernel_initializer='he_normal')(x)
        output = Activation('softmax')(output)
        model = Model(inputs=input, outputs=output)
        self.compile(model, loss)

class NoiseEstimator():

    def __init__(self, classifier, row_normalize=True, alpha=0.0,outlier_filter=100,cliptozero=False, verbose=0):
        """classifier: an ALREADY TRAINED model. In the ideal case, classifier
        should be powerful enough to only make mistakes due to noise in feedback."""

        self.classifier = classifier
        self.row_normalize = row_normalize
        self.alpha = alpha
        self.outlier_filter = outlier_filter
        self.cliptozero = cliptozero
        self.verbose = verbose

    def fit(self, X):

        # number of classes
        c = self.classifier.classes
        T = np.empty((c, c))

        # predict probability on the fresh sample
        eta_corr = self.classifier.predict_proba(X)

        # find a 'perfect example' for each class
        for i in np.arange(c):
            eta_thresh = np.percentile(eta_corr[:, i],self.outlier_filter,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)

            for j in np.arange(c):
                T[i, j] = eta_corr[idx_best, j]

        self.T = T
        return self

    def predict(self):

        T = self.T
        c = self.classifier.classes

        if self.cliptozero:
            idx = np.array(T < 10 ** -6)
            T[idx] = 0.0

        if self.row_normalize:
            row_sums = T.sum(axis=1)
            T /= row_sums[:, np.newaxis]

        if self.verbose > 0:
            print(T)

        if self.alpha > 0.0:
            T = self.alpha * np.eye(c) + (1.0 - self.alpha) * T

        if self.verbose > 0:
            print(T)
            print(np.linalg.inv(T))

        return T



np.random.seed(1337)  # for reproducibility

trial=1

def build_file_name(loc, dataset,run):
    file = 'output/' + loc + dataset + '_' +str(run)
    return file


def train_and_evaluate(dataset, loss, run=0, num_batch=128,asymmetric=0,data=None):
    val_split = 0.1
    outlier_filter = 89
    if dataset == 'mnist':
        kerasModel = MNISTModel(num_batch=num_batch,data=data)
    elif dataset == 'iris':
        kerasModel = IRISModel(num_batch=num_batch,data=data)
    elif dataset == 'usps':
        kerasModel = USPSModel(num_batch=num_batch,data=data)
    elif dataset== 'fashionmnist':
        kerasModel=FashionMNIST(num_batch=num_batch,data=data)
    elif dataset == 'synsep':
        kerasModel = SynSepModel(num_batch=num_batch,data=data)
        outlier_filter=94
    else:
        ValueError('No dataset given.')
        sys.exit()

    # optimizer
    kerasModel.optimizer = Adagrad()

    # the data, shuffled and split between train and test sets
    X_train, X_test, y_train, y_test = kerasModel.get_data()

    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, kerasModel.classes)
    Y_test = to_categorical(y_test, kerasModel.classes)

    # keep track of the best model
    model_file = build_file_name('tmp_model/', dataset, run)
    kerasModel.build_model(loss)

    # fit the model
    # print("##################################3333",model_file)
    history = kerasModel.fit_model(model_file, X_train, Y_train,validation_split=val_split)



    # score = kerasModel.evaluate_model(X_test, Y_test)
    kerasModel.build_model('crossentropy')
    kerasModel.load_model(model_file)
    
    # Estimating Noise 
    est = NoiseEstimator(classifier=kerasModel, alpha=0.0,outlier_filter=outlier_filter)
    P_est = est.fit(X_train).predict()
    # print('Estimated Noise matrix :\n', P_est)
    

    return P_est