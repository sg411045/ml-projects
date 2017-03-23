# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:58:49 2016

@author: s.gopalakrishnan
"""

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# baseline_model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    #optimizer - adam or sgd
    sgd = SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # compile
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

# fix seed for random generator
seed = 7
numpy.random.seed(seed)

# load the dataset
dataframe = pandas.read_csv("../datasets/iris.csv", delimiter="," , header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
y = dataset[:,4]

encoder = LabelEncoder()
encoder.fit(y)
encoder_y = encoder.transform(y)

# convert integers to dummy variables
dummy_y = np_utils.to_categorical(encoder_y)

# Split-out validation dataset
validation_size = 0.33

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=500, batch_size=5, verbose=0)

# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

history  = estimator.fit(X, dummy_y, validation_split=validation_size)

# plottings
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()