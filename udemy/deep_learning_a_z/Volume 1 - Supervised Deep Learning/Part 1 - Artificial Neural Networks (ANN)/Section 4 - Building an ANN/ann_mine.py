# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Churn_Modelling.csv")

X=dataset.iloc[:, 3:13].values
y=dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import keras


from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(6, kernel_initializer='uniform', input_dim=11, activation='relu'))

classifier.add(Dense(6, activation='relu',  kernel_initializer='uniform'))

classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

ypred=classifier.predict(X_test)