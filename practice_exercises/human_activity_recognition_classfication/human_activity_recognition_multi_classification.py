import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from scipy import stats
from scipy.stats import norm
from scipy import stats
from scipy.stats import norm
from numpy import genfromtxt


X_train=pd.read_csv("data/train/X_train.txt",delim_whitespace=True, header=None)
y_train=pd.read_csv("data/train/y_train.txt",delim_whitespace=True, header=None)
X_test=pd.read_csv("data/test/X_test.txt",delim_whitespace=True, header=None)
y_test=pd.read_csv("data/test/y_test.txt",delim_whitespace=True, header=None)


from sklearn.model_selection import train_test_split
X_train, X_test1, y_train, y_test1= train_test_split(X_train, y_train, train_size=0.2)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.preprocessing import LabelEncoder
encoder =  LabelEncoder()
y1 = encoder.fit_transform(y_train)

Y = pd.get_dummies(y1).values


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()


classifier.add(Dense(600, input_dim=561, activation='relu'))
classifier.add(Dense(600,  activation='relu'))
classifier.add(Dense(600,  activation='relu'))
classifier.add(Dense(6, activation='softmax'))


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, Y, batch_size = 32, epochs = 100)


ypred=classifier.predict(X_test)


import numpy as np
ypred_max=np.argmax(ypred, axis=1)
ypred_max=ypred_max+1
y_test_max=np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,ypred_max))
print(confusion_matrix(y_test,ypred_max))


from sklearn.metrics import accuracy_score
accuracy_score(ypred_max, y_test)