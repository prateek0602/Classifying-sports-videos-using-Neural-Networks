#imports
import keras
import numpy as np
np.random.seed(420)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model,Model
import cv2
import matplotlib.pyplot as plt

X=np.load('X_train10.npy')
Y=np.load('Y_train10.npy')
X_test=np.load('X_validation10.npy')
Y_test=np.load('Y_validation10.npy')

X=X.reshape(132,40,512)

from keras.layers.recurrent import LSTM
model = Sequential()
model.add(LSTM(512, return_sequences=False,input_shape=(40,512)))
#model.add(LSTM(256))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
#model.add(Dense(64, activation='relu'))
model.add(Dense(9, activation='softmax'))

X_test=X_test.reshape(20,40,512)

model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

model.fit(X,Y,batch_size=1,validation_data=(X_test, Y_test),verbose=1,epochs=60)