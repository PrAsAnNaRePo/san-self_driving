import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.losses
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense
from random import shuffle

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D


#phy = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(phy[0], True)

data1 = np.load(r'D:\fucking code\datasets\san-auto\CAR_160x120_0.npy', allow_pickle=True)
data2 = np.load(r'D:\fucking code\datasets\san-auto\CAR_160x120_10.npy', allow_pickle=True)
data3 = np.load(r'D:\fucking code\datasets\san-auto\CAR_160x120_12.npy', allow_pickle=True)
data4 = np.load(r'D:\fucking code\datasets\san-auto\CAR_160x120_16.npy', allow_pickle=True)
data5 = np.load(r'D:\fucking code\datasets\san-auto\CAR_160x120_24.npy', allow_pickle=True)
data6 = np.load(r'D:\fucking code\datasets\san-auto\CAR_160x120_30.npy', allow_pickle=True)

data = []
data.extend(data1)
data.extend(data2)
data.extend(data3)
data.extend(data4)
data.extend(data5)
data.extend(data6)

forwards = []
lefts = []
rights = []
back = []

for im, lbl in data:
    if lbl[0] == 1:
        forwards.append([im, [1,0,0,0]])
    if lbl[1] == 1:
        back.append([im, [0,0,0,1]])
    if lbl[2] == 1:
        lefts.append([im, [0,1,0,0]])
    if lbl[3] == 1:
        rights.append([im, [0,0,1,0]])

forwards = forwards[0:len(back)]
lefts = lefts[0:len(back)]
rights = rights[0:len(back)]

tot_data = forwards + lefts + rights
shuffle(tot_data)

train_data = tot_data[0:int(len(tot_data) * 0.8)]
test_data = tot_data[int(len(tot_data) * 0.8):]

x_train = np.array([i[0] for i in train_data]).reshape(len(train_data), 120, 160, 1)
x = []
for i in x_train:
    x.append(cv2.merge([i, i, i]))
x_train = np.array(x)
print(x_train.shape)

x_test = np.array([i[0] for i in test_data]).reshape(len(test_data), 120, 160, 1)
x = []
for i in x_test:
    x.append(cv2.merge([i, i, i]))
x_test = np.array(x)
print(x_test.shape)
y_train = np.array([i[1] for i in train_data])
y_test = np.array([i[1] for i in test_data])

base_model = Xception(weights=None, include_top=False, input_shape=x_train.shape[1:])
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# putput layer
predictions = Dense(4, activation='softmax')(x)
# model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'],
              )

model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), batch_size=12)

models.save_model(model, 'MOD-v2')
