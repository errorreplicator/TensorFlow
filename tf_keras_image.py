import os
import cv2
import random
import numpy as np
import keras
import tensorflow as tf
from tensorflow.contrib import keras



import timer

np.set_printoptions(linewidth=102)
path = '../Data/'
catalogs = ['Dog','Cat']

images = []
input_size = 5000
resolution = 50

for x in catalogs:
    curr_catal = os.path.join(path,x)
    index = 0
    class_num = catalogs.index(x)
    for file in os.listdir(curr_catal):
        index += 1

        try:
            image = cv2.imread(os.path.join(curr_catal, file))#,cv2.IMREAD_GRAYSCALE)
            grey = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            image_resize = cv2.resize(grey, (resolution, resolution))
            images.append([image_resize, class_num])
        except Exception as e:
            pass

        if index >= input_size: break

random.shuffle(images)

X_train = []
y_train = []

for x,y in images:
    X_train.append(x)
    y_train.append(y)

X_train = np.array(X_train)
X_train = X_train.reshape(len(images),resolution,resolution,1)

X_train = X_train/255.0 #### Normalization

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(256,(3,3),input_shape=(resolution,resolution,1)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(256,(3,3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))
#
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5,batch_size=50,verbose=1)

# Other ways of normalization ??? line 73
# Why sigmoid and why 1 output instead of 2 [0,1] cat and dog (becuse binary crossentropy??)
# Tune in model
# go through conv and MAxPool description and understanc
# is it working - cv2.cvtColor

# PC CPU - 256x256x64x1 - input_size = 5000 - time: 28:55:0232
# PC GPU - 256x256x64x1 - input_size = 5000 - time: 0:01:38:000
# Laptop GPU - 256x256x64x1 - input_size = 5000 - time: 0:06:33.311 - accuracy 0.7579
# Laptop GPU - 256x256x64x1 - input_size = 5000 | cv2.cvtColor - time: 0:06:10.712 - accuracy 0.7754
# Laptop GPU - 256x256x64x1 - input_size = 5000 | cv2.cvtColor - time: 0:05:46.000 - accuracy 0.7894
