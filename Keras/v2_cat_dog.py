# import tensorflow as tf
from tensorflow.contrib import keras
import cv2
import os
import numpy as np

import matplotlib.pyplot as plt

km = keras.models
kl = keras.layers

path = 'C:\Dataset\img'
catalogs = ['Dog','Cat']
resolution = 50
input_size = 3000
epoch = 10
catDog_list = []

for folder in catalogs:
    index = 0
    fol_path = os.path.join(path,folder)
    # print(fol_path)
    for file in os.listdir(fol_path):
        try:
            image = cv2.imread(os.path.join(fol_path, file))  # ,cv2.IMREAD_GRAYSCALE)
            grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            print(f'Error at file index {index} with path:', fol_path, '\\', file, sep='')
            pass

        image_resize = cv2.resize(grey, (resolution, resolution))
        # image_normal = tf.keras.utils.normalize(image_resize,axis=1) #axis 0 - 0.69
        catDog_list.append([image_resize, catalogs.index(folder)])
        index+=1
        if index > input_size: break
X_train = []
y_train = []

for x,y in catDog_list:
    X_train.append(x)
    y_train.append(y)

X = np.array(X_train).reshape(-1,resolution,resolution,1)
X=X/255.0

model = km.Sequential()

model.add(kl.Conv2D(128,(3,3),input_shape=X.shape[1:]))
model.add(kl.Activation('relu'))
model.add(kl.MaxPool2D(pool_size=(2,2)))

model.add(kl.Conv2D(256,(3,3)))
model.add(kl.Activation('relu'))
model.add(kl.MaxPool2D(2,2))

model.add(kl.Flatten())

model.add(kl.Dense(64))
model.add(kl.Activation('relu'))
model.add(kl.Dense(1))
model.add(kl.Activation('sigmoid'))

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X,y_train,epochs=epoch,batch_size=50)


