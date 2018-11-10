import os
import cv2
import random
import numpy as np
import keras

# import matplotlib.pyplot as plt
import tensorflow as ts
np.set_printoptions(linewidth=102)
import timer
path = '../Data/'
catalogs = ['Dog','Cat']
# catalogs = ['test']
images = []
input_size = 5000
resolution = 50

for x in catalogs:
    curr_catal = os.path.join(path,x)
    index = 0
    class_num = catalogs.index(x)
    for file in os.listdir(curr_catal):
        index += 1
        # print(os.path.join(curr_catal,file))


        try:
            image = cv2.imread(os.path.join(curr_catal, file),cv2.IMREAD_GRAYSCALE)
            image_resize = cv2.resize(image, (resolution, resolution))
            images.append([image_resize, class_num])
            # grey = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        except Exception as e:
            pass


        if index >= input_size: break


            # img = images[0][0][0]
            # print(img)
            # print(len(img))
            # # print(len(images))
random.seed(3)
random.shuffle(images)
# print(len(images))
# print('shuffled table\n',images[0][0][0])
# print(images[1][0][0])
# print(images[2][0][0])
            # images
# = np.asarray(images)
            # # print(images[0])
            #
            #
X_train = []
y_train = []

for x,y in images:
    X_train.append(x)
    y_train.append(y)
# print('after X created\n', X_train[0][0])
            # print(y_train[0])
            # plt.imshow(images[0][0],cmap='gray' )
            # plt.show()
            #
            # print(y_train[1])
            # plt.imshow(images[1][0],cmap='gray' )
            # plt.show()
# img = images[0][0] ###############################################
X_train = np.array(X_train)
X_train = X_train.reshape(len(images),resolution,resolution,1)
# print(X_train.shape)
# print(y_train)
# print('reshape\n',X_train[0][0])

X_train = X_train/255.0 #### WTF ?

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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5)

# Other ways of normalization ??? line 73
# Why sigmoid and why 1 output instead of 2 [0,1] cat and dog (becuse binary crossentropy??)
# Tune in model
# go through conv and MAxPool description and understanc

# CPU - 256x256x64x1 - input_size = 5000 - time: 28:55:0232