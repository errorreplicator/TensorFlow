import os
import cv2
import random
import numpy as np
# import keras
import matplotlib.pyplot as plt

path = '../Data/'
# catalogs = ['Dog','Cat']
catalogs = ['test']
images = []
input_size = 1000
resolution = 100

for x in catalogs:
    curr_catal = os.path.join(path,x)
    index = 0
    class_num = catalogs.index(x)
    for file in os.listdir(curr_catal):
        index += 1
        image = cv2.imread(os.path.join(curr_catal,file))

        try:
            grey = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        except Exception as e:
            pass

        image_resize = cv2.resize(grey,(resolution,resolution))
        images.append([image_resize,class_num])
        if index >= input_size: break

# print(len(images))
random.shuffle(images)
images = np.asarray(images)
# print(images[0])


X_train = []
y_train = []

for x,y in images:
    X_train.append(x)
    y_train.append(y)

# print(y_train[0])
# plt.imshow(images[0][0],cmap='gray' )
# plt.show()
#
# print(y_train[1])
# plt.imshow(images[1][0],cmap='gray' )
# plt.show()

X_train = np.array(X_train)
X_train = X_train.reshape(input_size*2,resolution,resolution,1)

    # model = keras.models.Sequential()
    #
    # model.add(keras.layers.Conv2D(128,(3,3),input_shape=(resolution,resolution,1)))
    # model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    #
    # model.add(keras.layers.Conv2D(128,(3,3)))
    # model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    #
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(32))
    # model.add(keras.layers.Activation('relu'))
    #
    # model.add(keras.layers.Dense(1))
    # model.add(keras.layers.Activation('sigmoid'))
    # #
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(X_train,y_train,epochs=5)



# from keras.models import Sequential
# from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
#
# model = Sequential()
#
# model.add(Conv2D(128, (3, 3), input_shape=(resolution, resolution,1)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(128, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# model.fit(X_train, y_train, batch_size=32, epochs=4, validation_split=0.3)