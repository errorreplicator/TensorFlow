import os
import cv2
import random
import numpy as np
import keras

path = '../Data/'
catalogs = ['Dog','Cat']
images = []
input_size = 1000

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

        image_resize = cv2.resize(grey,(60,60))
        images.append([image_resize,class_num])
        if index >= input_size: break

random.shuffle(images)
# images = np.asarray(images)
X_train = []
y_train = []

for x,y in images:
    X_train.append(x)
    y_train.append(y)
X_train = np.array(X_train)
X_train = X_train.reshape(input_size*2,60,60,1)
# print()

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(256,(3,3),input_shape=(60,60,1)))
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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5)