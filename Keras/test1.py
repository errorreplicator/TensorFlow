import os
import cv2
import random
import numpy as np
import keras
import matplotlib.pyplot as plt

path = '../Data/'
catalogs = ['Dog','Cat']
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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
X = np.array(X_train).reshape(-1, resolution, resolution, 1)
X = X/255.0

model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y_train, batch_size=32, epochs=4)