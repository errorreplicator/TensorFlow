import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pandas as pd
import columnize

np.set_printoptions(linewidth=102)


DATADIR = "./Data/"

# CATEGORIES = ["Dog", "Cat"]
CATEGORIES = ["test"]

IMG_SIZE = 50
input_size = 2000
training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        index = 0

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            index+=1
            # print(os.path.join(path,img))
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            if index>=input_size: break

create_training_data()
# print(training_data)
# print(training_data[0][0][0])
# print(type(training_data[0][0][0]))
# print(columnize.columnize(training_data[0][0][0]),30)
# plt.imshow(training_data[0][0]) #Stage ONE - lets do it step by step
# plt.show()

# print(len(training_data))



import random
random.seed(3)
random.shuffle(training_data)
print('shuffled table \n',training_data[0][0][0])
# print(training_data[0][0][1])
#
# # for sample in training_data[:10]:
# #     print(sample[1])
#
X = []
y = []
#
for features,label in training_data:
    X.append(features)
    y.append(label)
print('after X created \n',X[0][0])
# # print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
#
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)# !!!!!!!!!!!!!!!!!!!!!################%CHECK this reshape with mine reshape !!!!!!!!!!!!@@@@@@@@@@@@%%%%%%%%%%%%%@@@@@@@@@@!!!!!!!
print(X.shape)
# print(y)
print('reshape \n',X[0][0])



# # import tensorflow as tf
# # from tensorflow.keras.datasets import cifar10
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
#
# X = X/255.0
#
# model = Sequential()
#
# model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(256, (3, 3)))
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
# model.fit(X, y, batch_size=32, epochs=4, validation_split=0.3)