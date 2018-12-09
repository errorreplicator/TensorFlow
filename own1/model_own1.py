# import own1.preproc_data as ow
import own1.preprocess_lego as ow
import tensorflow as tf
import numpy as np
oop = tf.keras.optimizers
from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler

training_list = ['daisy','rose']

X_train, y_train = ow.process(d_type='train',f_list=training_list,flag='no_all')
X_test, y_test = ow.process(d_type='validate',flag='no_all',f_list=training_list)

X_train = np.array(X_train)
X_test = np.array(X_test)
shape = X_train.shape[1]
X_train = X_train.reshape(len(X_train),shape,shape,1)
X_test = X_test.reshape(len(X_test),shape,shape,1)
X_train = X_train/255.0 #Normalization
X_test = X_test/255.0

import pickle

pikcling = open('C:/Dataset/pickles/daisy_rose.pkl','wb')
pickle.dump([X_train,y_train,X_test,y_test],file=pikcling)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu',input_shape=(shape,shape,1)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

optimize = oop.Adam(lr=0.001)

model.compile(optimizer=optimize, loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=5,batch_size=10,verbose=1)


# loss: 0.5509 - acc: 0.7150 - val_loss: 7.7375 - val_acc: 0.5188

# why pool_size (3,3) cause dimension error
# see images clasified incorrectly
# how to tune in hyperparamiters - how to find best ones
# make the network work for all 10 classes clasification
# unsupervize learning of those classes ?

# rmsprop - loss: 0.2188 - acc: 0.9120
# oop.Adam(lr=0.001) - loss: 0.1974 - acc: 0.9240
# oop.Adam(lr=0.01) - loss: 0.3469 - acc: 0.8498