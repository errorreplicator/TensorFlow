import keras
import tensorflow as tf
import timer

minst = keras.datasets.mnist

(x_train,y_train), (x_test,y_test) = minst.load_data()
# print(len(x_train))
# print(len(x_train[0]))
# print(x_train[0])

x_train = keras.utils.normalize(x_train,axis=1)
x_test = keras.utils.normalize(x_test,axis=1)
# keras.utils.normalize()
model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=4)
# print(tf.VERSION)
# print(tf.keras.__version__)
# print(keras.__version__)

#step 1 -  784 neurons -  loss: 0.0397 - acc: 0.9870
#step 2 -  784 neurons - dropauts 0.25: loss: 0.0637 - acc: 0.9798
#step 3 -  128 neurons -
#step 4 -  128 neurons - dropauts 0.25: Elapsed time: 0:00:49.117571
#step 4a PC -  128 neurons - dropauts 0.25: Elapsed time: 0:00:22.729088