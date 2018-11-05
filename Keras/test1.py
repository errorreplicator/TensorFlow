import keras
import tensorflow as tf
import timer

minst = keras.datasets.mnist

(x_train,y_train), (x_test,y_test) = minst.load_data()
# print(len(x_train))
# print(len(x_train[0]))
# print(x_train[0])
img_rows, img_cols = 28,28
x_train = keras.utils.normalize(x_train,axis=1)
x_test = keras.utils.normalize(x_test,axis=1)
if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# keras.utils.normalize()
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
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
#step 5a - CONV lyer - 0:58:30.648199 :0 WOW time