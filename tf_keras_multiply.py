import time
from tensorflow.contrib import keras
from tensorflow.contrib.keras import callbacks
import pickle
hm_nodes = [64,128,256]
hm_convs = [1,2,3]
hm_dense = [0,1,2]

resolution = 50
open_pickle = open('c:\Dataset\dogcat\X_train_y_train.pkl','rb')
X_train, y_train = pickle.load(open_pickle)

for  dense in hm_dense:
    for nodes in hm_nodes:
        for convs in hm_convs:

            NAME = f'dense-{dense}-nodes-{nodes}-convs-{convs}-{int(time.time())}'
            tensorboard = callbacks.TensorBoard(log_dir=f'C:/Board/{NAME}')

            model = keras.models.Sequential()

            model.add(keras.layers.Conv2D(nodes, (3, 3), input_shape=(resolution, resolution, 1)))
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

            for _ in range(convs):
                model.add(keras.layers.Conv2D(nodes, (3, 3)))
                model.add(keras.layers.Activation('relu'))
                model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

            model.add(keras.layers.Flatten())
            for _ in range(dense):
                model.add(keras.layers.Dense(nodes))
                model.add(keras.layers.Activation('relu'))

            model.add(keras.layers.Dense(1))
            model.add(keras.layers.Activation('sigmoid'))
            #
            model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=5, validation_split=0.3, batch_size=50, verbose=1,
                      callbacks=[tensorboard])