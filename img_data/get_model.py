import tensorflow as tf

lay = tf.keras.layers

def model_bin(class_reso):
    model = tf.keras.models.Sequential()
    model.add(lay.Conv2D(360,kernel_size=(3,3),activation='relu',input_shape=(class_reso,class_reso,1)))
    model.add(lay.MaxPool2D(pool_size=(2,2)))
    model.add(lay.Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(lay.MaxPool2D(pool_size=(2,2)))
    model.add(lay.Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(lay.MaxPool2D(pool_size=(2,2)))
    model.add(lay.Flatten())
    model.add(lay.Dense(64,activation='relu'))
    model.add(lay.Dense(1,activation='sigmoid'))
    return model

