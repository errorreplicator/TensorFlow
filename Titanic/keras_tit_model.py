import tensorflow as tf
import Titanic.data_read_v2 as dr

X_train, y_train = dr.preprocessing()


INPUT_SHAPE = X_train.shape[1]
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=7,activation='relu',kernel_initializer='uniform',input_dim=INPUT_SHAPE))
model.add(tf.keras.layers.Dropout(rate=0.2))

model.add(tf.keras.layers.Dense(units=80,activation='relu',kernel_initializer='uniform'))
model.add(tf.keras.layers.Dropout(rate=0.2))

model.add(tf.keras.layers.Dense(units=60,activation='relu',kernel_initializer='uniform'))
model.add(tf.keras.layers.Dropout(rate=0.2))

model.add(tf.keras.layers.Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train,y_train,batch_size=16,epochs=50)

# model.evaluate()
#
# input_dim=7 - loss: -85.0179 - acc: 0.5948
# input_dim=8 loss: 0.4306 - acc: 0.8160 # reatures from Kaggle || https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook
# after normalization improvements loss: 0.4292 - acc: 0.8047
# after normalizing everything loss: 0.4490 - acc: 0.8058
# after dummy_vars - loss: 0.4324 - acc: 0.8126