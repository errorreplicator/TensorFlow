import Recur.data_proc as ds
import tensorflow as tf
from tensorflow.contrib.keras import layers,models
from tensorflow.contrib.keras import callbacks
# from tensorflow.contrib import cudnn_rnn
# from tensorflow.contrib.cudnn_rnn import CudnnLSTM
# from tensorflow.contrib.keras import layers

X_train, y_train,X_valid, y_valid = ds.time_data()

print(f"train data: {len(X_train)} validation: {len(X_valid)}")
print(f"Dont buys: {y_train.count(0)}, buys: {y_train.count(1)}")
print(f"VALIDATION Dont buys: {y_valid.count(0)}, buys: {y_valid.count(1)}")

# tf.keras.layers.CuDNNLSTM


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.CuDNNLSTM(128,input_shape=(X_train.shape[1:]),return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())

model.add(tf.keras.layers.CuDNNLSTM(128,return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())

model.add(tf.keras.layers.CuDNNLSTM(128))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())

model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(2,activation='softmax'))

opt = tf.contrib.keras.optimizers.Adam(lr=0.001,decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    batch_size=30,
    epochs=5,
    validation_data=(X_valid,y_valid)
)
score = model.evaluate(X_valid,y_valid)

print(f'Test loss:{score[0]}')
print(f'Test accuracy:{score[1]}')


