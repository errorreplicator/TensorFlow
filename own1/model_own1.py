import own1.preproc_data as ow
import tensorflow as tf
import numpy as np

X_train, y_train = ow.process(d_type='train',f_list=['ship','truck'],flag='no_all')
X_test, y_test = ow.process(d_type='test',flag='no_all',f_list=['ship','truck'])

X_train = np.array(X_train)
X_train = X_train.reshape(len(X_train),32,32,1)

X_train = X_train/255.0 #Normalization

print(len(X_train))
print(X_train.shape)
print(len(y_train))
# print(y_train.shape)
print('-'*10)
print(len(X_test))
print(len(y_test))
# print(y_test.shape)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu',input_shape=(32,32,1)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

# model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
# model.fit(X_train,y_train,epochs=5,batch_size=50)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5,batch_size=50,verbose=1)



# why pool_size (3,3) cause dimension error
# see images clasified incorrectly
# how to tune in hyperparamiters - how to find best ones
# make the network work for all 10 classes clasification
# unsupervize learning of those classes ?