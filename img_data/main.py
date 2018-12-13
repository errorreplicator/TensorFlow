from img_data import display,readin, get_model
import tensorflow as tf
path_train = 'C:/Dataset/flowers/train/'

X_train, y_train = readin.read_data(path=path_train)
print(X_train.shape)
img_reso = X_train.shape[1]
# img_number = X_train.shape[0]
X_train = X_train.reshape(len(X_train),img_reso,img_reso,1)
X_train = X_train/255.0 # normalization - for improvement
# model = get_model.model_bin(X_train.shape[1])
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])
# model.fit(X_train,y_train,epochs=5)
# model.save('../models/test1.mdl')
model = tf.keras.models.load_model('../models/test1.mdl')

print(X_train.shape)
print(y_train.shape)
print(y_train[10:20])
prediction = model.predict(X_train[10:20])
print(prediction)

# odkomentowac model i w jednym przebiegu uczenie i predict (bo po load model predictions sa same 0.5121 jakby model zgadywal)
# plus:
# 1.obrazy do tablicy
# 2. tablica shuffle
# 3. train
# 4. display [:10] correctly interpreted
# 5. display [:10] incorrectly interpreted
# 6. display 10 most uncertain
# display.plots(X_train[:10],titles=y_train[:10])