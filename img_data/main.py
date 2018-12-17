from img_data import display,data_wr, get_model
import tensorflow as tf
import pickle
############# SHOULD HELP make reproducable code#############################
# import random as rn
import numpy as np
np.set_printoptions(linewidth=700)
# import os
# os.environ['PYTHONHASHSEED'] = '0'
# np.random.seed(37)
# rn.seed(1254)
# tf.set_random_seed(89)
# from keras import backend as K
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(),config=session_conf)
# K.set_session(sess)
#################################################################################
path_train = 'C:/Dataset/flowers/train/'

# #######################LOAD DATA & TRAIN MODEL #############################
# X_train, y_train = data_wr.read_data(path=path_train)
# # print(X_train.shape)
pickle_open = open('../Data/daisy_rose_1393_60_60.pkl','rb')
X_train,y_train,labels = pickle.load(pickle_open)
print(labels)
img_reso = X_train.shape[1]
# display.plots([X_train[0]],titles=['check 1']) #checkmark
X_train = X_train.reshape(len(X_train),img_reso,img_reso,1)
X_train = X_train/255.0 # normalization - for improvement
# ############################################################################
# model = get_model.model_bin(X_train.shape[1])
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
# model.fit(X_train,y_train,epochs=5)
# model.save('../models/test1.mdl')
# model.save_weights('../models/test1_weights.wgs')
# ############################################################################

# ######################LOAD##################################################
model = tf.keras.models.Sequential()
model = tf.keras.models.load_model('../models/test1.mdl')
model.load_weights('../models/test1_weights.wgs')
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])
# ############################################################################

#################################Prediction###################################
prediction = model.predict(X_train)
predict_class = model.predict_classes(X_train)

index_bad , probs_bad   = data_wr.get_all_incorrect(y_train,predict_class,prediction)
index_good, probs_good = data_wr.get_all_correct(y_train,predict_class,prediction)

correct_img = []
incorrect_img = []

pickle_open = open('../Data/daisy_rose_1393_60_60.pkl','rb')
X_train,y_train,labels = pickle.load(pickle_open)
# display.plots([X_train[0]],titles=['check 2']) #checkmark

for x in index_good:
    correct_img.append(X_train[x])
for x in index_bad:
    incorrect_img.append(X_train[x])

correct_img = np.array(correct_img)
incorrect_img = np.array(incorrect_img)
display.plots(correct_img[:10],titles=probs_good[:10],rows=2)


# 1. how to display top correct and top incorrect ?

# 4. display [:10] correctly interpreted - DONE
# 5. display [:10] incorrectly interpreted - DONE
# 6. display 10 most uncertain <---------------------------------
# display.plots(X_train[:10],titles=y_train[:10])