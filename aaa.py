import time
from tensorflow.contrib.keras import callbacks

NAME = f'cnn256x256-{int(time.time())}'
tensorboard = callbacks.TensorBoard(log_dir=f'C:/Board/{NAME}')

pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)

model.fit(X_train,y_train,epochs=5, validation_split=0.3 ,batch_size=50,verbose=1,callbacks=[tensorboard])

np.set_printoptions(linewidth=102)

###################PICKLE#############################################################
dump_pickle = open('./Data/one_hot_table.pkl','wb')
pickle.dump([x_train,y_train],dump_pickle)
dump_pickle.close()


#######################################################################################