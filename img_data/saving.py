import pickle
from img_data.data_wr import read_data

path_train = 'C:/Dataset/flowers/train/'
X_train, y_train = read_data(path=path_train)

print(X_train.shape)

dump_path = '../Data/daisy_rose_1393_60_60.pkl'
labels = ['daisy','rose']
picklun = open(dump_path,'wb')
pickle.dump([X_train,y_train,labels],picklun)
picklun.close()