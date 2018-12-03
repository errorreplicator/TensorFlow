import cv2
import os
import random

def process(d_type, flag='all',f_list = []):
    # FOLDERS = ['ship','truck']
    PATH = f'C:/Dataset/cifar10/{d_type}/'
    FOLDERS = []
    images = []
    if flag == 'all':
        for x in os.listdir(PATH):
            FOLDERS.append(x)
    else:
        FOLDERS = f_list

    for x in FOLDERS:
        for file in os.listdir(f'{PATH}/{x}'): #5k in each train folder
            img = cv2.imread(os.path.join(f'{PATH}/{x}',file))
            img_grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            images.append([img_grey,FOLDERS.index(x)])



    random.shuffle(images)

    X_train = []
    y_train = []
    if d_type == 'train':
        for x,y in images:
            X_train.append(x)
            y_train.append(y)

        return (X_train,y_train)
    else:
        X_test = []
        y_test = []

        for x, y in images:
            X_test.append(x)
            y_test.append(y)

        return (X_test, y_test)