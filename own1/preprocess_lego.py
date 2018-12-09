import cv2
import os
import random

def process(d_type, flag='all',f_list = []):
    # FOLDERS = ['ship','truck']
    PATH = f'C:/Dataset/flowers/{d_type}/'
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
            img_grey = cv2.resize(img_grey,(60,60))
            images.append([img_grey,FOLDERS.index(x)])



    random.shuffle(images)

    X_ = []
    y_ = []
    for x,y in images:
        X_.append(x)
        y_.append(y)

    return (X_,y_)