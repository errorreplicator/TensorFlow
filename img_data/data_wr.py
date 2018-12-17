import numpy as np
import os
import cv2
import random

def read_data(path,reso=60,flag='all',fol_list=[]):
    resolution = reso
    FOLDERS = []
    img_list = []
    # index=0
    if not fol_list: #if list is enpty
        for folder in os.listdir(path):
            FOLDERS.append(folder)
    else:
        FOLDERS = fol_list

    for index,folder in enumerate(FOLDERS):
        print(f'{index}: {folder}')
    for f in FOLDERS:
        index = FOLDERS.index(f)
        curr_folder = os.path.join(path,f)
        for file in os.listdir(curr_folder):
            img = os.path.join(curr_folder,file)
            img = cv2.imread(img)
            img_grey = cv2.cvtColor(cv2.resize(img,(reso,reso)),cv2.COLOR_RGB2GRAY)
            img_list.append([img_grey,index])
            # index+=1
            # if index>5:
            #     break

    random.shuffle(img_list)
    X_ = []
    y_ = []
    for x,y in img_list:
        X_.append(x)
        y_.append(y)

    X_ = np.array(X_)
    y_ = np.array(y_)
    return (X_,y_)

def get_all_incorrect(y_train,predict_class,predictions):
    classes = predict_class.reshape((-1,))
    diff_index = np.nonzero(classes != y_train)
    konvert = np.asarray(diff_index)
    diff_index = konvert[0][:]
    propabilities = []
    for x in diff_index:
        propabilities.append(predictions[x][0])
    return (diff_index, propabilities)

def get_all_correct(y_train,predict_class,predictions):
    classes = predict_class.reshape((-1,))
    diff_index = np.nonzero(classes == y_train)
    konvert = np.asarray(diff_index)
    diff_index = konvert[0][:]
    propabilities = []
    for x in diff_index:
        propabilities.append(predictions[x][0])
    return (diff_index, propabilities)