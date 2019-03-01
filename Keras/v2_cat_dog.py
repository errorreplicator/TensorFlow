# import tensorflow as tf
# from tensorflow.contrib import keras
import cv2
import os
import matplotlib.pyplot as plt

path = 'C:\Dataset\img'
catalogs = ['Dog','Cat']
resolution = 50
catDog_list = []

for folder in catalogs:
    index = 0
    fol_path = os.path.join(path,folder)
    print(fol_path)
    for file in os.listdir(fol_path):
        # print(fol_path,'\\',file,sep='')
        image = cv2.imread(os.path.join(fol_path,file))
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_resize = cv2.resize(grey, (resolution, resolution))
        plt.imshow(image_resize)
        plt.show()
        index+=1
        if index>3: break
