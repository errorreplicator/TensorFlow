import os
import cv2


a = 4
b = 12
c = 71

wynik = 0

if a>b and a>c:
    print(f'a:{a}')
else:
    if b>c:
        print(f'b:{b}')
    else:
        print(f'c:{c}')

# path = 'c:/Dataset/flowers/daisy/train/'
# index = 0
# for file in os.listdir(path):
#     print(file)
#     img = cv2.imread(os.path.join(path,file))
#     print(img.shape)
#     index+=1
#     if index == 3:
#         break
