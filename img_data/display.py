import matplotlib.pyplot as plt
import numpy as np

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        #print(ims.shape)
#         if (ims.shape[-1] != 3):
#             ims = ims.transpose((1,2,0)) #oryginally was: ims = ims.transpose((0,2,3,1))
    #print(ims.shape)
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none',cmap='gray')
    plt.show()
