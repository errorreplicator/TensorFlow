# zmienna1  =[1,2]
# zmienna2 = [256,512]
# zmienna3 = [0,1]
#
# for zm3 in zmienna3:
#     for zm2 in zmienna2:
#         for zm1 in zmienna1:
#             print(f'zm1-{zm1},zm2-{zm2},zm3-{zm3}')

import time
import pandas as pd
x = 1535214540
print(time.strftime('%Y-%m-%d %H:%M:%S', ))
print(pd.to_datetime(x,unit='s'))
