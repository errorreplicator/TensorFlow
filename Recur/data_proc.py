import pandas as pd
import glob
import os
# pd.set_option('display.max_colwidth', 10)
# pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)

path = 'c:\Dataset\crypto_data'
main_set = pd.DataFrame()

all_files = glob.glob(path + '/*.csv')

for file in all_files:
    fname = os.path.splitext(str(file))[0].split('\\')[-1]
    dataset = pd.read_csv(file,names=['time','low','high','open',f'{fname}_close',f'{fname}_volume'],index_col=['time'])
    dataset = dataset[[f'{fname}_close',f'{fname}_volume']]

    if len(main_set)==0:
        main_set = dataset
    else:
        main_set = pd.merge(main_set, dataset, on='time', how='left').fillna(method='ffill')
print(main_set.head())
print(main_set.shape)
print(pd.isnull(main_set).sum())






