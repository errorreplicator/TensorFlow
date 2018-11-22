import pandas as pd
import glob
import os
import time
# pd.set_option('display.max_colwidth', 10)
# pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)

path = 'c:\Dataset\crypto_data'
main_set = pd.DataFrame()
f_names = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
all_files = glob.glob(path + '/*.csv')
TYPE_TO_PRED = f_names[1]
PRED_PERIODS = 3

for file in all_files:
    fname = os.path.splitext(str(file))[0].split('\\')[-1]
    dataset = pd.read_csv(file,names=['time','low','high','open',f'{fname}_close',f'{fname}_volume'])
    dataset.set_index('time',inplace=True)
    dataset = dataset[[f'{fname}_close',f'{fname}_volume']]

    if len(main_set)==0:
        main_set = dataset
    else:
        main_set = pd.merge(main_set, dataset, on='time', how='left').fillna(method='ffill')

main_set['future'] = main_set[f'{TYPE_TO_PRED}_close'].shift(-PRED_PERIODS)
main_set.dropna(axis=0,inplace=True)

main_set['time_col'] = pd.to_datetime(main_set.index,unit='s')
print(main_set.head(10))





