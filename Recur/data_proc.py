import pandas as pd
import glob
import os
from sklearn import preprocessing
from collections import deque
import numpy as np
import random

pd.set_option('display.expand_frame_repr', False)

path = 'c:\Dataset\crypto_data'
main_set = pd.DataFrame()
f_names = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
all_files = glob.glob(path + '/*.csv')
TYPE_TO_PRED = f_names[1]
PRED_PERIODS = 3
SEQ_LEN = 5

def convert_binary(current,future):
    if future > current:
        return 1
    else:
        return 0

def preprocess(df):
    df = df.drop('future',axis=1)
    for col in df.columns:
        if col != 'label':
            df[col] = df[col].pct_change()
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    sequence = []
    prev_days = deque(maxlen=SEQ_LEN)
    print(df.head(5))
    for i in df.values:
        prev_days.append(list(i[:-1])) #tutorial n for n in i[:-1]
        if len(prev_days) == SEQ_LEN:
            sequence.append([np.array(prev_days),i[-1]])
    random.shuffle(sequence)
    return (sequence)

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
main_set.drop(['BCH-USD_close','BCH-USD_volume'],axis=1,inplace=True) #comment out before learning
# print(main_set.tail(10))

main_set['label'] = main_set.apply(lambda main_set: convert_binary(current=main_set['LTC-USD_close'],future=main_set['future']),axis=1)

####################SHAFLE ??? before split ???#######################
# print(main_set.shape)
# print(main_set.head())

point = int(0.05*len(main_set))
split_point = list(main_set.index.values)[-point] ##############tutorial used sorted instead of list
train_data = main_set[main_set.index.values>split_point]
test_data = main_set[main_set.index.values<=split_point]


####Little test
dataframe = main_set[['LTC-USD_close','LTC-USD_volume','future','label']]
preprocess(dataframe.head(15))

