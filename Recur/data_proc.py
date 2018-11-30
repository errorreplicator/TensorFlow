import pandas as pd
import glob
import os
from sklearn import preprocessing
from collections import deque
import numpy as np
import random

pd.set_option('display.expand_frame_repr', False)
pd.set_option('chained_assignment',None)

F_NAME = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
TYPE_TO_PRED = F_NAME[1]
PRED_PERIODS = 3
SEQ_LEN = 30

def convert_binary(current,future):
    if future > current:
        return 1
    else:
        return 0

def preprocess(df):
    df.drop('future',axis=1)
    df.dropna(inplace=True)
    for col in df.columns:
        if col != 'label':
            # df[col] = df.loc[:,col].pct_change()
            df.loc[:,col] = df[col].pct_change()
            # df[col] = preprocessing.scale(df[col].values)
            # df[col] = preprocessing.scale(df.loc[:,col])
            df.loc[:,col] = preprocessing.scale(df[col])
    df.dropna(inplace=True)

    sequence = []
    prev_days = deque(maxlen=SEQ_LEN)
    # print(df.head(5))
    for i in df.values:
        prev_days.append(list(i[:-1])) #tutorial n for n in i[:-1]
        if len(prev_days) == SEQ_LEN:
            sequence.append([np.array(prev_days),i[-1]])
    random.shuffle(sequence)

    zero = []
    one = []
    for el_1, el_2 in sequence:
        if el_2 == 1:
            one.append([el_1,el_2])
        else:
            zero.append([el_1,el_2])
    index = min(len(one),len(zero))
    zero = zero[:index]
    one = one[:index]
    sequence = zero + one
    random.shuffle(sequence)
    # print(sequence)
    X_return = []
    y_return = []
    for x,y in sequence:
        X_return.append(x)
        y_return.append(y)
    # print('X_return:',X_return)
    # print('y_return',y_return)
    return (np.array(X_return),y_return)


def time_data():
    path = 'c:\Dataset\crypto_data'
    main_set = pd.DataFrame()
    all_files = glob.glob(path + '/*.csv')


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
    main_set.dropna(inplace=True)

    # main_set['time_col'] = pd.to_datetime(main_set.index,unit='s')
    # main_set.drop(['BCH-USD_close','BCH-USD_volume'],axis=1,inplace=True) #comment out before learning


    main_set['label'] = main_set.apply(lambda main_set: convert_binary(current=main_set['LTC-USD_close'],future=main_set['future']),axis=1)

    point = int(0.05*len(main_set))
    split_point = sorted(main_set.index.values)[-point] ##############tutorial used sorted instead of list
    train_data = main_set[main_set.index.values<split_point]
    test_data = main_set[main_set.index.values>=split_point]
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    ####Little test
    X_train, y_train = preprocess(train_data)
    X_valid, y_valid = preprocess(test_data)
    return (X_train, y_train,X_valid, y_valid)

# print(f"train data: {len(X_train)} validation: {len(X_valid)}")
# print(f"Dont buys: {y_train.count(0)}, buys: {y_train.count(1)}")
# print(f"VALIDATION Dont buys: {y_valid.count(0)}, buys: {y_valid.count(1)}")
