import pandas as pd
pd.set_option('display.expand_frame_repr', False)
df = pd.read_csv('c:/Dataset/titanic/train_top.csv')
print(df.head())