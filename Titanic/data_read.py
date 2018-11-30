import pandas as pd
# pd.set_option('display.width', 320)
# pd.set_option('display.max_colwidth', -1)
pd.set_option('display.expand_frame_repr', False)
# pd.set_option('chained_assignment',None)




def preprocessing(d_type):
    FILE_PATCH = 'c:/Dataset/titanic/'
    x_df = pd.read_csv(f'{FILE_PATCH}/{d_type}.csv')
    df_all = pd.DataFrame()
    df_all = x_df
    # print(all_df.shape) #(891, 12)
    df_all.drop('PassengerId', axis=1, inplace=True)
    df_all.loc[df_all['Embarked'].isnull(),['Embarked']] = 'S'
    df_all['Sex'] = df_all['Sex'].map({'male':1, 'female':0})
    df_all.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
    df_all['Embarked'] = df_all['Embarked'].map({'S':0,'C':1,'Q':2})
    df_all.loc[df_all['Age'].isnull()] = df_all['Age'].median()
    categorical_val = pd.get_dummies(df_all[['Sex','Pclass','Embarked']],drop_first=True)
    # print(df_all.head())
    # df_all = pd.concat([df_all,categorical_val],axis=1)
    # print(df_all.head())
    ################Standarization###############################
    df_all['Age'] = df_all['Age'] / 100
    df_all['Fare'] = df_all['Fare'] / df_all['Fare'].max()
    #############################################################
    if d_type == 'train':
        X_train = df_all.drop('Survived',axis=1)
        y_train = df_all['Survived']
        return (X_train,y_train)
    else:
        return df_all

    # print(pd.isnull(df_all).sum())


X_train,y_train = preprocessing('train')
print(X_train.head())
