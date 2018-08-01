import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def get_train_data()->(np.ndarray, np.ndarray) :
    trainfile = 'data/train.csv'

    traindata = pd.read_csv(trainfile, sep=',')
    traindata['Age'].fillna(0, inplace=True)
    traindata['Sex'].replace('male', 1, inplace=True)
    traindata['Sex'].replace('female', 0, inplace=True)
    traindata['Embarked'].replace('C', 0, inplace=True)
    traindata['Embarked'].replace('Q', 1, inplace=True)
    traindata['Embarked'].replace('S', 2, inplace=True)
    traindata['Embarked'].replace('', -1, inplace=True)
    traindata.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
    train = traindata.values
    # print(traindata.head)

    x_train = train[:, 1:]
    y_train = train[:, 0]

    return x_train, y_train