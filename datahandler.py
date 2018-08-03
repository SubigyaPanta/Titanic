import pandas as pd
import numpy as np


def isNaN(num):
    return num != num


def get_train_data() -> (np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame):
    trainfile = 'data/train.csv'

    traindata = pd.read_csv(trainfile, sep=',')
    traindata['Sex'].replace('male', 1, inplace=True)
    traindata['Sex'].replace('female', 0, inplace=True)
    traindata['Embarked'] = traindata['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    traindata['HasCabin'] = traindata['Cabin'].apply(lambda x: 0 if isNaN(x) else 1)
    traindata.fillna(-1, inplace=True)

    filtered_data = traindata.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
    train = filtered_data.values
    print(filtered_data.head(5))

    x_train = train[:, 1:]
    y_train = train[:, 0]

    return x_train, y_train, traindata, filtered_data


def get_test_data() -> (np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame):
    testfile = 'data/test.csv'

    testdata = pd.read_csv(testfile, sep=',')
    testdata['Sex'].replace('male', 1, inplace=True)
    testdata['Sex'].replace('female', 0, inplace=True)
    testdata['Embarked'] = testdata['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

    testdata['HasCabin'] = testdata['Cabin'].apply(lambda x: 0 if isNaN(x) else 1)
    testdata.fillna(-1, inplace=True)
    # print(testdata[['HasCabin', 'Cabin']])

    filtered_data = testdata.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
    print(filtered_data.head(5))

    x_test = filtered_data.values

    answers = pd.read_csv('data/gender_submission.csv', sep=',')
    y_test = answers['Survived'].values

    return x_test, y_test, testdata, filtered_data
