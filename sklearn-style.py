import pandas as pd
import numpy as np
from math import isnan

import xgboost
from matplotlib import pyplot as plt
from matplotlib import pylab as plb
import seaborn as sb
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings

from sklearn.tree import ExtraTreeClassifier

import data

warnings.filterwarnings('ignore')
def isNaN(num):
    return num != num

def get_train_data()->(np.ndarray, np.ndarray, pd.DataFrame) :
    trainfile = 'data/train.csv'

    traindata = pd.read_csv(trainfile, sep=',')
    traindata['Sex'].replace('male', 1, inplace=True)
    traindata['Sex'].replace('female', 0, inplace=True)
    traindata['Embarked'] = traindata['Embarked'].map({'C':0, 'Q':1, 'S':2})

    traindata['HasCabin'] = traindata['Cabin'].apply(lambda x: 0 if isNaN(x) else 1)
    traindata.fillna(-1, inplace=True)

    filtered_data = traindata.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
    train = filtered_data.values
    print(filtered_data.head(5))

    x_train = train[:, 1:]
    y_train = train[:, 0]

    return x_train, y_train, traindata, filtered_data


def get_test_data()->(np.ndarray, np.ndarray, pd.DataFrame):
    testfile = 'data/test.csv'

    testdata = pd.read_csv(testfile, sep=',')
    testdata['Sex'].replace('male', 1, inplace=True)
    testdata['Sex'].replace('female', 0, inplace=True)
    testdata['Embarked'] = testdata['Embarked'].map({'C':0, 'Q':1, 'S':2}).astype(int)

    testdata['HasCabin'] = testdata['Cabin'].apply(lambda x: 0 if isNaN(x) else 1)
    testdata.fillna(-1, inplace=True)
    # print(testdata[['HasCabin', 'Cabin']])

    filtered_data = testdata.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
    print(filtered_data.head(5))

    x_test = filtered_data.values

    answers = pd.read_csv('data/gender_submission.csv', sep=',')
    y_test = answers['Survived'].values

    return x_test, y_test, testdata, filtered_data


x_train, y_train, raw_train_data, filtered_train_data = get_train_data()
x_test, y_test, raw_test_data, filtered_test_data = get_test_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Visualize data in graph
color = []
for i in y_train:
    if i ==0:
        color.append('red')
    else:
        color.append('blue')
print(x_train[0].shape, y_train.shape)
red = plt.scatter(filtered_train_data.loc[filtered_train_data['Survived'] == 0]['Age'], filtered_train_data.loc[filtered_train_data['Survived'] == 0]['Sex'], marker='x', color='r', alpha=0.8)
blue = plt.scatter(filtered_train_data.loc[filtered_train_data['Survived'] == 1]['Age'], filtered_train_data.loc[filtered_train_data['Survived'] == 1]['Sex'], marker='o', color='b', alpha=0.15)
plt.xlabel("AGE")
plt.ylabel("SEX")
plt.legend((red, blue), ('Dead', 'Alive'))
plt.show()


female = plt.bar('female', filtered_train_data.loc[filtered_train_data['Survived'] == 0]['Sex'].values.size)
male = plt.bar('male', filtered_train_data.loc[filtered_train_data['Survived'] == 1]['Sex'].values.size)
total = plt.bar('total', filtered_train_data['Sex'].values.size)
plt.show()

# See correlation between the variables
colormap = plt.cm.RdBu
plt.figure()
plt.title("Pearson's Correlation of Features", y=1.05, size=15)
sb.heatmap(filtered_train_data.corr(), linewidths=0.1, vmax=1, square=True, linecolor='white', annot=True)
plt.show()

# See plot between the variables
g = sb.pairplot(filtered_train_data, hue='Survived')
g.set(xticklabels=[])
plt.show()

# Some useful parameters
ntrain = x_train.shape[0]
ntest = x_test.shape[0]
random_state = 0 # for reproducibility
nfolds = 5
kf = KFold(ntrain, n_folds=nfolds, random_state=random_state)

def get_out_of_fold_prediction(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest, ))
    oof_test_skf = np.empty((nfolds, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Random forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

rf = RandomForestClassifier(**rf_params)

rf_oof_train, rf_oof_test = get_out_of_fold_prediction(rf, x_train, y_train, x_test)

# Extra trees
et = ExtraTreeClassifier(max_depth=8, min_samples_leaf=2)
et_oof_train, et_oof_test = get_out_of_fold_prediction(et, x_train, y_train, x_test)

# Ada boost
ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.75)
ada_oof_train, ada_oof_test = get_out_of_fold_prediction(ada, x_train, y_train, x_test)

# Gradient boosting
grd = GradientBoostingClassifier(n_estimators=500, max_depth=5, min_samples_leaf=2, verbose=0)
grd_oof_train, grd_oof_test = get_out_of_fold_prediction(grd, x_train, y_train, x_test)

# Support Vectors
svc = SVC(kernel='linear', C=0.025)
svc_oof_train, svc_oof_test = get_out_of_fold_prediction(svc, x_train, y_train, x_test)

print('Training Complete!')
rf_feature = rf.feature_importances_
feature_dataframe = pd.DataFrame({
    'features': filtered_train_data.columns[1:].values,
    'Random Forest': rf_feature,
    'Extra Trees': et.feature_importances_,
    'AdaBoost': ada.feature_importances_,
    'GradientBoost': grd.feature_importances_
})
print('features', filtered_train_data.columns[1:].values)
print(feature_dataframe.head())

feature_dataframe.plot(x='features', y='Random Forest', kind='bar', title='Random Forest', legend=False)
plt.show()

feature_dataframe.plot(x='features', y='Extra Trees', kind='bar', title='Extra Trees', legend=False)
plt.show()

feature_dataframe.plot(x='features', y='AdaBoost', kind='bar', title='AdaBoost', legend=False)
plt.show()

feature_dataframe.plot(x='features', y='GradientBoost', kind='bar', title='GradientBoost', legend=False)
plt.show()

# Creating new column containing the average of values
feature_dataframe['mean'] = feature_dataframe.mean(axis=1) # row-wise mean
print(feature_dataframe.head())

# Plot mean
feature_dataframe.plot(x='features', y='mean', kind='bar', title='Mean', legend=False)
plt.show()

# Second Level Predictions from First Level Output
base_prediction_train = pd.DataFrame({
    'RandomForest': rf_oof_train.ravel(),
    'ExtraTrees': et_oof_train.ravel(),
    'AdaBoost': ada_oof_train.ravel(),
    'GradientBoost': grd_oof_train.ravel()
})
print(base_prediction_train.head())

# Lets see correlation between columns for second layer prediction
plt.title("Pearson's Correlation For Second Layer", y=1.05, size=15)
sb.heatmap(base_prediction_train.corr(), linewidths=0.1, vmax=1, square=True, linecolor='white', annot=True)
plt.show()

# Combine training data
x_train_final = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train,
                                grd_oof_train, svc_oof_train), axis=1)
print(x_train_final)
x_test_final = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test,
                               grd_oof_test, svc_oof_test), axis=1)

# Finally call XGBClassifier... but Why ?
gbm = xgboost.XGBClassifier(n_estimators=2000, max_depth=4, min_child_weight=2,
                            gamma=0.9, subsample=0.8, colsample_bytree=0.8,
                            objective='binary:logistic', nthread=-1, scale_pos_weight=1)
gbm.fit(x_train_final, y_train)
predictions = gbm.predict(x_test_final) # Test Predictions
train_prediction = gbm.predict(x_train_final)
print('train accuracy: ', accuracy_score(y_train, train_prediction))

# Generate Submission File
submission = pd.DataFrame({
    'PassengerId': raw_test_data['PassengerId'].ravel(),
    'Survived': predictions
}, dtype=int)
submission.to_csv(path_or_buf='submissions/EnsembleSubmission.csv', index=False)


# Train
# random_forest = RandomForestClassifier()
# random_forest.fit(x_train, y_train)
# score_rf = random_forest.score(x_train, y_train)
# score_rf_test = random_forest.score(x_test, y_test)
# print('Random Forest Train: ', score_rf, "Test: ", score_rf_test)
# print(random_forest.feature_importances_)

'''
nn = MLPClassifier((5,5), max_iter=1000, solver='adam')
nn.fit(x_train, y_train)

score_nn = nn.score(x_train, y_train)
score_nn_test = nn.score(x_test, y_test)
print("NN Train:", score_nn, "NN Test: ", score_rf_test)

svm = SVC()
svm.fit(x_train, y_train)
score_svm = svm.score(x_train, y_train)
score_svm_test = svm.score(x_test, y_test)
print("SVM: Train", score_svm, "SVM Test:", score_svm_test)

lr = RidgeClassifier()
lr.fit(x_train, y_train)
score_lr = lr.score(x_train, y_train)
score_lr_test = lr.score(x_test, y_test)
print("Ridge Regr Train:", score_lr, "Test:", score_lr_test)

logistic = LogisticRegression()
logistic.fit(x_train, y_train)
score_logistic = logistic.score(x_train, y_train)
score_logistic_test = logistic.score(x_test, y_test)
print('Logistic Regr Train:', score_logistic, 'Test:', score_logistic_test)

nn_search = MLPClassifier()
nn_cv = GridSearchCV(nn_search, {
    'hidden_layer_sizes':[(2), (2,2), (3), (3,3), (4,4), (5,3), (5,4), (5,5), (10,5)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver':['adam', 'sgd'],
    # 'verbose':[5],
    'early_stopping': [True]
}, n_jobs=-1)
nn_cv.fit(x_train, y_train)
print(nn_cv.best_estimator_, nn_cv.best_score_)
'''

# plot decision boundary
# age_min, age_max = train_data['Age'].values.min() - 1, train_data['Age'].values.max() + 1
# y_min, y_max = y_train.min() -1, y_train.max() +1
#
# print(age_min, age_max, y_min, y_max)
#
# xx, yy = np.meshgrid(np.arange(age_min, age_max, 0.1),
#                      np.arange(y_min, y_max, 0.1) )
#
# z = nn.predict()


# Plot histogram
# probability of dying male of specific age
# probability of dying female of specific age

