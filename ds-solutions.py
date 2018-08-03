import pandas as pd
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from datahandler import get_train_data, get_test_data


x_train, y_train, raw_train, filtered_train = get_train_data()
x_test, y_test, raw_test, filtered_test = get_test_data()


print(raw_train.describe().to_string())
# with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
#     print(raw_train.describe())

nn = MLPClassifier((5,5), max_iter=1000, solver='adam')
nn.fit(x_train, y_train)

score_nn = nn.score(x_train, y_train)
score_nn_test = nn.score(x_test, y_test)
print("NN Train:", score_nn, "NN Test: ", score_nn_test)

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

