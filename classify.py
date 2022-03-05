import os
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB


dir = 'vectorized_tech_tfidf/'
arr = os.listdir(dir)

for file in arr:
    data = np.load(dir+file)
    X = data[:,:-1]
    y = data[:,-1]

    X = SelectKBest(chi2, k=10).fit_transform(X, y)

    res = []

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = GaussianNB()
        y_pred = clf.fit(X_train, y_train).predict(X_test)

        res.append(y_pred == y_test)
    
    res = np.array(res, dtype=int)
    accuracy = np.sum(res)/len(res)
    print(accuracy)
    # exit()