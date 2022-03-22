import os
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


dirs = ['tech/', 'sport/', 'politics/', 'entertainment/', 'business/']
v_methods = ['cv/', 'tf/', 'tfidf/']
vect_dir = 'vect/'

r=[]

for d in dirs:
    dir = "%s%s%s" % (vect_dir, v_methods[0], d) # temp tylko cv
    arr = os.listdir(dir)

    for file in arr:
        print(dir+file)
        if file=='.DS_Store':
            continue
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

        r.append(accuracy)
    plt.hist(r, bins=60)
    plt.savefig('foo_n.png')

    exit()