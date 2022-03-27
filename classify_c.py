import os
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score

dirs = ['tech/', 'sport/', 'politics/', 'entertainment/', 'business/']
v_methods = ['cv/', 'tf/', 'tfidf/']
vect_dir = 'vect_c/'

r = []

for d in dirs:
    dir = "%s%s%s" % (vect_dir, v_methods[0], d)
    arr = os.listdir(dir)

    for file in arr:
        res = []

        # test data
        if file=='.DS_Store':
            continue
        data = np.load(dir+file)
        X_test = data[:,:-1]
        y_test = data[:,-1]

        #collect other (train data)
        other_data = []
        for file_other in arr:
            if file_other in ['.DS_Store', file]:
                continue
            data = np.load(dir+file)
            other_data.append(data)
        other_data = np.array(other_data)
        other_data = other_data.reshape((other_data.shape[0]*other_data.shape[1], -1))
        X_train = other_data[:,:-1]
        y_train = other_data[:,-1]

        # select features
        X_all = np.concatenate((X_train,X_test), axis=0)
        y_all = np.concatenate((y_train,y_test), axis=0)

        selector = SelectKBest(chi2, k=10).fit(X_all, y_all)

        X_test = selector.transform(X_test)
        X_train = selector.transform(X_train)

        # classify
        clf = GaussianNB()
        y_pred = clf.fit(X_train, y_train).predict(X_test)

        bac = balanced_accuracy_score(y_pred, y_test)
        print(bac)
        res.append(bac)
        r.append(bac)

    plt.hist(r, bins=60)
    plt.savefig('foo_c.png')

    exit()