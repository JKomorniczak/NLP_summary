import os
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
import time
from sklearn.metrics import accuracy_score

"""
Tylko dla najlepszego klasyfikatora i ekstraktora na podstawie exp news
"""

np.random.seed(123)

def get_base_classifiers(random_state):
    return [
    GaussianNB(),
    ]

base_extractors = [
    SelectKBest(chi2, k=25)
    ]

dirs = ['tech/', 'sport/', 'politics/', 'entertainment/', 'business/']
v_methods = ['cv/', 'tf/', 'tfidf/']
vect_dir = 'vect_c/'

# count number of files in each cat
news_in_cat = [0,0,0,0,0]
for c_id, category in enumerate(dirs):
    dir = "%s%s%s" % (vect_dir, v_methods[0], category)
    arr = os.listdir(dir)
    if '.DS_Store' in arr:
        arr.remove('.DS_Store')
    news_in_cat[c_id] = len(arr)

reps = 1
random_states = np.random.randint(100,100000,reps)
base_clf_num = 1

results=[]
for news_num in news_in_cat:
    r = np.zeros((reps, len(v_methods), len(base_extractors), base_clf_num, news_num))
    results.append(r)
    # categories x rep x vectorization methods x base extractors x clfs x news


# repeats
start = time.time()
for rep, r in enumerate(random_states):
    base_classifiers = get_base_classifiers(r)

    # news categories
    for category_id, category in enumerate(dirs):
        # metody wektoryzacji
        for v_id, v in enumerate(v_methods):
            dir = "%s%s%s" % (vect_dir, v, category)
            arr = os.listdir(dir)
            if '.DS_Store' in arr:
                arr.remove('.DS_Store')
            
            # load data
            for n_id, news in enumerate(arr):
                data = np.load(dir+news)
                X_test = data[:,:-1]
                y_test = data[:,-1]

                #collect other (train data)
                other_data = []
                for news_other in arr:
                    if news_other == news:
                        continue
                    data = np.load(dir+news_other)
                    other_data.append(data)
                od = [o for o in other_data]
                other_data = np.concatenate((od), axis=0)
                
                X_train = other_data[:,:-1]
                y_train = other_data[:,-1]

                # ekstraktory
                for e_id, e in enumerate(base_extractors):

                    X_all = np.concatenate((X_train,X_test), axis=0)
                    y_all = np.concatenate((y_train,y_test), axis=0)

                    extractor = clone(e)
                    # gdy nie mozna ekstrakcji
                    try:
                        X_extracted = extractor.fit(X_all, y_all)
                        X_test_extracted = extractor.transform(X_test)
                        X_train_extracted = extractor.transform(X_train)
                    except:
                        X_test_extracted = np.copy(X_test)
                        X_train_extracted = np.copy(X_train)

                    # klasyfikatory
                    for c_id, base in enumerate(base_classifiers):                       
                        clf = clone(base)
                        y_pred = clf.fit(X_train_extracted, y_train).predict(X_test_extracted)
                        accuracy = accuracy_score(y_pred, y_test)

                        results[category_id][rep, v_id, e_id, c_id, n_id] = accuracy

                    print(accuracy)
                    print(category, v, e_id, c_id, n_id)
                    print('time:', time.time()-start)

            for r_id, r in enumerate(results):
                np.save('res_cat_%i'%r_id, r)
                        



