import os
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

np.random.seed(123)

def get_base_classifiers(random_state):
    return [
    GaussianNB(),
    KNeighborsClassifier(),
    MLPClassifier(random_state=random_state),
    DecisionTreeClassifier(random_state=random_state)
    ]

base_extractors = [
    SelectKBest(chi2, k=5),
    SelectKBest(chi2, k=10),
    SelectKBest(chi2, k=15),
    SelectKBest(chi2, k=20),
    SelectKBest(chi2, k=25),
    PCA(n_components=5),
    PCA(n_components=10),
    PCA(n_components=15),
    PCA(n_components=20),
    PCA(n_components=25)]

dirs = ['tech/', 'sport/', 'politics/', 'entertainment/', 'business/']
v_methods = ['cv/', 'tf/', 'tfidf/']
vect_dir = 'vect/'

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
base_clf_num = 4

results=[]
for news_num in news_in_cat:
    r = np.zeros((reps, len(v_methods), len(base_extractors), base_clf_num, news_num))
    results.append(r)
    # categories x rep x vectorization methods x base extractors x clfs x news


# repeats
for rep, r in enumerate(random_states):
    base_classifiers = get_base_classifiers(r)
    print(base_classifiers)

    # metody wektoryzacji
    for v_id, v in enumerate(v_methods):
        # news categories
        for category_id, category in enumerate(dirs):
            dir = "%s%s%s" % (vect_dir, v, category)
            arr = os.listdir(dir)
            if '.DS_Store' in arr:
                arr.remove('.DS_Store')
            
            # load data
            for n_id, news in enumerate(arr):
                data = np.load(dir+news)
                X = data[:,:-1]
                y = data[:,-1]

                # ekstraktory
                for e_id, e in enumerate(base_extractors):

                    extractor = clone(e)
                    
                    # gdy nie mozna ekstrakcji
                    try:
                        X_extracted = extractor.fit_transform(X, y)
                    except:
                        X_extracted = np.copy(X)

                    # klasyfikatory
                    for c_id, base in enumerate(base_classifiers):

                        loo = LeaveOneOut()
                        loo_res = []
                        for train_index, test_index in loo.split(X_extracted):
                            clf = clone(base)
                            y_pred = clf.fit(X_extracted[train_index], y[train_index]).predict(X_extracted[test_index])

                            loo_res.append(y_pred == y[test_index])

                        accuracy = np.sum(loo_res)/len(loo_res)
                        results[category_id][rep, v_id, e_id, c_id, n_id] = accuracy
                        print(accuracy)
                        print(category, rep, v, e_id, c_id, n_id)

        for r_id, r in enumerate(results):
            np.save('res_news_%i'%r_id, r)
                    



