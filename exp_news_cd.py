import numpy as np
from scipy.stats import rankdata
from scipy.stats import ranksums
from tabulate import tabulate
import Orange
import matplotlib.pyplot as plt

dirs = ['tech/', 'sport/', 'politics/', 'entertainment/', 'business/']
dir_names = ['tech', 'sport', 'politics', 'entertainment', 'business']
alfa = .05

for d_id, dir in enumerate(dirs):

    res = np.load('res/res_news_%i.npy' % d_id)[0]
    print(res.shape) # vactorizers, extractors, clfs, news

    res = res.reshape(res.shape[0], res.shape[1]*res.shape[2], res.shape[3])
    print(res.shape) # 3, 24, news

    extractors = ['KBest-5', 'KBest-15', 'KBest-25', 'PCA-5', 'PCA-15', 'PCA-25']
    clfs = ['GNB', 'KNN', 'MLP', 'DT']
    vectorizers = ['CV', 'TF', 'TFIDF']

    methods= []

    for e in extractors:
        for c in clfs:
            methods.append('%s-%s' % (e,c))

    for v_id, vect in enumerate(vectorizers):
        current_res = res[v_id].T
        print(current_res.shape)

        ranks = []
        for r in current_res:
            ranks.append(rankdata(r).tolist())
        ranks = np.array(ranks)
        print(ranks.shape)

        mean_ranks = np.mean(ranks, axis=0)

        k=24
        n = ranks.shape[0]
        cd = 2.77 * np.sqrt((k*(k+1))/(6*n))
        Orange.evaluation.graph_ranks(mean_ranks, methods, cd=cd, width=8, textspace=1.5, filename='foo2')
        plt.savefig('foo.png')
        exit()
