import numpy as np
import matplotlib.pyplot as plt

dirs = ['tech/', 'sport/', 'politics/', 'entertainment/', 'business/']

for d_id, dir in enumerate(dirs):

    res = np.load('res/res_news_%i.npy' % d_id)[0]
    print(res.shape) # vactorizers, extractors, clfs, news

    extractors = ['KBest-5', 'KBest-15', 'KBest-25', 'PCA-5', 'PCA-15', 'PCA-25']
    clfs = ['GNB', 'KNN', 'MLP', 'DT']
    vectorizers = ['CV', 'TF', 'TFIDF']

    for v_id, vect in enumerate(vectorizers):

        fig, ax = plt.subplots(6,4, figsize=(10,10), sharex=True, sharey=True)
        plt.suptitle('%s %s' % (dir, vect), fontsize=18)
        for e_id, e in enumerate(extractors):
            for clf_id, clf in enumerate(clfs):

                r = res[v_id,e_id,clf_id]
                ax[e_id, clf_id].hist(r, bins=30)

                if e_id==0:
                    ax[e_id, clf_id].set_title(clf)
                if clf_id==0:
                    ax[e_id, clf_id].set_ylabel(e)

        plt.tight_layout()
        plt.savefig('fig/e1_%s_%s.png' % (d_id,vect))
