import numpy as np
import matplotlib.pyplot as plt

dirs = ['tech/', 'sport/', 'politics/', 'entertainment/', 'business/']

for d_id, dir in enumerate(dirs):

    res_c = np.load('res/res_cat_%i.npy' % d_id)[0]
    res_n = np.load('res/res_news_%i.npy' % d_id)[0]

    res_n = res_n[:,2,0]
    res_c = res_c[:,0,0]

    print(res_n.shape) # vactorizers, extractors, clfs, news
    print(res_c.shape) # vactorizers, extractors, clfs, news

    extractors = ['KBest-25']
    clfs = ['GNB']
    vectorizers = ['CV', 'TF', 'TFIDF']

    fig, ax = plt.subplots(1,3, figsize=(8,4), sharex=True, sharey=True)
    plt.suptitle('%s - GNB, KBest-25' % (dir), fontsize=18)

    for v_id, vect in enumerate(vectorizers):
        for e_id, e in enumerate(extractors):
            for clf_id, clf in enumerate(clfs):

                ax[v_id].hist(res_c[v_id], bins=30, label='category', alpha=0.7)                
                ax[v_id].hist(res_n[v_id], bins=30, label='news', alpha=0.7)
                ax[v_id].set_title(vect)
                ax[v_id].set_xlim(0,1)

    plt.legend()
    plt.tight_layout()
    plt.savefig('fig/e1_c_%s.png' % (d_id))
