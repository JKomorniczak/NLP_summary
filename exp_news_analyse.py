import numpy as np
import matplotlib.pyplot as plt

res = np.load('res_news_0.npy')[0]
print(res.shape) # vactorizers, extractors, clfs, news

extractors = ['KBest5', 'KBest15', 'KBest25', 'PCA5', 'PCA15', 'PCA25']
clfs = ['GNB', 'KNN', 'MLP', 'DT']

fig, ax = plt.subplots(6,4, figsize=(12,12), sharex=True, sharey=True)

for e_id, e in enumerate(extractors):
    for clf_id, clf in enumerate(clfs):

        r = res[0,e_id,clf_id]
        ax[e_id, clf_id].hist(r, bins=30)
        ax[e_id, clf_id].set_title(clf+" "+e)

plt.tight_layout()
plt.savefig('foo.png')
