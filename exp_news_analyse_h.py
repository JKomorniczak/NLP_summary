import numpy as np
import matplotlib.pyplot as plt

dirs = ['tech', 'sport', 'politics', 'entertainment', 'business']
extractors = ['KBest-5', 'KBest-15', 'KBest-25', 'PCA-5', 'PCA-15', 'PCA-25']
clfs = ['GNB', 'KNN', 'MLP', 'DT']
vectorizers = ['CV', 'TF', 'TFIDF']

fig, ax = plt.subplots(5, 3, figsize=(10,13), sharex=True, sharey=True)

# collect
for d_id, dir in enumerate(dirs):
    res = np.load('res/res_news_%i.npy' % d_id)[0]
    # print(res.shape) # vactorizers, extractors, clfs, news
    for v_id, vect in enumerate(vectorizers):
        r_mean = np.mean(res[v_id], axis=2)

        axx = ax[d_id, v_id]
        axx.imshow(r_mean.T, vmax=1, vmin=0.5)
        # axx.set_title(dir+' '+vect)
        axx.set_xticks(range(len(extractors)))
        axx.set_xticklabels(extractors, rotation=90)
        axx.set_yticks(range(len(clfs)))
        axx.set_yticklabels(clfs)

        if v_id==0:
            axx.set_ylabel(dir)
        if d_id==0:
            axx.set_title(vect)

        for _a, __a in enumerate(extractors):
            for _b, __b in enumerate(clfs):
                axx.text(_a, _b, "%.3f" % (
                    r_mean[_a, _b]) , va='center', ha='center', c='white', fontsize=11)

plt.tight_layout()
plt.savefig('fig/hm_e1.png')

