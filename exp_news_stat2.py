import numpy as np
from scipy.stats import rankdata
from scipy.stats import ranksums
from tabulate import tabulate

dirs = ['tech/', 'sport/', 'politics/', 'entertainment/', 'business/']
dir_names = ['tech', 'sport', 'politics', 'entertainment', 'business']
alfa = .05

extractors = ['KBest-5', 'KBest-15', 'KBest-25', 'PCA-5', 'PCA-15', 'PCA-25']
clfs = ['GNB', 'KNN', 'MLP', 'DT']
vectorizers = ['CV', 'TF', 'TFIDF']

methods= []

for e in extractors:
    for c in clfs:
        methods.append('%s-%s' % (e,c))\

rows=[]
a = ['(%i)' %i for i in np.arange(1,len(methods)+1)]
rows.append(a)

for d_id, dir in enumerate(dirs):

    res = np.load('res/res_news_%i.npy' % d_id)[0]
    print(res.shape) # vactorizers, extractors, clfs, news

    res = res.reshape(res.shape[0], res.shape[1]*res.shape[2], res.shape[3])
    print(res.shape) # 3, 24, news

    a = ['' for i in methods]
    a[0] = dir_names[d_id]
    rows.append(a)
    
    for v_id, vect in enumerate(vectorizers):
        current_res = res[v_id].T
        print(current_res.shape)

        mean_news = np.around(np.mean(current_res, axis=1), decimals=3)
        rows.append(mean_news)

        ranks = []
        for r in current_res:
            ranks.append(rankdata(r).tolist())
        ranks = np.array(ranks)

        w_statistic = np.zeros((len(extractors), len(extractors)))
        p_value = np.zeros((len(extractors), len(extractors)))

        for i in range(len(extractors)):
            for j in range(len(extractors)):
                w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

        advantage = np.zeros((len(extractors), len(extractors)))
        advantage[w_statistic > 0] = 1

        significance = np.zeros((len(extractors), len(extractors)))
        significance[p_value <= alfa] = 1

        stat_better = significance * advantage

        print(stat_better)

        table = tabulate(rows, methods, floatfmt=".3f", tablefmt="latex")
        f = open("tables/table.tex", "w")
        f.write(table)
        f.close()
        

# exit()



