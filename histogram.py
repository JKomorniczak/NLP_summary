import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
subjects = ['tech', 'sport', 'politics', 'entertainment', 'business']

fig, ax = plt.subplots(5,2,figsize=(7,10))
ax[0,0].set_title('Article length in sentences')
ax[0,1].set_title('Sentences fraction in summaries')

for s_id, subject in enumerate(subjects):

    dir = 'prep/%s' % subject
    arr = os.listdir(dir)

    counts = []
    ir=[]

    for news in arr:
        if news=='.DS_Store':
            continue
        data = pd.read_csv("%s/%s" % (dir, news), index_col=0)

        c = len(data.index)
        counts.append(c)

        i = np.sum(data.Label == 1)/c
        ir.append(i)

    ax[s_id,0].hist(counts, bins=50)
    ax[s_id,0].set_xlim(0, 80)

    ax[s_id,1].hist(ir, bins=50)
    ax[s_id,1].set_xlim(0.35, 0.55)

    ax[s_id,0].grid()
    ax[s_id,1].grid()

    ax[s_id,0].set_ylabel(subject)

plt.tight_layout()
plt.savefig('foo.png')
