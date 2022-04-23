import numpy as np
import matplotlib.pyplot as plt

dir_names = ['tech', 'sport', 'politics', 'entertainment', 'business']
vectorizers = ['CV', 'TF', 'TFIDF']

fig, ax = plt.subplots(5, 3, figsize=(12,13), dpi=200)

for d_id, dir in enumerate(dir_names):
    for v_id, vect in enumerate(vectorizers):
        img = plt.imread('cd/%s%s.png' % (dir, vect))

        ax[d_id, v_id].imshow(img)
        ax[d_id, v_id].set_xticks([])
        ax[d_id, v_id].set_yticks([])

        if v_id==0:
            ax[d_id, v_id].set_ylabel(dir, fontsize=14)        
        if d_id==0:
            ax[-1, v_id].set_xlabel(vect, fontsize=14)

        
plt.tight_layout()
plt.savefig('foo.png')
        

        