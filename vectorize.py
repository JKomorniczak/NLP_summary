from numpy import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
import numpy as np
import pandas as pd

# vectorizer = TfidfVectorizer()
# vectorizer = TfidfVectorizer(use_idf=False)
# vectorizer = CountVectorizer()

# TF-IDF
dir = 'preprocessed_tech/'
arr = os.listdir(dir)

for file in arr:
    df = pd.read_csv(dir+file, index_col=0)
    sentences = df['Sentence']
    labels = df['Label']

    vectorizer = TfidfVectorizer()
    vectorizer.fit(sentences)

    vs=[]
    for s in sentences:
        v = vectorizer.transform([s]).toarray()[0,:]
        vs.append(v)

    array = np.concatenate((np.array(vs), labels.to_numpy()[:, np.newaxis]), axis=1)
    print(array.shape)

    np.save('vectorized_tech_tfidf/%s' % file, array)
    # exit()

