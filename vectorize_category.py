from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
import numpy as np
import pandas as pd

dirs = ['tech/', 'sport/', 'politics/', 'entertainment/', 'business/']
preprocessed_dir = 'prep/'

for d in dirs:
    dir = "%s%s" % (preprocessed_dir, d)
    arr = os.listdir(dir)

    category_sentences = []

    for file in arr:
        if file=='.DS_Store':
            continue
        df = pd.read_csv(dir+file, index_col=0)
        sentences = df['Sentence'].tolist()

        for s in sentences:
            category_sentences.append(s)

    # TF-IDF
    tfidf = TfidfVectorizer()
    tfidf.fit(category_sentences)

    # TF
    tf = TfidfVectorizer(use_idf=False)
    tf.fit(category_sentences)

    # TF
    cv = CountVectorizer()
    cv.fit(category_sentences)
    
    for file in arr:
        if file=='.DS_Store':
            continue
        print(dir+file)
        df = pd.read_csv(dir+file, index_col=0)
        sentences = df['Sentence']
        labels = df['Label']

        vs_tfidf=[]
        vs_tf=[]
        vs_cv=[]

        for s in sentences:
            v_tfidf = tfidf.transform([s]).toarray()[0,:]
            vs_tfidf.append(v_tfidf)

            v_tf = tf.transform([s]).toarray()[0,:]
            vs_tf.append(v_tf)

            v_cv = cv.transform([s]).toarray()[0,:]
            vs_cv.append(v_cv)


        array_tfidf = np.concatenate((np.array(vs_tfidf), labels.to_numpy()[:, np.newaxis]), axis=1)
        print(array_tfidf.shape)
        
        array_tf = np.concatenate((np.array(vs_tf), labels.to_numpy()[:, np.newaxis]), axis=1)
        print(array_tf.shape)
        
        array_cv = np.concatenate((np.array(vs_cv), labels.to_numpy()[:, np.newaxis]), axis=1)
        print(array_cv.shape)

        np.save('vect_c/tfidf/%s%s' % (d, file.split('.')[0]), array_tfidf)
        np.save('vect_c/tf/%s%s' % (d, file.split('.')[0]), array_tf)
        np.save('vect_c/cv/%s%s' % (d, file.split('.')[0]), array_cv)

        # exit()

