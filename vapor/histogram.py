import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import os

dir_text = 'BBC News Summary/News Articles/tech/'
dir_summ = 'BBC News Summary/Summaries/tech/'

arr = os.listdir(dir_text)

counts = []
ir=[]

for news in arr:
    with open(dir_text + news) as f:
        text = f.read()
    with open(dir_summ + news) as f:
        summ = f.read()
    
    summ = summ.replace('.', '. ')

    sentence_list_text = nltk.sent_tokenize(text)
    sentence_list_summ = nltk.sent_tokenize(summ)

    counts.append(len(sentence_list_text))
    ir.append(len(sentence_list_summ)/len(sentence_list_text))

    # # nw
    # w2v = gensim.downloader.load('word2vec-google-news-300')

    # for sentence in sentence_list:
    #     v = [w2v[x] for x in sentence_list[0]]
    #     print(v)

fig, ax = plt.subplots(2,1,figsize=(6,10))

ax[0].hist(counts, bins=50)
ax[0].set_title('sentence num')

ax[1].hist(ir, bins=50)
ax[1].set_title('IR')
plt.savefig('foo.png')

