import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import os

subjects = ['tech', 'sport', 'politics', 'entertainment', 'business']

for subject in subjects:

    dir_text = 'BBC News Summary/News Articles/' + subject + '/'
    dir_summ = 'BBC News Summary/Summaries/' + subject + '/'

    arr = os.listdir(dir_text)

    counts = []
    ir=[]

    for news in arr:
        with open(dir_text + news) as f:
            # News Articles/sport/199.txt - remove pound symbol
            text = f.read()
        with open(dir_summ + news) as f:
            summ = f.read()

        summ = summ.replace('.', '. ')

        sentence_list_text = nltk.sent_tokenize(text)
        sentence_list_summ = nltk.sent_tokenize(summ)

        counts.append(len(sentence_list_text))
        ir.append(len(sentence_list_summ)/len(sentence_list_text))

    fig, ax = plt.subplots(2,1,figsize=(6,10))

    ax[0].hist(counts, bins=50)
    ax[0].set_title(subject + ' - articles lengths')
    ax[0].set_xlabel("Article length")
    ax[0].set_ylabel("Number of articles")

    ax[1].hist(ir, bins=50)
    ax[1].set_title(subject + ' - sentences percentage in summaries')
    ax[1].set_xlabel("sentences percentage in summaries")
    ax[1].set_ylabel("Number of summaries")
    
    plt.savefig(subject + '.png')
