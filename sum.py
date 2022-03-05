from logging.config import stopListening
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import re
stopwords = stopwords.words('english')

# load full text
with open('BBC News Summary/News Articles/tech/002.txt') as f:
    text = f.readlines()

fulltext = ''
for t in text:
    fulltext += t

sentence_list = nltk.sent_tokenize(fulltext)

fulltext = re.sub('[^a-zA-Z]', ' ', fulltext )
fulltext= fulltext.lower()

# frequency-based
tokenized = word_tokenize(fulltext)

frequencies = {}
for token in tokenized:
    if token not in stopwords and token not in punctuation:
        if token in frequencies.keys():
            frequencies[token] +=1
        else:
            frequencies[token] =1
print(frequencies)


maximum_frequncy = max(frequencies.values())

for word in frequencies.keys():
    frequencies[word] = (frequencies[word]/maximum_frequncy)

print(frequencies)

sentence_scores = {}
for sent in sentence_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = frequencies[word]
                else:
                    sentence_scores[sent] += frequencies[word]

print(sentence_scores)

import heapq
summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)
print(summary)