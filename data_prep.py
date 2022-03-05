import nltk
import os
import pandas as pd

dir_text = 'BBC News Summary/News Articles/tech/'
dir_summ = 'BBC News Summary/Summaries/tech/'

arr = os.listdir(dir_text)
for news in arr:
    with open(dir_text + news) as f:
        title = f.readline()
        text = f.read()
    with open(dir_summ + news) as f:
        summ = f.read()
    
    summ = summ.replace('.', '. ')
    text = text.replace('\n', ' ')
    text = text.lstrip()

    summ = summ.replace('"', '')
    text = text.replace('"', '')
    # print(text)

    sentence_list_text = nltk.sent_tokenize(text)
    sentence_list_summ = nltk.sent_tokenize(summ)

    err = False
    for sent_sum in sentence_list_summ:
        if sent_sum not in sentence_list_text:
            # print(news, sent_sum)
            err = True
    if err:
        continue
            
    data=[]
    for sent_text in sentence_list_text:
        if sent_text in sentence_list_summ:
            label=1
        else:
            label=0
        
        data.append([sent_text, label])
    
    df = pd.DataFrame(data, columns = ['Sentence', 'Label'])
    df.to_csv('preprocessed_tech/%s'% news) 
    # exit()



  