import os
import pandas as pd
import re

total = [0,0,0,0,0]

dirs = ['tech/', 'sport/', 'politics/', 'entertainment/', 'business/']
dir_text = 'BBC News Summary/News Articles/'
dir_summ = 'BBC News Summary/Summaries/'

for d_i, d in enumerate(dirs):
    text_files = "%s%s" % (dir_text,d)
    summ_files = "%s%s" % (dir_summ,d)

    arr = os.listdir(text_files)
    for news in arr:
        with open(text_files + news) as f:
            title = f.readline()
            text = f.read()
        with open(summ_files + news) as f:
            summ = f.read()

        summ = re.sub("\"|-","",summ)
        text = re.sub("\"|-","",text)

        summ = re.sub("\s\s"," ",summ)
        text = re.sub("\s\s"," ",text)

        sentence_list_summ = re.split('(?=\.[A-Z])\.',summ)
        sentence_list_text = re.split('\.(?<=\.)[\s\n]',text)

        sentence_list_text_cln = []
        sentence_list_summ_cln = []

        for t in sentence_list_text:
            t = re.sub('\n', '', t)
            t = t.strip()
            if t=='':
                continue
            if t[-1]=='.':
                t=t[:-1]
            sentence_list_text_cln.append(t)
        for s in sentence_list_summ:
            s = re.sub('\n', '', s)
            s = s.strip()
            if s=='':
                continue
            if s[-1]=='.':
                s=s[:-1]
            sentence_list_summ_cln.append(s)

        err = False
        for sent_sum in sentence_list_summ_cln:
            if sent_sum not in sentence_list_text_cln:
                print(d, news, sent_sum)
                # exit()
                err = True
        if err:
            continue

        if len(sentence_list_text_cln)<3:
            continue
                
        data=[]
        for sent_text in sentence_list_text_cln:
            if sent_text in sentence_list_summ_cln:
                label=1
            else:
                label=0
            
            data.append([sent_text, label])
        
        df = pd.DataFrame(data, columns = ['Sentence', 'Label'])
        df.to_csv('prep/%s%s'% (d, news)) 
        total[d_i] += 1

        # print(sentence_list_text_cln)
        # exit()
print(total)    