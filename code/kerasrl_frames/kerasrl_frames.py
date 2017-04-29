import os
import json
import nltk
import gensim
import numpy as np
from gensim import corpora, models, similarities
import pickle
import csv
import json
from pandas.io.json import json_normalize
import numpy as np

model = gensim.models.Word2Vec.load('data/word2vec.bin');
file = open('data/frames.json');
x = json.load(file)
x_normalize = json_normalize(x)
turns = x_normalize['turns']

txt = []
for i in turns[1:100]:
    #print(i)
    cc = json_normalize(i)
    #print(cc['text'])
    txt.append(cc['text'])
	
txt_array = np.array(txt)
cor= txt_array

x=[]
y=[]


for i in range(len(cor)):
    for j in range(len(cor[i])):
        if j<len(cor[i])-1:
            x.append(cor[i][j]);
            y.append(cor[i][j+1]);

tok_x=[]
tok_y=[]
for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))
    
    

sentend=np.ones((300,),dtype=np.float32) 

vec_x=[]
for sent in tok_x:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_x.append(sentvec)
    
vec_y=[]
for sent in tok_y:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_y.append(sentvec)           
    
    
for tok_sent in vec_x:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_x:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)    
            
for tok_sent in vec_y:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_y:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)             
            
            
with open('conversation.pickle','w') as f:
    f = open('conversation.pickle','wb')
    pickle.dump([vec_x,vec_y],f)    