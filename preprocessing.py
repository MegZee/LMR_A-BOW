import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
import spacy
from spacy import lemmatizer
import operator

nlp= spacy.load('en_core_web_sm')


with open('contextWords/relocar/preps/relocar_train_metonymic.pkl','rb') as pickle_file:
    doc1= pickle.load(pickle_file)
    pickle_file.close()

with open('contextWords/relocar/preps/relocar_train_literal.pkl','rb') as pf:
    doc2= pickle.load(pf)
    pf.close()

doc1 = [item for sublist in doc1 for item in sublist]
doc2 = [item for sublist in doc2 for item in sublist]

stopwords = ['what', 'who', 'is', 'a', 'at', 'is', 'he',
             'the',',','.','and','(',')','that','was'
            ,'of','as','which','his','were','also','-','its']

filteredDoc1= [word for word in doc1 if word.lower() not in stopwords]
filteredDoc2= [word for word in doc2 if word.lower() not in stopwords]


keys=[]
values=[]
str_list=[]

def sort_Count(l):
    unique_words = set(l)

    for word in unique_words:
        keys.append(word)
        values.append(l.count(word))

    words_dict = {}

    for i in range(len(keys)):
         words_dict[keys[i]] = values[i]
    sorteddict = sorted(words_dict.items(),
                        key=operator.itemgetter(1),
                        reverse=True)
    words_dict=dict(sorteddict)
    return print(words_dict)

print('Metonymy Words:')
sort_Count(filteredDoc1)
print("Literal Words:")
sort_Count(filteredDoc2)