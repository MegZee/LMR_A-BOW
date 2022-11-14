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

# doc1_str = ' '
# doc2_str = ' '
#
# doc1_str = (doc1_str.join(doc1))
# doc2_str = (doc2_str.join(doc2))

# doc1= re.sub(r'\\', '', doc1)
# doc2= re.sub(r'\\', '', doc2)
#
# doc1= re.sub(r'<.*?>', ' ', doc1)
# doc2= re.sub(r'<.*?>', ' ', doc2)


# docList1=doc1_str.split()
# docList2=doc2_str.split()

stopwords = ['what', 'who', 'is', 'a', 'at', 'is', 'he',
             'the',',','.','and','(',')','that','was'
            ,'of','as','which','his','were','also','-','its']

resultDoc1= [word for word in doc1 if word.lower() not in stopwords]
# resultDoc1=' '.join(resultDoc1)

resultDoc2= [word for word in doc2 if word.lower() not in stopwords]
# resultDoc2=' '.join(resultDoc2)

print('resultDoc1: ',resultDoc1)

keys=[]
values=[]
str_list=[]

#----------///------------

def freq(str_list):

    # str=nlp(str)
    # for token in str:
    #     str_list.append(token.lemma_)


    # gives set of unique words
    unique_words = set(str_list)

    for words in unique_words:
        keys.append(words)
        values.append(str_list.count(words))

#----------///------------

def sorting(words_dict):
        words_dict = {}
        for i in range(len(keys)):
            words_dict[keys[i]] = values[i]
        #print(words_dict)
        sorteddict = sorted(words_dict.items(), key=operator.itemgetter(1), reverse=True)
        words_dict=dict(sorteddict)
        return words_dict



if __name__ == "__main__":

    sorteddoc1=(sorting(freq(resultDoc1)))
    sorteddoc2=(sorting(freq(resultDoc2)))
    print(sorteddoc1)
    print(sorteddoc2)








