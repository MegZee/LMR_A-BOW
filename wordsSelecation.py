import pickle
import spacy
from spacy.lang.en import English
from pprint import pprint
import re

en_nlp = spacy.load('en_core_web_sm')

tokenizer = English(parser=False)

featuresPath = 'LitFeaturesConll.pkl'
with open('Features extracted/' + featuresPath, 'rb') as pf:
    gloveFeatures1 = pickle.load(pf)
    pf.close()
with open('Features extracted/LitFeaturesReLocaR.pkl', 'rb') as pf1 :
    gloveFeatures2 = pickle.load(pf1)
    pf1.close()
with open('Features extracted/LitFeaturesSemeval.pkl','rb') as pf2:
    gloveFeatures3 = pickle.load(pf2)
    pf2.close()

gloveFeatures = set(gloveFeatures1+gloveFeatures2+gloveFeatures3)

while '.' in gloveFeatures: gloveFeatures.remove('.')

base = 3
nameTxt = 'wimcor_train_literal'
namePos = 'wimcor_train_literal_pos'

# txt = open("dataset/"+nameTxt+".txt", mode="r", encoding="utf-8")

txt = pickle.load(open("contextWords/wimcor/preps/"+nameTxt+".pkl", 'rb'))
loc = pickle.load(open("contextWords/wimcor/preps/"+namePos+".pkl", 'rb'))

def contexts(sentences, position):
    intersect = []
    mutualWords = []

    for sent,pos in zip(sentences, position):
        for word in sent:
            if word.lower() in gloveFeatures \
                    and word.lower() not in intersect:
                intersect.append(word.lower())

        if len(intersect)>1:
            mutualWords.append(intersect)

            intersect = []
        else:
            left = sent[:pos]
            right = sent[pos + 1:]

            while len(left) < base:
                left.insert(0, '0')

            while len(left) > base:
                del left[0]

            while len(right) < base:
                right.append('0')

            while len(right) > base:
                del right[len(right) - 1]
            mutualWords.append(left+right)
    # with open("contextWords/wimcor/all_words_"+nameTxt+".pkl", "wb") as pickle_file:
    #     pickle.dump(mutualWords, pickle_file)
    #     pickle_file.close()

    return pprint(mutualWords)


contexts(txt, loc)




