import spacy
import numpy as np
from scipy import spatial
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from spacy import lemmatizer
nlp = spacy.load("en_core_web_sm")

#---The Extracted words after Cleaning The Data----------

#-------------------SEMEVAL--------------------------

# metonymy= nlp(u'be on united new will would but two against may if first say aid other '
#               u'give world hold group during provide score defence good ')
#
#
# literal= nlp(u'states south year country all kingdom there west when see government '
#              u'northern state between into include some most company time national '
#              u'more high north president part ')

#-------------------CONLL--------------------------

# metonymy= nlp(u'say war against world after cup first friday but match beat group '
#               u'will soccer day united one minister win government take out european '
#               u'international last tell test second president trade qualifier call '
#               u'former play ')
#
#
# literal= nlp(u'new year police up between official south week state when visit west '
#              u'city people into capital hold run market meet talk since peace states '
#              u'bank northern town more kill country million mile national airport '
#              u'go live north east ')

#-----------------RELOCAR-----------------------

metonymy= nlp(u'against world play win '
              u'during first team represent make take soviet '
              u'independence union match nation tournament'
              u' diplomatic do final own debut help ')


literal= nlp(u'south part kingdom city state country area '
             u'east north most since locate west region '
             u'large western europe district find '
             u'eastern population empire place province')

# words that don't need to have similar vectors aka: "Excluded Words"

uniqueMet=['for','by','with','have','over','new','us']
uniqueLit=['in','to','from','on','be','where']


met_list=[]     #because GloVe Cannot Work With Spacy
lit_list=[]

for i in metonymy:
    met_list.append(i.text)

for j in literal:
    lit_list.append(j.text)

#----------------------------


#--Extracting GloVe Vectors------------------

embeddings_dict = {}
with open("glove.6B.50d/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
    f.close()
#------------------------------------------


#---Function To Extract The Closest Words ---------------

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(),
                  key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

#----------------------------------



#----Create a List with The Extracted Words + eliminate redundancy-------

def extended_Glove_Words(TheList, excludedWords):
    uniqueWords=excludedWords
    for word in TheList:
        terms = (find_closest_embeddings(embeddings_dict[word])[0:6])
        for term in terms:
            if term not in uniqueWords:
                uniqueWords.append(term)
    return uniqueWords
#--------------------------------------------


#------------ Main-------------
metFeaturesReLocaR= extended_Glove_Words(met_list, uniqueMet)
litFeaturesReLocaR= extended_Glove_Words(lit_list, uniqueLit)
print('Metonymy words: ',metFeaturesReLocaR)
print('literal words: ',litFeaturesReLocaR)


# with open("Features extracted/MetFeaturesConll.pkl", "wb") as pickle_file:
#       pickle.dump(metFeaturesReLocaR, pickle_file)
#       pickle_file.close()
# #
# with open("Features extracted/LitFeaturesConll.pkl", "wb") as pickle_file:
#     pickle.dump(litFeaturesReLocaR, pickle_file)
#     pickle_file.close()

with open("Features extracted/MetFeaturesConll.pkl", 'rb') as Pf:
    doc = pickle.load(Pf)
    Pf.close()

print('\n \n \n ',doc)











