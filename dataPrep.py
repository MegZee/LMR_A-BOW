import pickle
import pandas as pd
from pandas import DataFrame
import json
import numpy as np
name = 'wimcor_test'
with open('dataset/' + name + '.json') as f:
    data = json.load(f)
name_set = []
label_set = []
pos_set = []

for instance in data:
    name_set.append(instance['sentence'])
    label_set.append(instance['label'])
    pos_set.append(instance['pos'][0])

df = pd.DataFrame(list(zip(name_set, label_set, pos_set)),
                  columns=['Sentence', 'Label', 'Pos'])


df_lit = pd.DataFrame(df[df['Label'] == 0])
df_met = pd.DataFrame(df[df['Label'] == 1])

lit_train = df_lit['Sentence'].tolist()
met_train = df_met['Sentence'].tolist()

lit_train_pos = df_lit['Pos'].tolist()
met_train_pos = df_met['Pos'].tolist()



# with open('contextWords/semeval/preps/'+name+'_literal.pkl', 'wb') as fileLit:
#     pickle.dump(lit_train, fileLit)
#     fileLit.close()
#
# with open('contextWords/semeval/preps/'+name+'_literal_pos.pkl', 'wb') as fileLit:
#     pickle.dump(lit_train_pos, fileLit)
#     fileLit.close()
# #
# with open('contextWords/semeval/preps/'+name+'_metonymic.pkl', 'wb') as fileMet:
#     pickle.dump(met_train, fileMet)
#     fileMet.close()
# #
# with open('contextWords/semeval/preps/'+name+'_metonymic_pos.pkl', 'wb') as fileMet:
#     pickle.dump(met_train_pos, fileMet)
#     fileMet.close()
#
# #--------------------------------------------------------
f3 = pickle.load(open('contextWords/wimcor/preps/'+name+'_metonymic_pos.pkl', 'rb'))

print(len(f3))