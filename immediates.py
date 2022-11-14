import pickle
import pandas as pd
from pandas import DataFrame
import json
import numpy as np

name = 'wimcor_test_literal'
loc = 'wimcor_test_literal_pos'
base_length = 10
out = []

pos = pickle.load(open('contextWords/wimcor/preps/'+loc+'.pkl', 'rb'))
sents = pickle.load(open('contextWords/wimcor/preps/'+name+'.pkl', 'rb'))

left, right = [], []

for l,p in zip(sents,pos):
    left = l[:p]
    right = l[p+1:]

    while len(left) < base_length:
        left.insert(0,'0.0')

    while len(left)> base_length:
        del left[0]

    while len(right)< base_length:
        right.append('0.0')

    while len(right)> base_length:
        del right[len(right)-1]

    out.append((left,right))

print(out)

pickle.dump(out, open("baseline/wimcor/" + name + "_base10.pkl", "wb"))























# def immediate(theList, location):
#
#     l = theList
#     position = location
#
#     left = []
#     right = []
#     if location == 0:
#         while len(left) < base_length:
#             left.append('0.0')
#     else:
#         while position > 0:
#             left.insert(0, l[position-1])
#             position = position-1
#
#         while len(left) < base_length:
#             left.insert(0, '0.0')
#
#     position = location
#     while position < len(l)-1:
#             position = position + 1
#             right.append(l[position])
#
#     while len(right)< base_length:
#             right.append('0.0')
#
#     out.append((left,right))

#----------------------------------
