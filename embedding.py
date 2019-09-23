import requests
import urllib.parse
import json
import time
import pandas as pd
import spacy
import re
from pdb import set_trace
import os

amazon_lg = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')
amazon_lg_cat = amazon_lg['categories'].values.tolist()
amazon_lg_review = amazon_lg['reviews.text'].values.tolist()
labels = []
for cat in amazon_lg_cat:
    labels.append(re.split(',' , cat))
    
reviews = []
for review in amazon_lg_review:
    reviews.append(re.sub('[^a-zA-Z0-9_\s]+', '', review))
    
labels_reform = []
for label in labels:
    label_reform = []
    for i, l in enumerate(label):
        l = l.replace(' & ', '&')
        l_indexed = ['label_' + str(i+1) + '_' + w for w in l.split()]
        label_reform.append(l_indexed)
        
    labels_reform.append(label_reform)
    
with open('amazon_lg_review_label.txt', 'w') as f:
    for reviews, label_k in zip(reviews, labels_reform):
        f.write("%s\t" % reviews)
        for l in label_k[:-1]:
            f.write('%s\t' % ' '.join(l))
        
        f.write('%s\n' % ' '.join(label_k[-1]))
        
os.system("/home/baggiohbs_1989/Starspace/starspace train -trainFile amazon_lg_review_label.txt -model amazon_lg_review -fileFormat labelDoc -dim 96 -epoch 20")