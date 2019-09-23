import pandas as pd
import spacy
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

amazon_sm = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')
reviews = amazon_sm['reviews.text'].tolist()
ratings = amazon_sm['reviews.rating'].tolist()
# ratings = [1, 2, 3, 2, 2, 3, 4, 4, 5]
# nlp_blank = spacy.load('./model-blank/')
# nlp_en = spacy.load('./model-en-lg/')
# cat_blank = nlp_blank.get_pipe("textcat")
# cat_en = nlp_en.get_pipe("textcat")
# blank_scores, _ = cat_blank.predict([nlp_blank(r) for r in reviews])
# en_scores, _ = cat_en.predict([nlp_en(r) for r in reviews])
# blank_predict = blank_scores.argmax(axis=1)+1
# en_predict = en_scores.argmax(axis=1)+1
# blank_acc = accuracy_score(ratings, blank_predict)
# en_acc = accuracy_score(ratings, en_predict)
# blank_loss = log_loss(ratings, blank_scores, labels = [1,2,3,4,5])
# en_loss = log_loss(ratings, en_scores, labels = [1,2,3,4,5])
# with open('amazon_review_rating_classification_test.text', 'w') as f:
#     f.write('the accuracy for the blank model is {:2.3%}\n'.format(blank_acc))
#     f.write('the accuracy for the en_core_lg model is {:2.3%}\n'.format(en_acc))
#     f.write('the log loss for the blank model is {}\n'.format(blank_loss))
#     f.write('the log loss for the en_core_lg model is {}\n'.format(en_loss))


nlp_starspace = spacy.load('./pretrained-model/1000')
cat_starspace = nlp_starspace.get_pipe("textcat")
starspace_scores, _ = cat_starspace.predict([nlp_starspace(r) for r in reviews])
starspace_predict = starspace_scores.argmax(axis=1)+1
starspace_acc = accuracy_score(ratings, starspace_predict)
starspace_loss = log_loss(ratings, starspace_scores, labels = [1,2,3,4,5])
with open('amazon_review_rating_classification_test.text', 'a+') as f:
    f.write('the accuracy for the pretrained model is {:2.3%}\n'.format(starspace_acc))
    f.write('the log loss for the pretrained model is {}\n'.format(starspace_loss))