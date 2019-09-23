import csv
import pandas as pd
import json
import srsly
amazon_lg = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')
amazon_lg_review = amazon_lg['reviews.text'].values.tolist()
raw = [{'text': review} for review in amazon_lg_review]
srsly.write_jsonl("./text.jsonl", raw)