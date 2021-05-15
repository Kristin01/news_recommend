import sys
import os
import pandas as pd
import pickle
from gensim.models import KeyedVectors
import redis
import json
from util import text2vec
import time
import boto3

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

PREDICT_Q="predict_q"
r = redis.Redis()

print("Loading Google pre-trained model")
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Load from file
print("define sagemaker K-means endpoint")
def sagemaker_kmeans_predict(vec):
  payload = ",".join(vec)
  response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                        ContentType='text/csv',
                                        Body=payload)
  print("predict: {}".format(response))
  return response

print("Loading news data source")
pkl_filename = "news_df.pkl"
with open(pkl_filename, 'rb') as file:
    news_df = pickle.load(file)

def infer_cluster(text):
  vec = text2vec(model, text)
  return [sagemaker_kmeans_predict(vec), vec]

def recommend_news(text):
  [predicted_clustor, vec] = infer_cluster(text)
  recommends = pd.DataFrame([])
  news_ids = []
  for index, row in news_df.iterrows():
    if row["cluster"] == predicted_clustor:
      recommends = recommends.append(row)
      # TODO: replace with news id
      news_ids.append(str(row["title"]))
      if len(recommends) > 10:
        break
  return str(predicted_clustor[0]), news_ids

print("Recommendation start up complete")
while True:
    rid = r.lpop(PREDICT_Q)
    if rid != None:
        print(rid)
        text = r.get(rid).decode("utf-8")
        rid = rid.decode("utf-8")
        print(text)
        # Calculate the accuracy score and predict target values
        [cluster, news_ids] = recommend_news(text)
        print(cluster)
        print(news_ids)
        res = json.dumps({"cluster": cluster, "news_ids": news_ids})
        r.set(rid + "res", res)
    time.sleep(0.1)

