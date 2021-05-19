import sys
import os
import pandas as pd
import pickle
from gensim.models import KeyedVectors
import redis
import json
from util import text2vec
import time
from sagemaker import KMeansPredictor

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
kmeans_endpoint = KMeansPredictor(ENDPOINT_NAME)

PREDICT_Q="predict_q"
r = redis.Redis()

print("Loading Google pre-trained model")
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# define sagemaker K-means endpoint
print("define sagemaker K-means endpoint")
def sagemaker_kmeans_predict(vec):
    response = kmeans_endpoint.predict(vec)
    #print(response[0].label)
    return response[0].label.get("closest_cluster").float32_tensor.values[0]

def infer_cluster(text):
  vec = text2vec(model, text)
  return [sagemaker_kmeans_predict(vec)[0], vec]
  

print("Loading news data source")
pkl_filename = "news_df.pkl"
with open(pkl_filename, 'rb') as file:
    news_df = pickle.load(file)

def recommend_news(text):
  [predicted_clustor, vec] = infer_cluster(text)
  recommends = pd.DataFrame([])
  for index, row in news_df.iterrows():
    if row["cluster"] == predicted_clustor:
      recommends = recommends.append(row)
      if len(recommends) > 10:
        break
  return str(predicted_clustor[0]), recommends

print("Recommendation start up complete")
while True:
    rid = r.lpop(PREDICT_Q)
    if rid != None:
        print(rid)
        text = r.get(rid).decode("utf-8")
        rid = rid.decode("utf-8")
        print(text)
        # Calculate the accuracy score and predict target values
        [cluster, recommends] = recommend_news(text)
        print(cluster)
        print(recommends)
        res = json.dumps({"cluster": cluster, "news": recommends.to_json()})
        r.set(rid + "res", res)
    time.sleep(0.1)

