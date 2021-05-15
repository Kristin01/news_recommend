import sys
import numpy as np
import pandas as pd
import re
import gc

from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from tqdm import tqdm
tqdm.pandas()

from util import text_to_wordlist
from util import text2vec

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text

from os import listdir
from os.path import isfile, join
import json

import sagemaker
from sagemaker import get_execution_role
from sagemaker import KMeans
sess = sagemaker.Session()
bucket = sess.default_bucket()

source_dir = sys.argv[1]
source_files = [join(source_dir, f) for f in listdir(source_dir) if isfile(join(source_dir, f))]
news_jsons = []
for f in source_files:
  news_jsons.append(json.loads(open(f).read()))
news_df = pd.json_normalize(news_jsons)
news_df['ori_text'] = news_df[['title', 'body']].agg(' '.join, axis=1)
news_df['words'] = news_df.ori_text.progress_apply(text_to_wordlist)

## Load Google pretrained model
# wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
# gzip -d GoogleNews-vectors-negative300.bin.gz
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
def gtext2vec(text):
    return text2vec(model,text)
news_df['vectors'] = news_df.words.progress_apply(gtext2vec)

## Clustering and generating scatter
X = np.concatenate(news_df['vectors'].values)

## run sagemaker kmeans
role = get_execution_role()
num_clusters = 10
kmeans = KMeans(
    role=role,
    train_instance_count=1,
    train_instance_type="ml.m5.4xlarge",
    output_path="s3://" + bucket + "/news_kmeans/",
    k=num_clusters,
)
kmeans.fit(kmeans.record_set(X))

## deploy sagemaker kmeans endpoint
kmeans_predictor = kmeans.deploy(initial_instance_count=1, instance_type="ml.t2.medium")
news_df['cluster'] = kmeans_predictor.predict(X)

## Save News
news_df = news_df.drop(["ori_text", "words"], axis=1)
news_df.to_pickle('news_df.pkl')

## Save Model 
import pickle
pkl_filename = "model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(kmeans, file)
