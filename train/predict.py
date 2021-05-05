import sys
import pickle
from gensim.models import Word2Vec, KeyedVectors

from util import text_to_wordlist
from util import text2vec

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# Load from file
with open(pkl_filename, 'rb') as file:
    kmeans_model = pickle.load(file)

def infer_cluster(text):
  vec = text2vec(model, text)
  return [kmeans_model.predict(vec), vec]

news_path = sys.argv[1]
text = open(news_path).read()
pkl_filename = "model.pkl"


# Calculate the accuracy score and predict target values
[cluster, vec] = infer_cluster(text)
print(cluster)
print(vec)
