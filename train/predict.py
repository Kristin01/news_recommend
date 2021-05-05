import sys
import pickle

from util import text_to_wordlist
from util import text2vec

def infer_cluster(model, text):
  vec = text2vec(text)
  return [model.predict(vec), vec]

news_path = sys.argv[1].read()
text = open(news_path)
pkl_filename = "model.pkl"
# Load from file
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

# Calculate the accuracy score and predict target values
[cluster, vec] = infer_cluster(model, text)
print(cluster)
print(vec)
