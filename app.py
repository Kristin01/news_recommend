from flask import Flask
from flask import request
import redis
import time
import config
import uuid

app = Flask(__name__)
r = redis.Redis()


@app.route('/')
def index():
    return 'Welcome to News Recommendation!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        rid = str(uuid.uuid4())
        text = request.form['text']
        r.set(rid, text)
        r.rpush(config.PREDICT_Q, rid)
        for i in range(50):
            res = r.get(rid + "res")
            if res == None:
                time.sleep(0.1)
            else:
                return res, 200
        return "server is busy, please try again", 503
    else:
        return "Bad request", 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
