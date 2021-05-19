# news_recommend
recommend news with Amazon SageMaker

## Setup Environment

1. Create an Notebook in Sagemaker (instance type c4.2xlarge, disksize 20GB) and run
2. upload setup-aws.ipynb to the notebook and run all cells

It will download the code from github and download Google pre-trained word2vec model.

## Training

- Open train/train_aws.ipynb  and run all

It will train a brand new model with Sagemaker and start an inference endpoint.
You will get an inference endpoint from Sagemaker. Note the inference endpoint from SageMaker-Inference (`inference_endpoint`).
You will also get news_df.pkl which contains the source news and will be used in inference web service.
It is already saved in s3 with path `nyu-cc-final-recommend-news/news_df.pkl`

## Start Inference Web Service

1. create an EC2 (instance type c4.2xlarge, disksize 20GB, ubuntu, open 80 port, require external ip `external_ip`)
2. run `python -m pip install -r requirements.txt` to install dependences
3. check if redis-server is installed, otherwise run `apt-get install redis-server`
3. run `git clone https://github.com/Kristin01/news_recommend.git`
4. run `sudo python app.py &` to start the inference web service

## Start Inference Service

- run `sudo ENDPOINT_NAME=${inference_endpoint} python predict_aws.py` to start inference service

## Validation

1. At local chrome web brower, install RestMan
2. In the Host of RestMan, put the `external_ip`
3. In the Action of RestMan, put `POST`
4. In the body of RestMan, put `text = Summary is coming` 
5. run and you should see recommends news in the response