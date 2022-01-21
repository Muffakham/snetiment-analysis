from flask import Flask
from flask_ngrok import run_with_ngrok
from flask import request
from flask_cors import CORS, cross_origin
from pre_processing.pipeline import pre_process
import pickle
import pandas as pd
import numpy as np



def predictions(data):
  data = data["data"]


  labelToSentiment = {0:"Negative", 1:"Neutral", 2:"Positive"}#dictionary to map the sentimnet label
  n = {'text': [data]}
  df = pd.DataFrame(n) #converting the given twxt to dataframe
  #print(df)

  #creating data pre processing pipeline
  p = pre_process(df,"./models/")
  ed = p.transform()


  #load the Logistic regression model
  with open("./models/lr_model.pk", "rb") as f: model = pickle.load(f)
  pred = model.predict(ed)#prediction from the model
  
  pred = pred.tolist()
  pred = [labelToSentiment[i] for i in pred]#converting the sentiment index to its label


  return pred


from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

app = Flask(__name__)

CORS(app)

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def run_app():
    return "/pred/ for Predictions"


@app.route('/pred/', methods=['GET', 'POST'])
def pred_app():        
    jsonData = request.get_json(force=True)    
    print(jsonData)
    return preds(jsonData)


if __name__ == '__main__':
    app.run()
