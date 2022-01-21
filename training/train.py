from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.metrics import classification_report
import json
import pickle
import pandas as pd
import numpy as np


def train_lr(config,X_train, X_test, y_train, y_test,storagePath):
  #print(config)
  config = config['logisticRegression']
  if config["gridSearch"]:
    param_grid = {
     'penalty' : ['l1', 'l2'],
     'C' : np.logspace(-4, 4, 20),
     'class_weight': ['balanced', None],
     'solver': ['liblinear']
    }
    base_estimator = LogisticRegression()
    sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,factor=2,max_resources=40).fit(X_train, y_train)
    lr = sh.best_estimator_
  else:
    lr = LogisticRegression(penalty = config['penalty'],
                            solver = config['solver'],
                            class_weight = config['class_weight'],
                            max_iter = config['max_iter'],
                            ).fit(X_train,y_train)
    
  y_pred = lr.predict(X_test)
  #print(len(y_pred))
  #print(len(y_test))
  target_names = ['negative', 'neutral', 'positive']
  report = classification_report(y_test, y_pred, target_names=target_names,output_dict=True)
  df = pd.DataFrame(report)
  df.to_csv("./models/report.csv")
  #classification_report_csv(report)
  filename = storagePath+"/lr_model.pk"
  with open(filename,'wb') as f: pickle.dump(lr, f)
       

def train_model(config, dataFilePath, destFilePath):
  
  dataFile = dataFilePath+"/dataset_embed_reduced.pk"
  labelFile = dataFilePath+"/dataset_labels.pk"
  with open(labelFile,'rb') as f: Y = pickle.load(f)
  f.close()
  with open(dataFile,'rb') as f: X = pickle.load(f)
  f.close()
  X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=42)
  for i in config.keys():
    model = i
    model_config = config[i]
    if model == "logisticRegression":
      train_lr(config,X_train, X_test, y_train, y_test,destFilePath)

def pass_data(args):
  with open(args.config, 'r') as json_file:
    config = json.load(json_file)
  train_model(config, args.datafile, args.destination)

def get_parser():
    """Get parser object."""
    

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )   
    parser.add_argument("-d", 
                        "--destination_folder", 
                        dest = "destination", 
                        default = "../model/", 
                        help="folder for storing model files")
    parser.add_argument("-df", 
                        "--dataFiles", 
                        dest = "datafile", 
                        default = "../pre_process/data/", 
                        help="folder name for extracted data")
    parser.add_argument( "-c",
                        "--config", 
                        dest = "config", 
                        default = "../training/config.json", 
                        help="file containing the model configurations")
    
    return parser

if __name__ == "__main__":
  args = get_parser().parse_args()
  pass_data(args)

