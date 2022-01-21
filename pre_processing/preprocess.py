import pandas as pd
import re
import sklearn
import string
import numpy as np
import os
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pipeline import pre_process




def readData(args):

  df = pd.read_csv(args.datafile+"/"+"dataset.csv")
  pre_process_obj = pre_process(df,args.destination)
  print(args.destination)
  pre_process_obj.fit(args.destination)

  #testing the pre-processing


  
  


def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t", True, "True", "TRUE"]:
        return True
    elif value in ["false", "no", "n", "0", "f", False, "False", "FALSE"]:
        return False

    return True

def get_parser():
    """Get parser object."""
    

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )   
    
    parser.add_argument("-m", 
                        "--model_folder", 
                        dest = "destination", 
                        default = "../models/", 
                        help="folder name for storing models")
    parser.add_argument("-df", 
                        "--dataFiles", 
                        dest = "datafile", 
                        default = "../dataset/processed/", 
                        help="folder name for extracted data")
    
    
    return parser

if __name__ == "__main__":
  args = get_parser().parse_args()
  readData(args)



    
  