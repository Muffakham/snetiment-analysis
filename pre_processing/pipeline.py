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


class pre_process():
  """
  this is the pipeline class that is responsible for 
    - cleaning data
    - training and storing TF_IDF adn PCA models
    - transforming the cleaned data into embeddings
  """

  def __init__(self, dataset, model_dir):

    #downloading and initializong the stopwords and the lemmeatixer
    nltk.download('wordnet')
    self.lemmatizer = WordNetLemmatizer()
    nltk.download('stopwords')
    self.stop_words = stopwords.words("english")
    self.df = dataset
    self.model_dir = model_dir
    

  def initializeModels(self):

    """
    intializing the TF-IDF and PCA models
    """
    self.vectorizer = TfidfVectorizer(analyzer='word',
                                stop_words='english',
                                ngram_range=(1,3), 
                                max_df=0.9, 
                                use_idf=True, 
                                smooth_idf=True, 
                                max_features=1000)
    
    self.pca = PCA(n_components=0.95)

  def removeSpecialCharacters(self, text):
    """
    removes the special characters in the string like punctuations, symbols, etc
    uses regular expressions
    """
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    return text.strip()

  def removeStopWords(self, text):
    """
    removes the stop wrods from the given text
    """
    text = " ".join([i for i in text.split(" ") if i not in self.stop_words])
    return text

  def lemmatization(self, text):
    """
    converts the words in a given texts into their base form (lemma's)
    """
    text = " ".join([self.lemmatizer.lemmatize(i) for i in text.split(" ")])
    return text

  def storeModels(self, model, modelDir, modelname):
      """
      stores the gvien model as a pickle file
      inputs - model, model name, directory where the model is stored
      """
      
      if not os.path.isdir(modelDir):
            os.makedirs(modelDir)
      filename = modelDir+"/"+modelname+".pk" 
      with open(filename,'wb') as f: pickle.dump(model, f)
      f.close()

  def loadModels(self, modelDir, modelname):
      """
      loads the gvien model from a pickle file
      inputs - model name, directory where the model is stored
      """
      
      if not os.path.isdir(modelDir):
            print("No such directory as - "+modelDir)
            print("please transform the data inorder to store the models .. ")
      
      filename = modelDir+"/"+modelname+".pk" 
      with open(filename,'rb') as f: 
        model = pickle.load(f)
        return model
      

  def tf_idf(self, dataset, predict): 

    """
    embeds the given text into TF_IDF embeddings
    Inputs: data - the dataset in dataframe format
            predict - booloean value, if false, tf-idf model is trained on the dataset, else it just converts the given dataset
    """
    #print(dataset)
    if predict:
      #tfIdfMat = self.vectorizer.fit(dataset['text'].tolist())
      tfIdfMat = self.vectorizer.transform(dataset['text'].tolist())
    else:
      self.vectorizer.fit(dataset['text'].tolist())
      tfIdfMat = self.vectorizer.transform(dataset['text'].tolist())
      #tfIdfMat  = self.vectorizer.fit_transform(dataset['text'].tolist())
    #print(tfIdfMat)
    return tfIdfMat

  def dimensionality_reduction(self, matrix, predict):
    """
    embeds the given text into PCA embeddings
    Inputs: data - the dataset in dataframe format
            predict - booloean value, if false, PCA model is trained on the dataset, else it just converts the given dataset
    """
    #print(matrix)
    if predict:
      tfIdfMat_reduced = self.pca.transform(matrix.toarray())
    else:
      self.pca.fit(matrix.toarray())
      tfIdfMat_reduced = self.pca.transform(matrix.toarray())

    #print(tfIdfMat_reduced)
    return tfIdfMat_reduced


  def storeData(self,dataFrame,embedData,reducedData,labels,filename,procecssedData):
    
    """
    this function stores the pre-processed data into pickle files.
    it stores the cleaned data, tf-idf embeddings, reduced data (PCA output), and dataset labels.

    """

    #print(os.path(procecssedData))
    if not os.path.isdir(procecssedData):
          os.makedirs(procecssedData)


    embedFileName = procecssedData+"/"+filename+"_embed.pk"
    reducedDataFileName = procecssedData+"/"+filename+"_embed_reduced.pk"
    labelFileName = procecssedData+"/"+filename+"_labels.pk"
    filename = filename+"_cleaned.csv"

    
    self.df.to_csv(procecssedData+'/'+filename, index=False)
    
    
    with open(embedFileName,'wb') as f: pickle.dump(embedData, f)
    f.close()
    with open(reducedDataFileName,'wb') as f: pickle.dump(reducedData, f)
    f.close()
    with open(labelFileName,'wb') as f: pickle.dump(labels, f)
    f.close()

  def fit(self, model_dir):

    """
    this function is the pipeline function that cleans the data, transforms it 
    and stores it along with storing and training the TF_IDF and PCA models
    """
    self.df['text'] = self.df['text'].str.lower()
    self.df["text"] = self.df["text"].apply(lambda text: self.removeSpecialCharacters(text))
    self.df["text"] = self.df["text"].apply(lambda text: self.removeStopWords(text))
    self.df["text"] = self.df["text"].apply(lambda text: self.lemmatization(text))
    self.initializeModels()
    ed = self.tf_idf(self.df, predict=False)
    red = self.dimensionality_reduction(ed, predict=False)
    labels = self.df['labels'].tolist()
    self.storeData(self.df,ed,red,labels,"dataset","./datasets/preProcessed")
    self.storeModels(self.vectorizer, model_dir, "tfidf")
    self.storeModels(self.pca, model_dir, "pca")


  def transform(self):

    """
    this is a pipeline funciton used while predicting new data.
    It follows the same approach as fit, except it does not train TF_IDF and PCA models
    it uses the trained models to tramsform the data.
    """
    self.df['text'] = self.df['text'].str.lower()
    self.df["text"] = self.df["text"].apply(lambda text: self.removeSpecialCharacters(text))
    self.df["text"] = self.df["text"].apply(lambda text: self.removeStopWords(text))
    self.df["text"] = self.df["text"].apply(lambda text: self.lemmatization(text))

    self.vectorizer = self.loadModels(self.model_dir, "tfidf")
    self.pca = self.loadModels(self.model_dir, "pca")

    #print(self.df)
    ed = self.tf_idf(self.df, predict=True)
    red = self.dimensionality_reduction(ed, predict=True)

    return red