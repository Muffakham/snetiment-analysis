# ML pipeline for Sentiment analysis.

## About the Projetc
  * This is an implementation of a simple ML pipeline to build a Softmax classifier for sentiment analysis.
  * The dataset used is the SST dataset, comprising of arounf 11K samples.
  * The classification model classifies a given text into threee sentiments:
    * Negative
    * Neutral
    * Positive
  * This pipelime contains the code for:
    * Ingesting the data - once the raw files of the dataset are entered into /datasets/raw, the pipelines extracts the data and stores it in .csv format.
    * Pre-processing - The data is cleaned (special character removal, stopwords removeal, lowercasing, lemmatization).
    * Embedding the data - The text data is converted into TF-IDF embeddings, followed by dimensionality reducntion using PCA.
    * Training the model -  A softmax classifier model is trained for this multi class classification task. 
    * Deploying the model - once the model is trained and tested, it is deployed using python flask. 
## Project Structure

      ├── datasets                    # contains all files realted to the data and data ingestion script
      │   ├── raw                     # contains all the raw files that make up the dataset
      │   ├── processed               # contains all the files that have been extracted from raw data files
      │   ├── preProcessed            # contains all the files after pre-processing the data
      │   └── ingestData.py           # python script to convert raw data into usable .csv file   
      ├── models                      # contains all the model pickle files and also the classification model report
      ├── training                    # contains the script to train the classification model, also houses the config file for model
      │   ├── train.py                # python script to train the logistic regression model
      │   └── config.json             # JSON file containing the options for training the model
      ├── pre_processing              # contains the data pre-processing script and the pipeline script for pre-processing
      │   ├── preprocess.py           # script to call the pre-process pipeline on the train data
      │   └── pipeline.py             # script containing the code for data-preprocessing pipeline (cleaning, transformation, embedding)
      ├── experiments                 # contains the YAML config file
      ├── main.py                     # main python file, reads the configuration from YAML file and runs various scripts
      ├── app.py                      # pyton script tp create the REST API endpoints for the model usign python Flask
      ├── requirements.txt            # contains the names of the packages required to run the project
      └── README.md                   # readme for the project
