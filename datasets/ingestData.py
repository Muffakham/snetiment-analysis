#data Ingestion
import pandas as pd
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def storeData(dataset,dataSplit,procecssedData):
  n = {
  "text": list(dataset.keys()),
  "labels": list(dataset.values())
  }

  if not os.path.isdir(procecssedData):
        os.makedirs(procecssedData)
  filename = dataSplit+".csv"
  df = pd.DataFrame.from_dict(n)
  df.to_csv(procecssedData+'/'+filename, index=False)


def ingestData(rawData, procecssedData):
  #reading data from the raw files
  with open(rawData+'/datasetSentences.txt') as f:
    lines = f.readlines()

  with open(rawData+'/dictionary.txt') as f:
    lines_phrases = f.readlines()

  with open(rawData+'/datasetSplit.txt') as f:
    lines_split = f.readlines()

  with open(rawData+'/sentiment_labels.txt') as f:
    lines_labels = f.readlines()


  #storing the sentences as keys and ids as values for faster search
  #ex: {'Disturbing and brilliant documentary .': '492'}
  sentences = {} 
  for i in lines[1:]:
    idx,sentence = i.split("\t")
    s = sentence.split("\n")
    sentences[s[0]] = idx

  #storing the phrases along with their ids
  #stroed as {id: text}
  phrases = {}


  #storing all the attributes of data, sentence id, phrase id, label, senetce
  #stored as {sentence id: sentence}
  """ex: 
  {'10078': {'label': 0,
    'phrase_id': '181920',
    'sentence': 'A prolonged extrusion of psychopathic pulp .'}
  }"""
  sent_ids = {} 
  for i in lines_phrases:
    phrase, idx = i.split("|")
    idx = idx.split("\n")[0]
    if sentences.get(phrase, None) != None:
      phrases[idx] = phrase
      sent_ids[sentences[phrase]] = phrase
      t = {
        'phrase_id': idx,
        'sentence': phrase
      }
      sent_ids[sentences[phrase]] = t



  dataset = {}
  #stroing senetences along with their labels
  #the labels are calculated based on sentiment score
  """if score < 0.4, label = 0(negative)
    if score > 0.4 and < 0.6, label = 1(neutral)
    if score > 0.6, label = 2(positive)
  """
  for i in lines_labels[1:]:
    idx, score = i.split("|")
    if phrases.get(idx,None) == None:
      continue
    else:
      score = float(score.split("\n")[0])
      if score <= 0.4:
        label = 0
        dataset[phrases[idx]] = label
      elif score > 0.4 and score <= 0.6:
        label = 1
        dataset[phrases[idx]] = label
      else:
        label = 2
        dataset[phrases[idx]] = label

      t = sent_ids[sentences[phrases[idx]]]
      t['label'] = label
      sent_ids[sentences[phrases[idx]]] = t

  train, val, test, dataset = {}, {}, {}, {}
  for i in lines_split:
    #print(i)
    idx, splitNum = i.split(",")
    splitNum = splitNum.split("\n")[0]
    if sent_ids.get(idx, None) != None:
      dataVal = sent_ids[idx]
      if splitNum == '1':
        train[dataVal['sentence']] = dataVal['label']

      elif splitNum == '2':
        val[dataVal['sentence']] = dataVal['label']

      else:
        test[dataVal['sentence']] = dataVal['label']
      dataset[dataVal['sentence']] = dataVal['label']

  storeData(train, "train", procecssedData)
  storeData(val, "val", procecssedData)
  storeData(test, "test", procecssedData)
  storeData(dataset, "dataset", procecssedData)


def get_parser():
    """Get parser object."""
    

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-d", 
                        "--destination_folder", 
                        dest = "destination", 
                        default = "../datasets/processed/", 
                        help="folder name for storing extracted data",
                       )
    parser.add_argument("-r", 
                        "--raw_data", 
                        dest = "rawFiles", 
                        default = "../datasets/raw/", 
                        help="folder name containing raw data files",
                       )
    
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    ingestData(args.rawFiles, args.destination)    


