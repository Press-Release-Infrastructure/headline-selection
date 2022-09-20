import re
import nltk
from fuzzywuzzy import fuzz
import string
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
                    
# import mysql.connector

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


#read lexisnexus company names
with open('dump_c.txt', encoding="utf8", errors='ignore') as f:
    lines = f.readlines()

company = []
array_companies = {}
year = []
for line in lines:
	line = line.replace('INSERT INTO \"cb_objects\" VALUES ', "")
	# line = line[1:-1]
	rows = line.split("),(")
	
	for row in rows:
		row = row.split(',')
		company.append(row[4].lower())
		array_companies[row[0][1:-1]] =  row[4]


#brill tagger
def train_brill_tagger(initial_tagger, train_sents, **kwargs):
    templates = [
            brill.Template(brill.Pos([-1])),
            brill.Template(brill.Pos([1])),
            brill.Template(brill.Pos([-2])),
            brill.Template(brill.Pos([2])),
            brill.Template(brill.Pos([-2, -1])),
            brill.Template(brill.Pos([1, 2])),
            brill.Template(brill.Pos([-3, -2, -1])),
            brill.Template(brill.Pos([1, 2, 3])),
            brill.Template(brill.Pos([-1]), brill.Pos([1])),
            brill.Template(brill.Word([-1])),
            brill.Template(brill.Word([1])),
            brill.Template(brill.Word([-2])),
            brill.Template(brill.Word([2])),
            brill.Template(brill.Word([-2, -1])),
            brill.Template(brill.Word([1, 2])),
            brill.Template(brill.Word([-3, -2, -1])),
            brill.Template(brill.Word([1, 2, 3])),
            brill.Template(brill.Word([-1]), brill.Word([1])),
            ]
      
    # Using BrillTaggerTrainer to train 
    trainer = brill_trainer.BrillTaggerTrainer(
            initial_tagger, templates, deterministic = True)
      
    return trainer.train(train_sents, **kwargs)



#load lexisnexius headlines
with open('dump.txt') as f:
    lines = f.readlines()

accuracy = []
num = []
non_company = []
year = []
count_too_short = 0 
count_no_AQ = 0
count = 0 
array = []
companies_appearing = []
for line in lines:
  line = line[1:-1]
  rows = line.split("),(")
  for row in rows:
    count += 1
    row = re.split("','", row)
    row[0] = row[0].split("'")[-1]
    row[1] = row[1].split("'")[0]
    headline = row[-3].translate(str.maketrans('','', string.punctuation))
    if len(headline.split(" ")) < 3 or len(headline.split(" ")) > 12:
      count_too_short += 1
      continue
    for i in range(10):
      try:
        if int(row[i][:4]) < 2022 and int(row[i][:4]) > 1990:
          year.append(int(row[i][:4]))
      except:
        continue
      
    found_AQRE = 0
    found_AQRR = 0

    #replace company names with aquirer or aquiree
    for word in headline.split(" "):
      try:
        companies_appearing.append(row[1])
        companies_appearing.append(row[0]) 
        if fuzz.token_set_ratio(word.lower(), array_companies[row[1]].lower()[1:-1]) > 50:
          if found_AQRE == 0:
            headline = headline.replace(word, "aquiree")
          else:
            headline = headline.replace(word, "")
          found_AQRE = 1
        if fuzz.token_set_ratio(word.lower(), array_companies[row[0]].lower()[1:-1]) > 50:
          if found_AQRR == 0:
            headline = headline.replace(word, "aquirer")
          else:
            headline = headline.replace(word, "")
          found_AQRR = 1
      except:
        continue
    if found_AQRR == 0 or found_AQRE == 0:
      count_no_AQ += 1
      continue
    for word in headline.split(" "):
      if word not in non_company:
        non_company.append(word.lower())

    #add new tag type AQE and AQR
    headline = nltk.pos_tag(nltk.word_tokenize(headline))
    for i in range(len(headline)):
      if headline[i][0] == "aquiree":
        headline[i] = tuple(["company", "AQE"])
      elif headline[i][0] == "aquirer":
        headline[i] = tuple(["company", "AQR"])
    array.append(headline)


years = list(set(year))
year_counts = [year.count(y) for y in years]
plt.bar(years, year_counts, 1)
plt.savefig('years.png')




#test/train split
train = array[:3 * len(array) // 4]
test = array[3 * len(array) // 4:]


from nltk.classify import NaiveBayesClassifier
from nltk import ngrams


#ngram features
def create_ngram_features(words, n=5):
    ngram_vocab = ngrams(words, n)
    my_dict = [(ng, True) for ng in ngram_vocab]
    my_dict = dict(my_dict)
    return my_dict

company_features = []
non_company_features = []

company = company[:len(non_company)]
for word in company:
  for sub_word in word.split(" "):
    if len(sub_word) < 4:
      continue
    company_features.append(create_ngram_features(word))
for word in non_company:
  for sub_word in word.split(" "):
    if len(sub_word) < 4:
      continue
    non_company_features.append(create_ngram_features(word))



company_features = [(word, "company") for word in company_features]
non_company_features = [(word, "non-company") for word in non_company_features]

train_set = company_features[:3 * len(company_features) // 4] + non_company_features[:3 * len(non_company_features) // 4]
test_set = company_features[3 * len(company_features) // 4:] + non_company_features[3 * len(non_company_features) // 4:]


# classifier = NaiveBayesClassifier.train(train_set)

# accuracy = nltk.classify.util.accuracy(classifier, test_set)
# print(accuracy)




#choose tagger type

from nltk.tag import brill, brill_trainer, tnt, CRFTagger, PerceptronTagger, SennaTagger
# import pycrfsuite


# tri = nltk.TrigramTagger(train)

# tnt_tagger = tnt.TnT()
# tnt_tagger.train(train)

# # crf_tagger = CRFTagger()
# # crf_tagger.train(train, 'model.crf.tagger')

# perceptron_tagger = PerceptronTagger(load=False)
# perceptron_tagger.train(train)

# brill = train_brill_tagger(perceptron_tagger, train)
# rules = brill.rules()

brill = nltk.HiddenMarkovModelTagger.train(train)
# for rule in rules:
#   print(rule.format("verbose"))





pred_prob = []
ground = []
right = 0
total = 0

#calculate accuracy on test set
for i in range(len(test)):
  test_words = [test[i][j][0] for j in range(len(test[i]))]
  test_labels = [test[i][j][1] for j in range(len(test[i]))]
  # entropy = brill.point_entropy(test_words)
  pred = brill.tag(test_words)
  pred_words= [pred[j][0] for j in range(len(pred))]
  pred_labels = [pred[j][1] for j in range(len(pred))]

  if "AQE" not in test_labels or "AQR" not in test_labels:
    continue


  AQE_non_company = [j for j in range(len(pred)) if (pred_words[j] != "company" and pred_labels[j] == "AQE")]
  AQR_non_company = [j for j in range(len(pred)) if (pred_words[j] != "company" and pred_labels[j] == "AQR")]
  
  pred_labels = [pred_labels[j] if (j not in AQR_non_company and j not in AQE_non_company) else "NNP" for j in range(len(pred))]




  if "AQE" in pred_labels and "AQR" not in pred_labels:
    unlabeled_company_index = [j for j in range(len(pred)) if (pred_words[j] == "company" and pred_labels[j] != "AQE")]
    if len(unlabeled_company_index) > 0:
      pred_labels[unlabeled_company_index[0]] = "AQR"
  elif "AQR" in pred_labels and "AQE" not in pred_labels:
    unlabeled_company_index = [j for j in range(len(pred)) if (pred_words[j] == "company" and pred_labels[j] != "AQR")]
    if len(unlabeled_company_index) > 0:
      pred_labels[unlabeled_company_index[0]] = "AQE"


  total += 2
  neither = 0

  if "AQE" in pred_labels and test_labels.index("AQE") == pred_labels.index("AQE"):
    right += 1
    # entropy[pred_labels.index("AQE")]
    ground.append(0)
  elif "AQE" in pred_labels:
    ground.append(1)
    neither += 1
  if "AQR" in pred_labels and test_labels.index("AQR") == pred_labels.index("AQR"):
    right += 1
    ground.append(0)
  elif "AQR" in pred_labels:
    ground.append(1)
    neither += 1



#classify novel headlines (CrunchBase)
from zipfile import ZipFile
import numpy as np

headline_to_classify = []
years = []
files = ['00a4409a-29f4-4b6d-8322-cb8eaaa2a2f2',
'0fd08174-0f9d-42aa-8984-d1231ab0b944',
'1deee14c-c43e-442d-9fa0-142c0176d6f3',
'2cb682a3-4b36-46cf-bc03-6e71713c180f',
'2d335fee-a32a-4675-bf8e-00011f133a9f',
'3d9e6930-e8c2-4827-86b0-40642303cf16',
'4c98f145-b16f-4cd4-887a-a79a3f0d4fd1',
'4ed6ea26-7a98-40f7-8811-1d48a2da345f',
'5d5b744c-059d-494c-918f-c091ac429d76',
'06bd6e27-99a5-4cca-b64f-5916613502ac',
'6dd5e0cb-6f29-406f-b9d4-7d7ecfc46f9a',
'7d8e5b48-78e2-45ba-89c9-a10931c40be6',
'7f79cb1a-d26d-4144-9b4b-2de32faee1e9',
'08e53cab-3258-428c-b059-14cf3f6def9a',
'9abda55f-e4c1-44ca-a150-098492c12d42',
'9df467ad-49b1-4f38-a1da-9e93a6d18cde',
'35aa53cb-e732-4cef-b570-8ea47b6ad8c8',
'42d25e59-c271-40ba-bf48-3e3173a4905e',
'58a41c1c-c766-4a4d-b456-40cfa5b57a79',
'95e1c955-94ae-4dcf-8456-84f3d448353c',
'182e73ea-f098-41f2-abba-17f3e5cad9f6',
'221c1e41-76ed-40b6-a5bf-2e10d1e55f53',
'328bb652-323e-46ba-a963-dd840e6ae786',
'378e856c-7703-40bc-958f-f29a39695187',
'0550eda1-96c0-4d8f-a88c-269fb9f32289',
'795ad137-d549-413d-85d0-1f5f084dcbfa',
'831ff016-df84-42b4-8b43-9cb73b88a236',
'900f7e82-6a60-463d-9eaa-42988af60781',
'918b4fa5-6bf1-4e5f-b435-f221287a15d9',
'1119f578-80ed-46c8-8de4-c3255de3e1e7',
'1666cce8-bbb6-4047-9ac9-833d46102f8d',
'2356fd67-9297-4814-80bf-efe83e0ec77f',
'4391ab6e-e44f-4dcb-9cda-60cfcda0ec26',
'6501d061-92f8-4203-9ba4-13b8dc347ad3',
'9624e9d0-654d-441f-a387-45df41472756',
'57966c25-33c5-425c-86d1-6ebc603efbe4',
'73007ddc-7e98-4902-8aea-a7493b052b58',
'907836fa-c8b8-4f2c-847a-0de93509ed26',
'2688731f-bf75-4b6e-8551-2c67b4d1514b',
'37138040-d876-4a6c-83b3-d7a82d59a9f7',
'80897898-49f4-40e5-a14a-0689f0589a06',
'a51abd73-4880-4d08-bbf1-7b89f2f8bfcc',
'a289e97f-8d1f-4d37-ac1f-be74e50e27b0',
'a74737c1-fa15-47c4-9505-6b21a1515b5d',
'a361643b-3d2c-4bb5-b3de-5c07aef19f0a',
'ab7690f0-2e8b-4660-a790-109f79ec10fc',
'ad14a20c-8643-4446-b3ee-394603efd40a',
'ae3c1677-34da-47c1-8083-3931373e4de5',
'afe1ef94-1a78-4eeb-852a-1f2d615befb0',
'b6bcb09b-567e-41bd-ab17-6c65998495cb',
'b664faf9-7405-4bff-9ee0-5af450a96def',
'bc472597-7815-4a52-bc83-7676866d5ca9',
'be5a55fe-99da-41b7-b8ed-e2485cb7069c',
'c2dbd97e-f508-409e-93e2-c07effd35b18',
'c9de348d-62a8-488a-b4ee-a5d7d05877c8',
'c37fe7f3-49d0-4d7d-a54a-c76d2d356519',
'c82d3ba9-f067-42da-8112-98143361c461',
'd0250d02-8e27-48fd-983d-cd2b83c83111',
'da67db3b-dca8-4fcc-afc1-616e6b7fd8f0',
'dbf8cd8c-95f3-4e25-957c-77759ce6d7f4',
'e4b87aa3-0741-436a-ba8a-420700e450fb',
'e90e0d5a-8292-4e25-aa62-fe62d22a175a',
'ea0d6464-dbcc-4994-940f-28e76700c767',
'ee37cc00-bb06-4285-b502-444a7120dd28',
'f5e295a6-270b-4457-a25d-7cf05a8ad88c',
'f5207717-5f8f-4a66-8514-be0a4070424f',
'f7055551-e503-4cb1-9b8f-eae9433fe564']



#read headlines
import re
tags = []
day = []
month = []
year = []
headline = []
id_headline = []
from collections import Counter
with ZipFile('157560.zip') as z:
    for filename in z.namelist():
        if not os.path.isdir(filename):
            # read the file
            with z.open(filename) as f:
                lines = f.readlines()
            i = 0
            for line in lines:
              line = str(line)
              if "hedline" in line:
                headline_to_classify.append(line[line.index("hedline><nitf:hl1>"):line.index("</nitf:hl1></nitf:hedline>", line.index("hedline") + 1) - 22])
              if 'year="' in line:
                years.append(line[line.index('year="') + 6: line.index('year="') + 10])



#calculate GLOVE embedding
sum = i
embeddings_dict = {}
with open("glove.6B.50d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector



#pre-process headlines
import statistics 
import numpy
import string 
headline_to_classify = []
for i in range(2):
  accuracy = []
  num = []
  non_company = []
  year = []
  count_too_short = 0 
  count_no_AQ = 0
  array = []

  with open('dump.txt') as f:
      lines = f.readlines()

  companies_appearing = []
  for line in lines:
    line = line[1:-1]
    rows = line.split("),(")
    for row in rows:
      row = re.split("','", row)
      row[0] = row[0].split("'")[-1]
      row[1] = row[1].split("'")[0]
      headline = row[-3].translate(str.maketrans('','', string.punctuation))
      if len(headline.split(" ")) < 3 or len(headline.split(" ")) > 12:
        count_too_short += 1
        continue
      for i in range(10):
        try:
          if int(row[i][:4]) < 2022 and int(row[i][:4]) > 1990:
            year.append(int(row[i][:4]))
        except:
          continue
      array.append(headline)



#calculate score based on GLOVE embeddings of aquisition headlines
  average = []
  for hl in array:
    for word in hl.split():

      try:
        average.append(list(embeddings_dict[word.lower()]))
      except:
        continue
  average = [*zip(*average)]
  ave = []
  for i in average:
    ave.append(statistics.mean(i))


  def sortSecond(val):
    sum = []
    for word in val.split():
      try:
        sum.append(embeddings_dict[word])
      except:
        continue
    if len(sum) < 1:
      return 1000000
    try:
      return numpy.linalg.norm(np.array(sum) - np.array(ave))
    except:
      return 1000000

  headline_to_classify += list(set(headline_to_classify))
  headline_to_classify.sort(key=sortSecond)
  headline_to_classify = headline_to_classify[:1000]



#save 200 headlines that are closest to GLOVE embedding of headlines
with open('headlines.txt', 'w') as f:
  top = headline_to_classify[:200]
  for head in top:
    f.write(head + "\n")