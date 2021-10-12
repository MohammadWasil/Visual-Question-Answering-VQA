# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 00:18:33 2021

@author: wasil
"""

# Load json question file.
import os
import re
import json
#import yaml
import pickle
import scipy.io
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

import h5py


import nltk
nltk.download("punkt")

from DataDownloader import Download_Data_extract, Download_VGG16_Weights, Download_COCO_Image, Download_extract_word_embedding
from utils import get_question_tokenizer, get_answers_matrix, get_coco_features, loadGloveModel
from Model import Model_1
# directories:
DATA = "Data/"
COCO = "coco/"
DATA_INTERMEDIATE = "Data Files"
PROCESSED_TRAINING_DATA = "Processed Training Data"

Download_Data_extract()

print("Loading files...")
with open(os.path.join(DATA, 'OpenEnded_mscoco_val2014_questions.json')) as file:
    open_question_json_val = json.load(file)
    
with open(os.path.join(DATA, 'OpenEnded_mscoco_train2014_questions.json')) as file:
    open_question_json_train = json.load(file)

with open(os.path.join(DATA, 'mscoco_val2014_annotations.json')) as file:
    annotation_json_val = json.load(file)
    
with open(os.path.join(DATA, 'mscoco_train2014_annotations.json')) as file:
    annotation_json_train = json.load(file)

print("Processing Textual Data...")
# Preprocess openended_question_json file
keys_to_remove = ["data_subtype", "data_type", "info", "license", "task_type"]
for key in keys_to_remove:
    open_question_json_val.pop(key, None)
    open_question_json_train.pop(key, None)

df_question_val = pd.DataFrame.from_dict(open_question_json_val["questions"], orient = "columns")
df_question_train = pd.DataFrame.from_dict(open_question_json_train["questions"], orient = "columns")

#Remove keys from answers json file.
keys_to_remove = ["data_subtype", "data_type", "info", "license"]
for key in keys_to_remove:
    annotation_json_val.pop(key, None)
    annotation_json_train.pop(key, None)
    
df_annotation_val = pd.DataFrame.from_dict(annotation_json_val["annotations"], orient = "columns")
df_annotation_train = pd.DataFrame.from_dict(annotation_json_train["annotations"], orient = "columns")

df_annotation_val = df_annotation_val.drop(columns = ["answer_type", "answers", "image_id", "question_type"])
df_annotation_train = df_annotation_train.drop(columns = ["answer_type", "answers", "image_id", "question_type"])

data_val_qa = pd.merge(df_question_val, df_annotation_val, how = "outer", on = "question_id")
data_train_qa = pd.merge(df_question_train, df_annotation_train, how = "outer", on = "question_id")

data_val_qa = data_val_qa.drop( columns = ["question_id"])
data_train_qa = data_train_qa.drop( columns = ["question_id"])


# Before saving the file, lets pre-process the questions.
# Find all the punctuations.
punctuation = []
length = int(len(data_train_qa))
for i in range(length):
    punctuation.append(re.findall(r"[^a-zA-Z0-9 ]", data_train_qa["question"][i]))

# Get all unique punctuation from Top 200 questions
myset = set()
for i in range(200):
    for j in range(len(punctuation[i])):
        myset.add(punctuation[i][j])
# print(myset) {'?', "'", ','}

#So, the punctuation we are going to foucs is on "'", while for the rest we are going to simply remove it.
question_punc = []
length = int(len(data_train_qa))
for i in range(length):  
    punctuation = re.findall(r"[']", data_train_qa["question"][i])
    if punctuation:
        question_punc.append([i, data_train_qa["question"][i]])
   
#question_punc[0:10]        

print("Creating Contraction list...")
# what's -> What is, we are going to use wikipedia data.
website_url = requests.get("https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions").text
soup = BeautifulSoup(website_url, 'lxml')

# we need "wikitable sortable" file from the html script.
table = soup.find('table', {'class' : 'wikitable sortable'})
#columns = soup.find_all('td')
contraction_list = []
meaning_list = []
for row in table.find_all('tr')[1:]:  # iterate over the rows, starting from 
                                      # the second (first one is the header row)
    contraction = row.find_all('td')[0]  #  the Symbol col is the first <td> in every row 
    meaning = row.find_all('td')[1]  #  the Symbol col is the first <td> in every row
        
    contraction_list.append(contraction.text)
    meaning_list.append(meaning.text)

# Preprocess the words
#splitword = []
#print(len(contraction_list))
#print(len(meaning_list))
for word in contraction_list:
  #print(word)
  if (' ' in word):
    position_to_add = contraction_list.index(word)
    meaning_word = meaning_list[position_to_add]
    
    word_to_process = re.sub(r'[(),]', "", word)
    splitword = word_to_process.split()
    length_split_word = len(splitword)
    #print(splitword)
    
    contraction_list.remove(word)
    meaning_list.pop(position_to_add)
    
    for i in range(len(splitword)):
        contraction_list.insert((position_to_add + i), splitword[i] )
        meaning_list.insert((position_to_add + i), meaning_word)
    break
    #print(splitword)
    
# pop out last two element
#del contraction_list[-2:]
#del meaning_list[-2:]

meaningfull_list = []
def process_word(word):
  
    word = re.sub(r"[0-9]", "", word)
    word = re.sub(r"\n", "", word)
    word = word.replace('[','').replace(']','')
    word = word.replace(' (colloquial)', '')
    word = word.replace(' (archaic)', '')
    word = word.replace(' (colloquial/Southern American English)', '')
    meaningfull_list.append(word)
  
    return meaningfull_list
#[re.sub(r'[0-9]', '', word) if ('[]' in word) else x for x in meaning_list]

for word in meaning_list:
    meaningfull_list = process_word(word)

# Now that we have process the words, lets split these too!

for word in meaningfull_list:
    if (' / ' in word or ", " in word ):
        position_to_split = meaningfull_list.index(word)

        word_to_process = re.sub(r' / ', ",", word)
        splitword = word_to_process.split(',')
        length_split_word = len(splitword)

        meaningfull_list.remove(word)
        meaningfull_list.insert((position_to_split ), splitword[0] ) # For simplicity lets add only 1st word.

# FOr "What are/what were" -> to "what are"
meaningfull_list_2 = []
for word in meaningfull_list:
  
    for alpha in word:
        if("/" in alpha):
            word = word.split("/")[0]
    meaningfull_list_2.append(word)  

another_list = []
for words in meaningfull_list_2:
    split_word = words.split(" ")
  
    if("has" in split_word):
        split_word = [word.replace("has", "is") for word in split_word]   
    another_list.append(" ".join(split_word))

#print(len(another_list) )
#print(len(meaningfull_list_2) )
dictionary = {k: v for k,v in zip(contraction_list, another_list)}
#del dictionary['']

dictionary["hasn't"] = "has not"
dictionary["whatcha"] = " ".join(dictionary["whatcha"].split(" ")[0:3])
# Step by step output is available on : https://github.com/MohammadWasil/Quora-Insincere-Question-Classification/blob/master/Quora_Insincere_Questions.ipynb

print("Contraction list created!")

print("Processing questions...")
# We have numbers here, we are going to convert it to english names.
numbers = []
length = int(len(data_train_qa))
for i in range(length):
    no = re.findall(r"[0-9]", data_train_qa["question"][i])
    if no:
        numbers.append(no)

# Get all unique punctuation from Top 200 questions
mysetnumber = set()
for i in range(600):
    for j in range(len(numbers[i])):
        mysetnumber.add(numbers[i][j])
#print(mysetnumber)
        
number_dictionary = {"1" : "one", "2" : "two", "3": "three", "4" : "four", "5": "five", "6": "six", "7" : "seven", "8": "eight", "9":"nine", "0":"zero"}

# Review the questions:
def mappingWords(questions,dictionary):
    return " ".join([dictionary.get(w,w) for w in questions.split()])

def mappingNumber(questions, number_dictionary):
    return " ".join([number_dictionary.get(w,w) for w in questions.split()])

def review_question(questions):
    questions = questions.lower()
    questions = mappingWords(questions, dictionary)
    questions = mappingNumber(questions, number_dictionary)
    questions = re.sub(r"[^A-Za-z0-9 ]", "", questions)
    return questions

train_question_list = []
validation_question_list = []
for i in range(len(data_train_qa["question"])):
    train_question_list.append(review_question(data_train_qa["question"][i]))
    
for i in range(len(data_val_qa["question"])):
    validation_question_list.append(review_question(data_val_qa["question"][i]))


df_train_questions = pd.DataFrame(train_question_list, columns = ["questions"])
df_val_questions = pd.DataFrame(validation_question_list, columns = ["questions"])

validation = pd.concat([data_val_qa, df_val_questions], axis = 1)
training   = pd.concat([data_train_qa, df_train_questions], axis = 1)

data_val_qa = validation.drop( columns = ["question"])
data_train_qa = training.drop( columns = ["question"])

data_train_qa = data_train_qa.reindex(columns = ["image_id", "questions", "multiple_choice_answer"])
data_val_qa = data_val_qa.reindex(columns = ["image_id", "questions", "multiple_choice_answer"])


print("Processing Done!")
print("Saving...")
with open(os.path.join(DATA_INTERMEDIATE, 'Training Data QA.pickle'), 'wb') as train:
    pickle.dump(data_train_qa, train, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(DATA_INTERMEDIATE, 'Validation Data QA.pickle'), 'wb') as val:
    pickle.dump(data_val_qa, val, protocol=pickle.HIGHEST_PROTOCOL)
    
print(data_train_qa.head())
print(data_val_qa.head())

# create tokenized question data
small_question_train_tokenize, small_question_val_tokenize, word_idx = get_question_tokenizer(types = "small")
# ((10000, 25), (2000, 25))

# save these tokenized questions.
h5f = h5py.File(os.path.join(PROCESSED_TRAINING_DATA, 'small_question_train_tokenize.h5'), 'w')
h5f.create_dataset('small_question_train_tokenize', data=small_question_train_tokenize)
h5f.close()

h5f_val = h5py.File(os.path.join(PROCESSED_TRAINING_DATA, 'small_question_val_tokenize.h5'), 'w')
h5f_val.create_dataset('small_question_val_tokenize', data=small_question_val_tokenize)
h5f_val.close()

file = open(os.path.join(PROCESSED_TRAINING_DATA, "word_idx.pickle"), "wb")
pickle.dump(word_idx, file)
file.close()

# Tokenize the annotations
print('Loading answers ...')
small_answers_train = get_answers_matrix('train', types = "small") # float64
small_answers_val = get_answers_matrix('val', types = "small")

h5_ans = h5py.File(os.path.join(PROCESSED_TRAINING_DATA,'small_answers_train.h5'), 'w')
h5_ans.create_dataset('small_answers_train', data = small_answers_train)
h5_ans.close()

h5_ans_val = h5py.File(os.path.join(PROCESSED_TRAINING_DATA, 'small_answers_val.h5'), 'w')
h5_ans_val.create_dataset('small_answers_val', data = small_answers_val)
h5_ans_val.close()
# ((30000, 1001), (6000, 1001))

# Load Word Embedding
print("Word Embedding...")
Download_extract_word_embedding()

gloveFile = 'glove.840B.300d.txt'
file = open(os.path.join(PROCESSED_TRAINING_DATA, "word_idx.pickle"), "rb")
word_idx = pickle.load(file)
file.close()

embedding_matrix_tokenize = loadGloveModel(gloveFile, word_idx) # (16110, 300)

file = open(os.path.join(PROCESSED_TRAINING_DATA, "embedding_matrix_tokenize.pickle"), "wb")
pickle.dump(embedding_matrix_tokenize, file)
file.close()

h5_feats = h5py.File(os.path.join(PROCESSED_TRAINING_DATA, 'embedding_matrix_tokenize.h5'), 'w')
h5_feats.create_dataset('embedding_matrix_tokenize', data = embedding_matrix_tokenize)
h5_feats.close()

# Now let move on to image analysis.
# COCO image dataset
print("Analysing COCO Image Dataset...")
print(" VGG16 trained images, which have been used in Visual Qa datasets. Since we are going to use VGG16 architecture, so we wont be training all the images, instead we will be using the image feature matrix from stanford, which have been already trained on VGG16 architecture. All these images are present in the training and validation data.")

#Download_VGG16_Weights()

with open(os.path.join(DATA, COCO, 'dataset.json')) as f:
    data = json.load(f)

#len(data["images"])   # This number of images is equal to the number of images in training and validation data.
#len(data_train_qa)/3 + len(data_val_qa)/3

# Lets create a relationaship between this coco data and trainnig data!
#data["images"][0]["cocoid"]

print("creating coco_vgg_id_map file(Used later on)")
map_list = [ [data["images"][i]["cocoid"], i]  for i in range(len(data["images"]))]

datafrm = pd.DataFrame(map_list)
datafrm.to_csv(os.path.join(DATA_INTERMEDIATE, 'coco_vgg_id_map.txt'), header = None, index = None, sep=' ', mode = 'a')
print("Created and Saved!")

print("Download image datasets!")
print("This might take some time... a lot of time perphas. :-|")
#Download_COCO_Image()
print("Downloading and Unzipping completed!")

# COCO feature matrix
#mat = scipy.io.loadmat(os.paht.join(DATA, COCO, '/vgg_feats.mat'))
#mat["feats"].shape (4096, 123287)

"""
vgg_feature matrix are the features of images which have been fed into vgg16 architecture, with 16 layers of convolutional neural network and Dense layer(with 4096 units), except the last dense layer(with 1000 units). So each of the image has beeen fed into the model and which gave us the feature vector of each image, in the shape of 4096 X 1. Concatenating each feature vector of every image, give us the matrix of shape 4096 X 123287. Since, 82783 + 40504 = 1232287 (Training and valiation data).

From coco_vgg_id_map.txt file, we have image_id in order of the columns of vgg_feats matrix. Selecting top 10 image id from coco_vgg_id_map file, means selecting top 10 columns from vgg_feats matix.
"""

print('Loading image features ...')
small_img_features_train = get_coco_features('train', types = "small")
small_img_features_val = get_coco_features('val', types = "small")

h5_feats = h5py.File(os.path.join(PROCESSED_TRAINING_DATA, 'small_img_features_train.h5'), 'w')
h5_feats.create_dataset('small_img_features_train', data = small_img_features_train)
h5_feats.close()

h5_feats_val = h5py.File(os.path.join(PROCESSED_TRAINING_DATA, 'small_img_features_val.h5'), 'w')
h5_feats_val.create_dataset('small_img_features_val', data = small_img_features_val)
h5_feats_val.close()
# ((2000, 4096), (10000, 4096))


#### training.
h5_img = h5py.File(os.path.join(PROCESSED_TRAINING_DATA, 'small_img_features_train.h5'), 'r')
img_features_train = h5_img['small_img_features_train'][:]

h5_img = h5py.File(os.path.join(PROCESSED_TRAINING_DATA,'small_img_features_val.h5'), 'r')
img_features_val = h5_img['small_img_features_val'][:]

h5_ans = h5py.File(os.path.join(PROCESSED_TRAINING_DATA,'small_answers_train.h5'), 'r')
answer_train = h5_ans['small_answers_train'][:]
h5_ans.close()

h5_ans = h5py.File(os.path.join(PROCESSED_TRAINING_DATA,'small_answers_val.h5'), 'r')
answer_val   = h5_ans['small_answers_val'][:]
h5_ans.close()

h5_que = h5py.File(os.path.join(PROCESSED_TRAINING_DATA,'small_question_train_tokenize.h5'), 'r')
question_train = h5_que['small_question_train_tokenize'][:]

h5_que = h5py.File(os.path.join(PROCESSED_TRAINING_DATA,'small_question_val_tokenize.h5'), 'r')
question_val = h5_que['small_question_val_tokenize'][:]

h5_que = h5py.File(os.path.join(PROCESSED_TRAINING_DATA,'embedding_matrix_tokenize.h5'), 'r')
embedding_matrix = h5_que['embedding_matrix_tokenize'][:]

img_features_train = np.repeat(img_features_train, 3, 0)
img_features_val = np.repeat(img_features_val, 3, 0)

question_train = np.repeat(question_train, 3, 0)
question_val = np.repeat(question_val, 3, 0)


Model_1(img_features_train, img_features_val, question_train, question_val, answer_train, answer_val, embedding_matrix)