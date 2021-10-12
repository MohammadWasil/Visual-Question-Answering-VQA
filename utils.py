# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 03:22:20 2021

@author: wasil
"""

import sys, os
import operator
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle
from nltk import word_tokenize
import scipy as sc
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

DATA = "Data/"
COCO = "coco/"
DATA_INTERMEDIATE = "Data Files"
PROCESSED_TRAINING_DATA = "Processed Training Data"

def get_question_tokenizer(types):
    data_path = "Training Data QA.pickle"
    data_path_val = "Validation Data QA.pickle"

    if ( types == "small"):
        num_data = 100
        num_data_val = 20
    elif (types == "full"):
        num_data = 248349
        num_data_val = 121512

    df = pd.read_pickle(os.path.join(DATA_INTERMEDIATE, data_path))
    df_val = pd.read_pickle(os.path.join(DATA_INTERMEDIATE, data_path_val))
    questions = df['questions'].values.tolist()
    questions_val = df_val['questions'].values.tolist()
   
    all_question = questions + questions_val
  
    tokenizer = Tokenizer(num_words = 10000)
    tokenizer.fit_on_texts(all_question)

    word_index = tokenizer.word_index

    # Save the tokenizer, so that we can use this tokenizer whenever we need to predict any reviews.
    with open(os.path.join(PROCESSED_TRAINING_DATA, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #tokenising train data
    train_question_tokenized = tokenizer.texts_to_sequences(questions)      
    questions = pad_sequences(train_question_tokenized, maxlen = 25)          # len(X_train) x 25

    #tokenising validation data
    val_question_tokenized = tokenizer.texts_to_sequences(questions_val)
    questions_val = pad_sequences(val_question_tokenized, maxlen = 25)               # len(X_val) X 25 

    return questions[0:num_data], questions_val[0: num_data_val], word_index
  
def get_questions_matrix(split):
  
    if split == 'train':
        data_path = 'data_train_qa.pickle'
    elif split == 'val':
        data_path = 'data_val_qa.pickle'
    else:
        print('Invalid split!')
        sys.exit()

    df = pd.read_pickle(os.path.join(DATA_INTERMEDIATE, data_path))
    questions = df[['questions']].values.tolist()
    word_idx = load_idx()
    seq_list = []

    for question in questions:
        words = word_tokenize(question[0])
        seq = []
        for word in words:
            seq.append(word_idx.get(word,0))
        seq_list.append(seq)
    question_matrix = pad_sequences(seq_list)
  
    question_matrix.astype('int32')
    return question_matrix

def int_to_answers():
    data_path = 'Training Data QA.pickle'
    
    df = pd.read_pickle(os.path.join(DATA_INTERMEDIATE, data_path))
    answers = df['multiple_choice_answer'].values.tolist()
    freq = defaultdict(int)
    for answer in answers:
        freq[answer[0].lower()] += 1
    int_to_answer = sorted(freq.items(),key=operator.itemgetter(1),reverse=True)[0:1000]
    int_to_answer = [answer[0] for answer in int_to_answer]
    return int_to_answer

top_answers = int_to_answers()

def answers_to_onehot():
	top_answers = int_to_answers()
	answer_to_onehot = {}
	for i, word in enumerate(top_answers):
		onehot = np.zeros(1001)
		onehot[i] = 1.0
		answer_to_onehot[word] = onehot
	return answer_to_onehot
	
answer_to_onehot_dict = answers_to_onehot()

def get_answers_matrix(split, types):
  
    if split == 'train':
        data_path = 'Training Data QA.pickle'
        if ( types == "small"):
            num_data = 300
        elif (types == "full"):
            num_data = 2483490 

    elif split == 'val':
        data_path = 'Validation Data QA.pickle'
        if (types == "small"):
            num_data = 60
        elif (types == "full"):
            num_data = 1215120 
    else:
        print('Invalid split!')
        sys.exit()
     
    df = pd.read_pickle(os.path.join(DATA_INTERMEDIATE, data_path))
    answers = df['multiple_choice_answer'].values.tolist()
    answer_matrix = np.zeros((len(answers),1001))
    default_onehot = np.zeros(1001)
    default_onehot[1000] = 1.0

    for i, answer in enumerate(answers):
        answer_matrix[i] = answer_to_onehot_dict.get(answer[0].lower(),default_onehot)
	
    answer_matrix.astype('int32')
    return answer_matrix[0:num_data]

def get_coco_features(split, types ):
    
    if split == 'train':
        data_path = 'Training Data QA.pickle'
        if ( types == "small"):
            num_data = 100
        elif (types == "full"):
            num_data = 82783

    elif split == 'val':
        data_path = 'Validation Data QA.pickle'
        if (types == "small"):
            num_data = 20
        elif (types == "full"):
            num_data = 40504
    else:
        print('Invalid split!')
        sys.exit()
  
    id_map_path = 'coco_vgg_id_map.txt'
    features_path = 'vgg_feats.mat'
    img_labels = pd.read_pickle(os.path.join(DATA_INTERMEDIATE, data_path))[['image_id']].drop_duplicates().values.tolist()
    img_ids = open(os.path.join(DATA_INTERMEDIATE, id_map_path)).read().splitlines()
    features_struct = sc.io.loadmat(os.path.join(DATA, COCO, features_path))

    id_map = {}
    for ids in img_ids:
        ids_split = ids.split()
        id_map[int(ids_split[0])] = int(ids_split[1])

    VGGfeatures = features_struct['feats']
    nb_dimensions = VGGfeatures.shape[0]
    nb_images = len(img_labels)
    image_matrix = np.zeros((nb_images,nb_dimensions))

    for i in range(nb_images):
        image_matrix[i,:] = VGGfeatures[:,id_map[img_labels[i][0]]]  
    image_matrix.astype('float32')
    return image_matrix[0:num_data]

def loadGloveModel(gloveFile, word_index):
    print("Loading Glove Model")
    f = open(os.path.join(DATA_INTERMEDIATE, "glove.840B.300d", gloveFile),'r', encoding='utf8')
    embedding_index = {}
    print("Opened!")
    for j, line in enumerate(f):
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        embedding_index[word] = embedding
    
    print("Done.",len(embedding_index)," words loaded!")
  
    # Now, we need to create embedding matrix.
    EMBEDDING_DIM = 300
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    print(embedding_matrix.shape)
  
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix