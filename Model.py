# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 02:30:47 2021

@author: wasil
"""

import tensorflow as tf
import os, pandas as pd, numpy as np
import tensorflow.keras.backend as k
import h5py, pickle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding, concatenate, Conv1D, Input, Embedding
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD, Adadelta

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.3)
#config = tf.ConfigProto(gpu_options = gpu_options)
#config.gpu_options.allow_growth=True
#session = tf.Session( config = config)
#k.clear_session()

def Model_1(img_features_train, img_features_val, question_train, question_val, answer_train, answer_val, embedding_matrix):
        
    NAME = "1_model_one_lstm_vgg"
    
    val_loss = []
    val_acc = []
    loss = []
    acc = []
    
    dropout_rate = 0.5
    print ("Creating text model...")
    question_input = Input(shape=(25, ))
    
    #x = Embedding(output_dim=512, input_dim=125, input_length=25)(question_input)
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights = [embedding_matrix], input_length=25, trainable = False)(question_input)
    lstm_1 = LSTM(units=512, return_sequences=False)(x)
    dropout__ques_1 = Dropout(dropout_rate)(lstm_1)
    #lstm_2 = CuDNNLSTM(units=512, return_sequences=False)(dropout__ques_1)
    #dropout__ques_2 = Dropout(dropout_rate)(lstm_2)
    
    dense_ques_1 = Dense(1024, activation='tanh')(dropout__ques_1)
    print ("Creating image model...")
    
    image_input = Input(shape=(4096, ) )
    #reshape = reshape((4096,))(image_input)
    dense_img_1 = Dense(1024,  activation='relu')(image_input)
    
    print ("Merging final model...")
    
    concatenate_1 = concatenate([dense_img_1, dense_ques_1])
    
    dropout_1 = Dropout(0.5)(concatenate_1)
    dense_1 = Dense(1001, activation = "tanh")(dropout_1)
    dropout_2 = Dropout(0.5)(dense_1)    
    dense_2 = Dense(1001, activation = "softmax")(dropout_2)
    
    model_1 = Model(inputs=[image_input, question_input], outputs=dense_2)
    opt = SGD(lr = 0.01)
    model_1.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    model_1.summary() 
    
    history = model_1.fit([img_features_train,question_train ],answer_train, epochs = 50, shuffle = True, 
                validation_data = [[img_features_val, question_val], answer_val])
    
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    acc = history.history['accuracy']
    models = ["1_model_one_lstm_vgg" for i in range(50)]
    model_metric_1 = pd.DataFrame(np.column_stack([models, acc, loss, val_acc, val_loss]), 
                                   columns=['models', 'accuracy', 'loss', 'validation_accuracy', 'validation_loss'])
    model_metric_1.to_csv("model_metric_1.csv")
    
    model_1.save('1_model_one_lstm_vgg.h5')
    
