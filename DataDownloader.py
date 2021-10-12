# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 01:00:17 2021

@author: wasil
"""
import os
import zipfile

# directories:
DATA = "Data/"
COCO = "coco/"
DATA_INTERMEDIATE = "Data Files"

def Download_Data_extract():
    print("Downloading training and validation data")
    
    if(os.path.isfile(os.path.join('Annotations_Train_mscoco.zip')) == False):
        os.system("wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip")
    
    if(os.path.isfile(os.path.join('Questions_Val_mscoco.zip')) == False):
        os.system("wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip")
        
    if(os.path.isfile(os.path.join('Annotations_Val_mscoco.zip')) == False):
        os.system("wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip")
        
    if(os.path.isfile(os.path.join('Questions_Train_mscoco.zip')) == False):
        os.system("wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip")
        
    print("Unzipping files...")
    
    if(os.path.isfile(os.path.join(DATA, 'mscoco_val2014_annotations.json')) == False): 
        with zipfile.ZipFile("Annotations_Val_mscoco.zip", 'r') as zip_ref:
            zip_ref.extractall(DATA)
    
    if(os.path.isfile(os.path.join(DATA, 'OpenEnded_mscoco_val2014_questions.json')) == False) or (os.path.isfile(os.path.join(DATA, 'MultipleChoice_mscoco_val2014_questions.json')) == False):
        with zipfile.ZipFile("Questions_Val_mscoco.zip", 'r') as zip_ref:
            zip_ref.extractall(DATA)
    
    if(os.path.isfile(os.path.join(DATA, 'mscoco_train2014_annotations.json')) == False): 
        with zipfile.ZipFile("Annotations_Train_mscoco.zip", 'r') as zip_ref:
            zip_ref.extractall(DATA)
    
    if(os.path.isfile(os.path.join(DATA, 'OpenEnded_mscoco_train2014_questions.json')) == False) or (os.path.isfile(os.path.join(DATA, 'MultipleChoice_mscoco_train2014_questions.json')) == False):
        with zipfile.ZipFile("Questions_Train_mscoco.zip", 'r') as zip_ref:
            zip_ref.extractall(DATA)
        
def Download_VGG16_Weights():
    print("Downloading VGG16 weights on COCO dataset")
    if (os.path.isfile(os.path.join('coco.zip')) == False):
        os.system('wget http://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip')
        print("Downloaded!")
    else:
        print("Already downloaded!")
    
    # The files we need. 
    if (os.path.isfile(os.path.join(DATA, COCO, "dataset.json")) == False) or (os.path.isfile(os.path.join(DATA, COCO, "vgg_feats.mat")) == False):
        print("Unzipping...")
        with zipfile.ZipFile("coco.zip", 'r') as zip_ref:
            zip_ref.extractall(DATA)
        print("Unzipping done!")
    else:
        print("Already unzipped")
        
def Download_COCO_Image():
    
    #if (os.path.isfile(os.path.join('train2014.zip')) == False):
    #    os.system('wget http://images.cocodataset.org/zips/train2014.zip') # 13Gigs file.
    
    if (os.path.isfile(os.path.join('val2014.zip')) == False):
        os.system('wget http://images.cocodataset.org/zips/val2014.zip') # 6 Gigs file.
    else:
        print("COCO Validation data already downloaded!")

    # Since training is very large, and will take long time to unzip, we are going to unzip validation data.
    # Although we dont need the image dataset, but still we are going to look at it.
    # some if statements.
    print("Unzipping validation data, since training data is quite huge...")
    if (os.path.isdir(os.path.join(DATA, "val2014")) == False):
        with zipfile.ZipFile("val2014.zip", 'r') as zip_ref:
            zip_ref.extractall(DATA)
    else:
        print("COCO Validation data already unzipped!")
            
def Download_extract_word_embedding():
    
    
    if (os.path.isfile(os.path.join('glove.840B.300d.zip')) == False):
        print("Downloading Glove word embdding")
        print("This might be slow")
        os.system("http://nlp.stanford.edu/data/glove.840B.300d.zip")
    else:
        print("Already downloaded GloVe Embedding")
    
    if (os.path.isfile(os.path.join(DATA_INTERMEDIATE, "glove.840B.300d", "glove.840B.300d.txt")) == False):
        with zipfile.ZipFile("glove.840B.300d.zip", 'r') as zip_ref:
            zip_ref.extractall(DATA_INTERMEDIATE)
    else:
        print("GloVe Word EMbedding already unzipped!")
        
        
        
        
        
        