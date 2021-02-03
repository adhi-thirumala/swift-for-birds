#!/usr/bin/env python
# coding: utf-8

# In[8]:


#importing the necessary modules
import shutil
from imutils import paths
from random import shuffle
import os
from datetime import datetime
import numpy as np


# In[9]:


#Start by getting the ID of all animals that were photographed
#In this example each picture is named as: TagID_date_time.jpg (e.g. "01103F7D5A_2018-11-26_07-56-03.jpg")
#It is possible to split each picture name by "_" and get the first element which
#corresponds to the tagID

#get the path to the dataset
Dataset="/dev/sda1/Swift Project Images/"
#list all pictures in the dataset
imagePaths = sorted(list(paths.list_images(Dataset)))
#create an empty list to store all the individuals ID
Individuals=[]
#loop through all pictures, split by "_", get the first element and append to the Individuals ID list
for i in range(0, len(imagePaths)):
    if imagePaths[i].split("/")[-1:][0].split("_")[0] not in Individuals:
        Individuals.append(imagePaths[i].split("/")[-1:][0].split("_")[0])


# In[11]:


#after listing all individuals create two empty folders for each individual 
#one for the training and another for the validaiton dataset

#define the folder were the training and validation datasets will be placed
root_dir="/dev/sda1/actualting/"

#loop through all individuals and create a folder for the training dataset
# and a folder for the validation dataset
for i in range(0, len(Individuals)):
    train_dir=root_dir+"/train/"+Individuals[i]#variable with the full path of the training folder
    val_dir=root_dir+"/val/"+Individuals[i]#variable with the full path of the validation folder
    if not os.path.exists(train_dir):#condition for if the folder already exists
        os.makedirs(train_dir)#create the folder
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)   


# In[12]:


#in this example we are going to select 900 pictures for training and 100 pictures for validation.
#We setup the condition that the training and the validation pictures should be taken on different days
#in order to avoid having pictures that are very similiar in both datasets, which 
#could result in overfitting the CNN
#For the same reason we also limit a maximum of 25 pictures per day in order to promote
#variation in the pictures (e.g. different weather conditions)

#define the number of validation pictures and the number of training pictures
N_val_pics=459
N_train_pics=1836

#create two empty lists to store the pictures files that are going to be movedto the training 
#and validation fodlers
training_pictures=[]
validation_pictures=[]

#loop through each individual









            
    #shuffle the dates and the images directory to select random images from each day
    shuffle(days)
    shuffle(imagePaths_individual)


    



















            
    #randomly threshold the list of training and validation dateset
    #to have only the number of picures needed (in this example 900 and 100)
    shuffle(validation_pictures)
    shuffle(training_pictures)
    validation_pictures=validation_pictures[0:N_val_pics]
    training_pictures=training_pictures[0:N_train_pics]
    
    #loop through the list of pictures
    #move the pictures files to the validation folder
    for i in range(0, N_val_pics + 1):
        #get the picture name (e.g. "01103F7D5A_2018-11-26_07-56-03.jpg")
        val_file_name=validation_pictures[i].split(os.path.sep)[-1]
        #create a variable with the directory and the name of the pictures file
        output_name_val=root_dir+"/val/"+individual+"/"+val_file_name
        #move the file
        shutil.move(validation_pictures[i], output_name_val)
        
    #move the files to the training folder
    for i in range(0, len(training_pictures)):
        train_file_name=training_pictures[i].split(os.path.sep)[-1]
        output_name_train=root_dir+"/train/"+individual+"/"+train_file_name
        shutil.move(training_pictures[i], output_name_train)
        
    #before passing to the next individual empty the pictures lists
    validation_pictures=[]
    training_pictures=[]


# In[ ]:




